# train.py
"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘

CLI ABLATION FLAGS (Part 2):
  --scheduler   {noam, fixed}          → ablation 2.1
  --use_scale   {true, false}          → ablation 2.2  (√dk scaling)
  --pos_encoding {sinusoidal, learned} → ablation 2.4
  --smoothing   float (0.0 or 0.1)    → ablation 2.5

FIXES vs original:
  1. run_name / ckpt_path were undefined → now derived from args.
  2. wandb artifact block now correctly references local variables.
  3. evaluate_bleu: corpus_bleu() returns 0-1; multiplied by 100.
  4. save_checkpoint: also saves warmup_steps.
  5. Gradient norm of Q/K weights logged every step when
     --log_grad_norm is set (ablation 2.2).
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε. Use 0.0 for standard CE (ablation 2.5).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : [batch * tgt_len, vocab_size]
            target : [batch * tgt_len]
        Returns:
            Scalar loss.
        """
        smooth_val = self.smoothing / (self.vocab_size - 2)   # exclude correct + pad
        dist = torch.full_like(logits, smooth_val)
        dist[:, self.pad_idx] = 0.0
        dist.scatter_(1, target.unsqueeze(1), self.confidence)

        mask = (target == self.pad_idx)
        dist[mask] = 0.0

        log_probs = torch.log_softmax(logits, dim=-1)
        loss      = -(dist * log_probs).sum(dim=-1)
        non_pad   = (~mask).sum().clamp(min=1)
        return loss.sum() / non_pad


# ══════════════════════════════════════════════════════════════════════
#  GRADIENT NORM HELPER  (ablation 2.2)
# ══════════════════════════════════════════════════════════════════════

def _log_qk_grad_norms(model: Transformer, step: int, wandb_run) -> None:
    """
    Log the L2 gradient norm of all W_q and W_k weight matrices across
    every encoder and decoder layer. Called every step during ablation 2.2.
    """
    q_norms, k_norms = [], []

    for enc_layer in model.encoder.layers:
        wq = enc_layer.self_attn.W_q.weight
        wk = enc_layer.self_attn.W_k.weight
        if wq.grad is not None:
            q_norms.append(wq.grad.norm().item())
        if wk.grad is not None:
            k_norms.append(wk.grad.norm().item())

    for dec_layer in model.decoder.layers:
        for mha in (dec_layer.self_attn, dec_layer.cross_attn):
            if mha.W_q.weight.grad is not None:
                q_norms.append(mha.W_q.weight.grad.norm().item())
            if mha.W_k.weight.grad is not None:
                k_norms.append(mha.W_k.weight.grad.norm().item())

    if q_norms:
        wandb_run.log({
            "grad_norm/W_q_mean": sum(q_norms) / len(q_norms),
            "grad_norm/W_k_mean": sum(k_norms) / len(k_norms),
            "step": step,
        })


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model:     Transformer,
    loss_fn:   nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler  = None,
    epoch_num: int  = 0,
    is_train:  bool = True,
    device:    str  = "cpu",
    log_grad_norm: bool = False,   # ablation 2.2
    wandb_run  = None,
    step_offset: int = 0,          # total steps before this epoch (for grad logging)
) -> tuple:
    """
    Run one epoch of training or evaluation.

    Returns:
        avg_loss  : float — average per-token loss.
        total_steps: int  — steps taken this epoch (training only, else 0).
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    pad_idx      = 1
    steps_taken  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} epoch {epoch_num}"):
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            B, T, V = logits.shape
            loss = loss_fn(
                logits.contiguous().view(B * T, V),
                tgt_out.contiguous().view(B * T),
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                # ablation 2.2: log Q/K gradient norms before clipping
                if log_grad_norm and wandb_run is not None:
                    _log_qk_grad_norms(model, step_offset + steps_taken, wandb_run)

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                steps_taken += 1

            non_pad      = (tgt_out != pad_idx).sum().item()
            total_loss  += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / max(total_tokens, 1), steps_taken


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model:        Transformer,
    src:          torch.Tensor,
    src_mask:     torch.Tensor,
    max_len:      int,
    start_symbol: int,
    end_symbol:   int,
    device:       str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Returns:
        ys : Generated token indices, shape [1, out_len].
    """
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys       = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model:           Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device:  str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    from dataset import sos_idx, eos_idx, pad_idx

    model.eval()
    hypotheses = []
    references = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval"):
            src = src.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)
                src_mask = make_src_mask(src_i, pad_idx).to(device)

                out = greedy_decode(
                    model, src_i, src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                )

                pred_ids = out[0].tolist()
                if sos_idx in pred_ids:
                    pred_ids = pred_ids[pred_ids.index(sos_idx) + 1:]
                if eos_idx in pred_ids:
                    pred_ids = pred_ids[:pred_ids.index(eos_idx)]
                pred_tokens = [tgt_vocab.lookup_token(idx) for idx in pred_ids]

                ref_ids = tgt[i].tolist()
                if sos_idx in ref_ids:
                    ref_ids = ref_ids[ref_ids.index(sos_idx) + 1:]
                if eos_idx in ref_ids:
                    ref_ids = ref_ids[:ref_ids.index(eos_idx)]
                ref_tokens = [tgt_vocab.lookup_token(idx) for idx in ref_ids]

                hypotheses.append(pred_tokens)
                references.append([ref_tokens])

    # corpus_bleu returns 0-1; scale to 0-100
    return corpus_bleu(references, hypotheses) * 100.0


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model:     Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    path:      str = "checkpoint.pt",
) -> None:
    """Save model + optimizer + scheduler + vocab + config."""
    src_stoi = model.src_vocab.stoi if hasattr(model, 'src_vocab') else {}
    tgt_stoi = model.tgt_vocab.stoi if hasattr(model, 'tgt_vocab') else {}
    warmup_steps = getattr(scheduler, 'warmup_steps', 4000)

    torch.save(
        {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'warmup_steps':         warmup_steps,
            'src_vocab':            src_stoi,
            'tgt_vocab':            tgt_stoi,
            'model_config': {
                'src_vocab_size': model.src_vocab_size,
                'tgt_vocab_size': model.tgt_vocab_size,
                'd_model':        model.d_model,
                'N':              model.N,
                'num_heads':      model.num_heads,
                'd_ff':           model.d_ff,
                'dropout':        model.dropout_rate,
                'pos_encoding':   getattr(model, 'pos_encoding', 'sinusoidal'),
                'use_scale':      getattr(model, 'use_scale', True),
            },
        },
        path,
    )
    print(f"Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path:      str,
    model:     Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler  = None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt.get('epoch', 0)
    print(f"Checkpoint loaded ← {path}  (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER  (Part 2 ablation flags)
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Train Transformer for DE→EN translation")

    # ── Model ──────────────────────────────────────────────────────────
    p.add_argument("--d_model",    type=int,   default=256)
    p.add_argument("--N",          type=int,   default=8,   help="Number of encoder/decoder layers")
    p.add_argument("--num_heads",  type=int,   default=8)
    p.add_argument("--d_ff",       type=int,   default=512)
    p.add_argument("--dropout",    type=float, default=0.2)

    # ── Training ───────────────────────────────────────────────────────
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--num_epochs",   type=int,   default=35)
    p.add_argument("--warmup_steps", type=int,   default=4000)
    p.add_argument("--max_len",      type=int,   default=100)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--pin_memory",   action="store_true", default=True)

    # ── Part 2 ablation flags ──────────────────────────────────────────
    p.add_argument(
        "--scheduler",
        type=str, choices=["noam", "fixed"], default="noam",
        help="2.1 — noam: warmup+decay  |  fixed: constant lr",
    )
    p.add_argument(
        "--fixed_lr",
        type=float, default=1e-4,
        help="2.1 — learning rate to use when --scheduler fixed",
    )
    p.add_argument(
        "--use_scale",
        type=lambda x: x.lower() != "false", default=True,
        metavar="true|false",
        help="2.2 — include 1/√dk scaling in attention (default: true)",
    )
    p.add_argument(
        "--log_grad_norm",
        action="store_true", default=False,
        help="2.2 — log W_q / W_k gradient norms to W&B every step",
    )
    p.add_argument(
        "--pos_encoding",
        type=str, choices=["sinusoidal", "learned"], default="sinusoidal",
        help="2.4 — positional encoding type",
    )
    p.add_argument(
        "--smoothing",
        type=float, default=0.1,
        help="2.5 — label smoothing ε (0.0 = standard cross-entropy)",
    )

    # ── Misc ───────────────────────────────────────────────────────────
    p.add_argument("--run_name", type=str, default=None,
                   help="W&B run name (auto-generated from ablation flags if omitted)")
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(args) -> None:
    import wandb
    from dataset import Multi30kDataset, pad_idx
    from lr_scheduler import NoamScheduler, FixedLRScheduler

    # ── Auto-generate run name from ablation flags ────────────────────
    # FIX: run_name was undefined in original code
    if args.run_name is None:
        args.run_name = (
            f"sched={args.scheduler}"
            f"_scale={args.use_scale}"
            f"_pe={args.pos_encoding}"
            f"_ls={args.smoothing}"
        )

    config = dict(
        d_model       = args.d_model,
        N             = args.N,
        num_heads     = args.num_heads,
        d_ff          = args.d_ff,
        dropout       = args.dropout,
        batch_size    = args.batch_size,
        num_epochs    = args.num_epochs,
        warmup_steps  = args.warmup_steps,
        smoothing     = args.smoothing,
        max_len       = args.max_len,
        num_workers   = args.num_workers,
        pin_memory    = args.pin_memory,
        # ablation metadata logged to W&B
        scheduler     = args.scheduler,
        fixed_lr      = args.fixed_lr,
        use_scale     = args.use_scale,
        pos_encoding  = args.pos_encoding,
        log_grad_norm = args.log_grad_norm,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Run: {args.run_name}")

    wandb.init(project="da6401-a3", name=args.run_name, config=config)
    wandb.config.update(config, allow_val_change=True)
    cfg = wandb.config

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = Multi30kDataset(split="train")
    train_loader, val_loader, test_loader = train_ds.get_dataloaders(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg.d_model,
        N              = cfg.N,
        num_heads      = cfg.num_heads,
        d_ff           = cfg.d_ff,
        dropout        = cfg.dropout,
        pos_encoding   = cfg.pos_encoding,   # ablation 2.4
        use_scale      = cfg.use_scale,      # ablation 2.2
    ).to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    # ablation 2.1: noam vs fixed LR
    if cfg.scheduler == "noam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(
            optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps
        )
    else:  # fixed
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.fixed_lr, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = FixedLRScheduler(optimizer)

    # ablation 2.5: smoothing=0.0 → standard CE; smoothing=0.1 → label smoothing
    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = pad_idx,
        smoothing  = cfg.smoothing,
    )

    # Attach vocabs so save_checkpoint can bundle them
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    global_steps = 0
    best_val_loss = float('inf')   

    for epoch in range(cfg.num_epochs):
        train_loss, steps = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
            log_grad_norm=cfg.log_grad_norm,  # ablation 2.2
            wandb_run=wandb,
            step_offset=global_steps,
        )
        global_steps += steps

        val_loss, _ = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch,
                            path=os.path.join(args.ckpt_dir, f"{args.run_name}_BEST.pt"))


        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={current_lr:.2e}"
        )
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
        })

        # FIX: ckpt_path was undefined in original — now properly set
        ckpt_path = os.path.join(args.ckpt_dir, f"{args.run_name}_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, path=ckpt_path)

        safe_run_name = (
            args.run_name
            .replace("=", "_")
            .replace("/", "_")
            .replace(" ", "_")
        )

        artifact = wandb.Artifact(
            name=f"{safe_run_name}-epoch-{epoch}",
            type="model"
        )
        artifact.add_file(ckpt_path)                  # FIX: ckpt_path now defined
        wandb.log_artifact(artifact)

    # ── Final BLEU ────────────────────────────────────────────────────
    bleu = evaluate_bleu(
        model, test_loader, tgt_vocab, device=device, max_len=cfg.max_len
    )
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    run_training_experiment(args)