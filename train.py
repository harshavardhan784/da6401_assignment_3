# train.py
"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol,         │
  │                end_symbol, device)  → torch.Tensor [1, out_len]     │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘

Design notes:
  - evaluate_bleu returns 0-100 as per assignment spec (20M test criterion).
  - run_epoch prints per-epoch: train loss, val loss, val BLEU.
  - LabelSmoothingLoss supports eps=0.0 (standard cross-entropy) for ablation.
  - All Part-2 experiment flags (noam/fixed LR, scaling on/off, label
    smoothing, learned pos enc) are accepted as arguments so parts can be
    reused in part2_experiments.py with minimal changes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import os

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing cross-entropy loss as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth[correct] = 1 - eps
        y_smooth[other]   = eps / (vocab_size - 2)   # -2: exclude correct & pad
        y_smooth[pad]     = 0

    When eps=0.0 this reduces to standard cross-entropy (used in ablation 2.5).

    Args:
        vocab_size : Number of output classes.
        pad_idx    : Index of <pad> token — always gets zero probability.
        smoothing  : Smoothing factor ε (default 0.1; set 0.0 for no smoothing).
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
            logits : (batch * tgt_len, vocab_size)
            target : (batch * tgt_len,)
        Returns:
            Scalar mean loss over non-pad positions.
        """
        if self.smoothing == 0.0:
            # Standard cross-entropy, ignoring pad positions
            return nn.functional.cross_entropy(
                logits, target, ignore_index=self.pad_idx
            )

        smooth_val = self.smoothing / max(self.vocab_size - 2, 1)
        dist = torch.full_like(logits, smooth_val)
        dist[:, self.pad_idx] = 0.0
        dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Zero out pad-token rows
        mask = (target == self.pad_idx)
        dist[mask] = 0.0

        log_probs = torch.log_softmax(logits, dim=-1)
        loss      = -(dist * log_probs).sum(dim=-1)
        non_pad   = (~mask).sum().clamp(min=1)
        return loss.sum() / non_pad


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
    use_scaling: bool = True,   # Part-2 ablation 2.2: toggle √(d_k) scaling
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        use_scaling : If False, patches scaled_dot_product_attention to skip
                      the 1/√d_k factor (ablation 2.2 only). Restored after epoch.
    Returns:
        avg_loss : Average token loss over the epoch.
    """
    import model as model_module

    model.train() if is_train else model.eval()

    # ── Optional: disable 1/√d_k scaling for ablation 2.2 ────────────
    original_sdpa = None
    if not use_scaling:
        import math, torch as _torch, torch.nn.functional as _F
        from typing import Optional as _Opt, Tuple as _Tup

        original_sdpa = model_module.scaled_dot_product_attention

        def _no_scale_sdpa(Q, K, V, mask=None):
            scores = _torch.matmul(Q, K.transpose(-2, -1))   # no / sqrt(d_k)
            if mask is not None:
                scores = scores.masked_fill(mask, float('-inf'))
            attn_w = _F.softmax(scores, dim=-1)
            attn_w = _torch.nan_to_num(attn_w, nan=0.0)
            return _torch.matmul(attn_w, V), attn_w

        model_module.scaled_dot_product_attention = _no_scale_sdpa

    total_loss   = 0.0
    total_tokens = 0
    pad_idx      = 1

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    try:
        with ctx:
            for batch in tqdm(
                data_iter,
                desc=f"{'Train' if is_train else 'Val  '} epoch {epoch_num:02d}",
                leave=False,
            ):
                src, tgt = batch
                src = src.to(device)
                tgt = tgt.to(device)

                tgt_in  = tgt[:, :-1]   # decoder input  (drop last token)
                tgt_out = tgt[:, 1:]    # expected output (drop first token)

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
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                non_pad      = (tgt_out != pad_idx).sum().item()
                total_loss  += loss.item() * non_pad
                total_tokens += non_pad
    finally:
        # Always restore original SDPA even if an exception occurs
        if original_sdpa is not None:
            model_module.scaled_dot_product_attention = original_sdpa

    return total_loss / max(total_tokens, 1)


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

    Args:
        model        : Trained Transformer.
        src          : Source token ids, shape (1, src_len).
        src_mask     : Source padding mask, shape (1, 1, 1, src_len).
        max_len      : Maximum number of output tokens.
        start_symbol : BOS token index.
        end_symbol   : EOS token index.
        device       : Compute device.

    Returns:
        ys : Generated token indices, shape (1, out_len).
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
        bleu_score : Corpus-level BLEU in range 0–100.

    Note: nltk's corpus_bleu() returns 0–1; we multiply by 100 to match
    the assignment specification ("corpus-level BLEU score" section 1.3 /
    test-set performance criterion).
    """
    from dataset import sos_idx, eos_idx, pad_idx

    model.eval()
    hypotheses = []
    references = []
    smoother   = SmoothingFunction().method1

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
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

                # Hypothesis: strip BOS / EOS
                pred_ids = out[0].tolist()
                if sos_idx in pred_ids:
                    pred_ids = pred_ids[pred_ids.index(sos_idx) + 1:]
                if eos_idx in pred_ids:
                    pred_ids = pred_ids[:pred_ids.index(eos_idx)]
                pred_tokens = [tgt_vocab.lookup_token(idx) for idx in pred_ids]

                # Reference: strip BOS / EOS from ground truth
                ref_ids = tgt[i].tolist()
                if sos_idx in ref_ids:
                    ref_ids = ref_ids[ref_ids.index(sos_idx) + 1:]
                if eos_idx in ref_ids:
                    ref_ids = ref_ids[:ref_ids.index(eos_idx)]
                ref_tokens = [tgt_vocab.lookup_token(idx) for idx in ref_ids]

                hypotheses.append(pred_tokens)
                references.append([ref_tokens])   # nltk expects list-of-lists

    # corpus_bleu returns 0–1; scale to 0–100
    score = corpus_bleu(references, hypotheses, smoothing_function=smoother) * 100.0
    return score


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
    """
    Save a full checkpoint including:
      - model weights
      - optimizer state
      - scheduler state
      - vocab dictionaries (for infer())
      - model config (for architecture reconstruction)
    """
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
                'src_vocab_size':  model.src_vocab_size,
                'tgt_vocab_size':  model.tgt_vocab_size,
                'd_model':         model.d_model,
                'N':               model.N,
                'num_heads':       model.num_heads,
                'd_ff':            model.d_ff,
                'dropout':         model.dropout_rate,
                'learned_pos_enc': model.learned_pos_enc,
            },
        },
        path,
    )
    print(f"  ✔ Checkpoint saved → {path}  (epoch {epoch})")


def load_checkpoint(
    path:      str,
    model:     Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler  = None,
) -> int:
    """
    Load a checkpoint into model (and optionally optimizer/scheduler).
    Returns the saved epoch number.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt.get('epoch', 0)
    print(f"  ✔ Checkpoint loaded ← {path}  (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  MAIN TRAINING ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment(
    # ── Architecture ──────────────────────────────────────────────────
    d_model:         int   = 256,
    N:               int   = 8,
    num_heads:       int   = 8,
    d_ff:            int   = 512,
    dropout:         float = 0.2,
    learned_pos_enc: bool  = True,   # Part-2 ablation 2.4
    # ── Training ──────────────────────────────────────────────────────
    batch_size:      int   = 128,
    num_epochs:      int   = 30,
    warmup_steps:    int   = 4000,
    smoothing:       float = 0.2,     # Part-2 ablation 2.5: set 0.0 for no smoothing
    fixed_lr:        float = None,    # Part-2 ablation 2.1: set e.g. 1e-4 for fixed LR
    use_scaling:     bool  = True,    # Part-2 ablation 2.2: set False to remove 1/√d_k
    max_len:         int   = 100,
    # ── Logistics ─────────────────────────────────────────────────────
    ckpt_dir:        str   = "checkpoints",
    run_name:        str   = "baseline",
    wandb_project:   str   = "da6401-a3",
    resume_from:     str   = None,    # path to checkpoint to resume from
    eval_bleu_every: int   = 1,       # compute val BLEU every N epochs
) -> None:
    """
    Full training pipeline. All Part-2 ablation flags are exposed as
    arguments so part2_experiments.py can call this with different configs
    without duplicating code.
    """
    import wandb
    from dataset import Multi30kDataset, pad_idx
    from lr_scheduler import NoamScheduler, FixedLRScheduler

    config = dict(
        d_model=d_model, N=N, num_heads=num_heads, d_ff=d_ff, dropout=dropout,
        learned_pos_enc=learned_pos_enc,
        batch_size=batch_size, num_epochs=num_epochs, warmup_steps=warmup_steps,
        smoothing=smoothing, fixed_lr=fixed_lr, use_scaling=use_scaling,
        max_len=max_len, run_name=run_name,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Run: {run_name}   Device: {device}")
    print(f"{'='*60}")

    wandb.init(project=wandb_project, name=run_name, config=config, reinit=True)
    cfg = wandb.config

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = Multi30kDataset(split="train")
    train_loader, val_loader, test_loader = train_ds.get_dataloaders(
        batch_size=cfg.batch_size
    )
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size  = len(src_vocab),
        tgt_vocab_size  = len(tgt_vocab),
        d_model         = cfg.d_model,
        N               = cfg.N,
        num_heads       = cfg.num_heads,
        d_ff            = cfg.d_ff,
        dropout         = cfg.dropout,
        learned_pos_enc = cfg.learned_pos_enc,
        checkpoint_path = "NOT_AUTOGRADER",   # training mode — skip download
    ).to(device)

    # Attach vocabs so save_checkpoint can bundle them
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    if cfg.fixed_lr is not None:
        # Ablation 2.1: constant learning rate, no warm-up
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.fixed_lr, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = FixedLRScheduler(optimizer)
        print(f"  Scheduler: Fixed LR = {cfg.fixed_lr}")
    else:
        # Default: Noam schedule
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(
            optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps
        )
        print(f"  Scheduler: Noam (d_model={cfg.d_model}, warmup={cfg.warmup_steps})")

    # ── Loss function ─────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = pad_idx,
        smoothing  = cfg.smoothing,
    )
    print(f"  Label smoothing: ε={cfg.smoothing}  |  Scaling: {cfg.use_scaling}")
    print(f"  Layers: N={cfg.N}  heads={cfg.num_heads}  d_ff={cfg.d_ff}")

    # ── Optional resume ───────────────────────────────────────────────
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        start_epoch = load_checkpoint(resume_from, model, optimizer, scheduler) + 1

    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────
    best_val_bleu = -1.0
    best_ckpt_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")

    for epoch in range(start_epoch, cfg.num_epochs):
        # ── Train ──────────────────────────────────────────────────
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
            use_scaling=cfg.use_scaling,
        )

        # ── Validate ───────────────────────────────────────────────
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
            use_scaling=cfg.use_scaling,
        )

        # ── Val BLEU ───────────────────────────────────────────────
        val_bleu = 0.0
        if (epoch + 1) % eval_bleu_every == 0:
            val_bleu = evaluate_bleu(
                model, val_loader, tgt_vocab, device=device, max_len=cfg.max_len
            )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_BLEU={val_bleu:.2f} | "
            f"lr={current_lr:.2e}"
        )

        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_bleu":   val_bleu,
            "lr":         current_lr,
        })

        # ── Save per-epoch checkpoint ──────────────────────────────
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_epoch{epoch}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch, path=ckpt_path)

        # ── Save best checkpoint ───────────────────────────────────
        if val_bleu > best_val_bleu:
            best_val_bleu = val_bleu
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt_path)
            print(f"  ★ New best val BLEU: {val_bleu:.2f}  → saved {best_ckpt_path}")

    # ── Final test BLEU ───────────────────────────────────────────────
    print("\nLoading best checkpoint for final test evaluation …")
    if os.path.exists(best_ckpt_path):
        load_checkpoint(best_ckpt_path, model)

    test_bleu = evaluate_bleu(
        model, test_loader, tgt_vocab, device=device, max_len=cfg.max_len
    )
    print(f"\nTest BLEU: {test_bleu:.2f}")
    wandb.log({"test_bleu": test_bleu})
    wandb.finish()
    return test_bleu


if __name__ == "__main__":
    run_training_experiment()