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
"""

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
        smoothing  (float): Smoothing factor ε (default 0.1).
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
            logits : shape [batch * tgt_len, vocab_size]
            target : shape [batch * tgt_len]

        Returns:
            Scalar loss.
        """
        # Build smoothed distribution
        smooth_val = self.smoothing / (self.vocab_size - 2)   # -2: exclude correct + pad
        dist = torch.full_like(logits, smooth_val)
        dist[:, self.pad_idx] = 0.0
        dist.scatter_(1, target.unsqueeze(1), self.confidence)

        # Zero out rows where the target is <pad> (don't compute loss there)
        mask = (target == self.pad_idx)
        dist[mask] = 0.0

        # KL-divergence loss: sum over vocab, mean over non-pad tokens
        log_probs  = torch.log_softmax(logits, dim=-1)
        loss       = -(dist * log_probs).sum(dim=-1)
        non_pad    = (~mask).sum().clamp(min=1)
        return loss.sum() / non_pad


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Returns:
        avg_loss : Average loss over the epoch.
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    pad_idx      = 1   # from dataset.py constant

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} epoch {epoch_num}"):
            src, tgt = batch
            src = src.to(device)   # [B, S]
            tgt = tgt.to(device)   # [B, T]

            # teacher-forcing: decoder input = tgt[:-1], target = tgt[1:]
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx)

            logits = model(src, tgt_in, src_mask, tgt_mask)
            # logits: [B, T-1, vocab]

            B, T, V = logits.shape
            loss = loss_fn(
                logits.contiguous().view(B * T, V),
                tgt_out.contiguous().view(B * T),
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            non_pad     = (tgt_out != pad_idx).sum().item()
            total_loss  += loss.item() * non_pad
            total_tokens += non_pad

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
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
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    from dataset import sos_idx, eos_idx, pad_idx

    model.eval()
    hypotheses  = []
    references  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval"):
            src = src.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)                      # [1, S]
                src_mask = make_src_mask(src_i, pad_idx).to(device) # [1,1,1,S]

                out = greedy_decode(
                    model, src_i, src_mask,
                    max_len=max_len,
                    start_symbol=sos_idx,
                    end_symbol=eos_idx,
                    device=device,
                )

                # Convert predicted token ids → tokens (strip BOS/EOS)
                pred_ids = out[0].tolist()
                if sos_idx in pred_ids:
                    pred_ids = pred_ids[pred_ids.index(sos_idx) + 1:]
                if eos_idx in pred_ids:
                    pred_ids = pred_ids[:pred_ids.index(eos_idx)]
                pred_tokens = [tgt_vocab.lookup_token(idx) for idx in pred_ids]

                # Reference: strip BOS/EOS from ground-truth
                ref_ids = tgt[i].tolist()
                if sos_idx in ref_ids:
                    ref_ids = ref_ids[ref_ids.index(sos_idx) + 1:]
                if eos_idx in ref_ids:
                    ref_ids = ref_ids[:ref_ids.index(eos_idx)]
                ref_tokens = [tgt_vocab.lookup_token(idx) for idx in ref_ids]

                hypotheses.append(pred_tokens)
                references.append([ref_tokens])   # torchtext expects list-of-lists

    score = corpus_bleu(references, hypotheses) * 100.0
    return score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'model_config': {
                'src_vocab_size': model.src_vocab_size,
                'tgt_vocab_size': model.tgt_vocab_size,
                'd_model':        model.d_model,
                'N':              model.N,
                'num_heads':      model.num_heads,
                'd_ff':           model.d_ff,
                'dropout':        model.dropout_rate,
            },
        },
        path,
    )
    print(f"Checkpoint saved to {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    epoch = ckpt.get('epoch', 0)
    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    import wandb
    from dataset import Multi30kDataset, pad_idx
    from lr_scheduler import NoamScheduler

    # ── Hyperparameters ───────────────────────────────────────────────
    config = dict(
        d_model      = 256,
        N            = 3,
        num_heads    = 8,
        d_ff         = 512,
        dropout      = 0.1,
        batch_size   = 128,
        num_epochs   = 10,
        warmup_steps = 4000,
        smoothing    = 0.1,
        max_len      = 100,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config

    # ── Data ──────────────────────────────────────────────────────────
    train_ds                        = Multi30kDataset(split="train")
    train_loader, val_loader, test_loader = train_ds.get_dataloaders(batch_size=cfg.batch_size)
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
    ).to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    loss_fn   = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = pad_idx,
        smoothing  = cfg.smoothing,
    )

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )
        print(f"Epoch {epoch:02d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        save_checkpoint(model, optimizer, scheduler, epoch, path=f"checkpoint_epoch{epoch}.pt")

    # ── Final BLEU ────────────────────────────────────────────────────
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=cfg.max_len)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({"test_bleu": bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()