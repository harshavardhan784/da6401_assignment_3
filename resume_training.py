"""
resume_training.py — Continue training from a saved checkpoint
DA6401 Assignment 3

Loads vocab directly from the checkpoint (no spaCy rebuild needed),
then streams the raw HuggingFace dataset through a lightweight collator.
Much faster startup than re-running Multi30kDataset from scratch.

Usage:
    python resume_training.py --checkpoint checkpoints/checkpoint_epoch19.pt
    python resume_training.py --checkpoint checkpoints/checkpoint_epoch19.pt --epochs 15
    python resume_training.py --checkpoint checkpoints/checkpoint_epoch19.pt --epochs 10 --lr_reset

Options:
    --checkpoint   Path to the .pt checkpoint file  (required)
    --epochs       How many MORE epochs to train     (default: 10)
    --lr_reset     Re-warm the Noam scheduler from step 0 instead of
                   resuming the original LR schedule
"""

import argparse
import os
import torch
import torch.nn as nn
import wandb
from functools import partial
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy

from dataset import Vocab, pad_idx, unk_idx, sos_idx, eos_idx
from model   import Transformer
from train   import LabelSmoothingLoss, run_epoch, evaluate_bleu, save_checkpoint
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Resume transformer training from checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to checkpoint .pt file")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of ADDITIONAL epochs to train (default: 10)")
    p.add_argument("--lr_reset", action="store_true",
                   help="Reset the LR scheduler instead of resuming it")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_len",    type=int, default=100)
    p.add_argument("--wandb_project", default="da6401-a3")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════
#  LIGHTWEIGHT DATASET — uses pre-built vocab from checkpoint
#  No vocab rebuild, no full spaCy pipeline, just tokenize + lookup
# ══════════════════════════════════════════════════════════════════════

class FastMulti30k(Dataset):
    """
    Tokenizes on-the-fly using pre-built stoi dicts from the checkpoint.
    Avoids the slow _build_vocab() loop that crashes on some environments.
    """
    def __init__(self, split: str, src_stoi: dict, tgt_stoi: dict,
                 spacy_de, spacy_en):
        raw        = load_dataset("bentrevett/multi30k")
        self.data  = raw[split]
        self.src_stoi  = src_stoi
        self.tgt_stoi  = tgt_stoi
        self.spacy_de  = spacy_de
        self.spacy_en  = spacy_en

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        src_tokens = [t.text.lower() for t in self.spacy_de.tokenizer(ex["de"])]
        tgt_tokens = [t.text.lower() for t in self.spacy_en.tokenizer(ex["en"])]

        src_ids = ([sos_idx]
                   + [self.src_stoi.get(t, unk_idx) for t in src_tokens]
                   + [eos_idx])
        tgt_ids = ([sos_idx]
                   + [self.tgt_stoi.get(t, unk_idx) for t in tgt_tokens]
                   + [eos_idx])

        return (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long))


def _collate(batch, pad_idx):
    src_batch, tgt_batch = zip(*batch)
    return (pad_sequence(src_batch, batch_first=True, padding_value=pad_idx),
            pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx))


def build_dataloaders(src_stoi, tgt_stoi, batch_size):
    print("  Loading spaCy models …")
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    collate = partial(_collate, pad_idx=pad_idx)

    print("  Building train/val/test datasets from HuggingFace …")
    train_ds = FastMulti30k("train",      src_stoi, tgt_stoi, spacy_de, spacy_en)
    val_ds   = FastMulti30k("validation", src_stoi, tgt_stoi, spacy_de, spacy_en)
    test_ds  = FastMulti30k("test",       src_stoi, tgt_stoi, spacy_de, spacy_en)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate, num_workers=4, pin_memory=True)

    # Wrap stoi dicts as Vocab objects for evaluate_bleu compatibility
    src_vocab = Vocab(src_stoi)
    tgt_vocab = Vocab(tgt_stoi)

    print(f"  src_vocab={len(src_vocab)}  tgt_vocab={len(tgt_vocab)}")
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


# ══════════════════════════════════════════════════════════════════════
#  LOAD CHECKPOINT
# ══════════════════════════════════════════════════════════════════════

def load_for_resume(ckpt_path: str, device: str, lr_reset: bool):
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    cfg = state["model_config"]
    model = Transformer(
        src_vocab_size = cfg["src_vocab_size"],
        tgt_vocab_size = cfg["tgt_vocab_size"],
        d_model        = cfg["d_model"],
        N              = cfg["N"],
        num_heads      = cfg["num_heads"],
        d_ff           = cfg["d_ff"],
        dropout        = cfg["dropout"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    print(f"  Model restored  (src_vocab={cfg['src_vocab_size']}, "
          f"tgt_vocab={cfg['tgt_vocab_size']}, d_model={cfg['d_model']})")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    if "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        print("  Optimizer state restored")

    warmup_steps = state.get("warmup_steps", 4000)
    scheduler    = NoamScheduler(optimizer, d_model=cfg["d_model"],
                                 warmup_steps=warmup_steps)
    if lr_reset:
        print("  LR scheduler reset to step 0  (--lr_reset flag)")
    elif "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
        print("  Scheduler state restored")

    start_epoch = state.get("epoch", 0) + 1
    print(f"  Resuming from epoch {start_epoch}")

    # Vocab stoi dicts saved by save_checkpoint
    src_stoi = state["src_vocab"]
    tgt_stoi = state["tgt_vocab"]
    print(f"  Vocab loaded from checkpoint  "
          f"(src={len(src_stoi)}, tgt={len(tgt_stoi)})")

    return model, optimizer, scheduler, start_epoch, cfg, src_stoi, tgt_stoi


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model + vocab from checkpoint (no rebuild) ───────────────
    model, optimizer, scheduler, start_epoch, cfg, src_stoi, tgt_stoi = \
        load_for_resume(args.checkpoint, device, args.lr_reset)

    # ── Build dataloaders using checkpoint vocab ───────────────────────
    print("Loading dataset …")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = \
        build_dataloaders(src_stoi, tgt_stoi, args.batch_size)

    # Attach vocabs so save_checkpoint bundles them
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = pad_idx,
        smoothing  = 0.1,
    )

    end_epoch   = start_epoch + args.epochs
    ckpt_dir    = os.path.dirname(args.checkpoint) or "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── W&B ───────────────────────────────────────────────────────────
    wandb.init(
        project = args.wandb_project,
        name    = f"resume_from_epoch{start_epoch - 1}",
        config  = {
            **cfg,
            "resumed_from"  : args.checkpoint,
            "start_epoch"   : start_epoch,
            "extra_epochs"  : args.epochs,
            "lr_reset"      : args.lr_reset,
            "batch_size"    : args.batch_size,
        },
    )

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, end_epoch):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        # BLEU every 2 epochs and on the final epoch
        val_bleu = None
        if epoch % 2 == 0 or epoch == end_epoch - 1:
            val_bleu = evaluate_bleu(
                model, val_loader, tgt_vocab,
                device=device, max_len=args.max_len,
            )

        print(
            f"Epoch {epoch:02d}  train={train_loss:.4f}  val_loss={val_loss:.4f}"
            + (f"  val_bleu={val_bleu:.4f}" if val_bleu is not None else "")
        )

        log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        if val_bleu is not None:
            log["val_bleu"] = val_bleu
        wandb.log(log)

        # Save every epoch
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            path=os.path.join(ckpt_dir, f"checkpoint_epoch{epoch}.pt"),
        )

    # ── Final test BLEU ───────────────────────────────────────────────
    test_bleu = evaluate_bleu(
        model, test_loader, tgt_vocab,
        device=device, max_len=args.max_len,
    )
    print(f"\nTest BLEU: {test_bleu:.4f}")
    wandb.log({"test_bleu": test_bleu})
    wandb.finish()


if __name__ == "__main__":
    main()