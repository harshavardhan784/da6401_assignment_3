"""
ablation_smoothing.py  —  Section 2.5: Decoder Sensitivity — Label Smoothing
DA6401 Assignment 3

Trains two identical models:
  1. Label smoothing  ε = 0.1  (as in the paper)
  2. Label smoothing  ε = 0.0  (standard cross-entropy)

Additionally logs "Prediction Confidence" — the softmax probability
assigned to the correct token — to W&B so you can see the over-confidence
effect directly.

Run:
    python ablation_smoothing.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from dataset import Multi30kDataset, pad_idx
from model import Transformer, make_src_mask, make_tgt_mask
from train import LabelSmoothingLoss, run_epoch, evaluate_bleu
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  Prediction-Confidence helper
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_prediction_confidence(
    model: Transformer,
    data_loader,
    device: str,
    num_batches: int = 5,
) -> float:
    """
    Average softmax probability of the CORRECT token across `num_batches`
    validation batches (ignoring <pad> positions).

    Returns:
        mean_confidence : float in (0, 1).
    """
    model.eval()
    total_conf  = 0.0
    total_count = 0

    for i, (src, tgt) in enumerate(data_loader):
        if i >= num_batches:
            break

        src, tgt = src.to(device), tgt.to(device)
        tgt_in   = tgt[:, :-1]
        tgt_out  = tgt[:, 1:]

        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_in, pad_idx)

        logits = model(src, tgt_in, src_mask, tgt_mask)    # (B, T, V)
        probs  = F.softmax(logits, dim=-1)                  # (B, T, V)

        B, T, V = probs.shape
        # Gather the probability of the correct token at each position
        correct_probs = probs.gather(
            dim=-1,
            index=tgt_out.unsqueeze(-1).clamp(0, V - 1)
        ).squeeze(-1)   # (B, T)

        non_pad = (tgt_out != pad_idx)
        total_conf  += correct_probs[non_pad].sum().item()
        total_count += non_pad.sum().item()

    model.train()
    return total_conf / max(total_count, 1)


# ══════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════

CFG = dict(
    d_model      = 256,
    N            = 3,
    num_heads    = 8,
    d_ff         = 512,
    dropout      = 0.1,
    batch_size   = 128,
    num_epochs   = 10,
    warmup_steps = 4000,
    max_len      = 100,
)


def run_variant(smoothing: float, device: str,
                train_loader, val_loader, test_loader,
                src_vocab, tgt_vocab) -> None:

    label = f"smoothing_{smoothing:.1f}".replace(".", "_")
    pretty = f"ε={smoothing}"
    print(f"\n{'='*60}\n  Label smoothing: {pretty}\n{'='*60}")

    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model   = CFG["d_model"],
        N         = CFG["N"],
        num_heads = CFG["num_heads"],
        d_ff      = CFG["d_ff"],
        dropout   = CFG["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer, d_model=CFG["d_model"], warmup_steps=CFG["warmup_steps"]
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx    = pad_idx,
        smoothing  = smoothing,
    )

    run = wandb.init(
        project = "da6401-a3",
        name    = f"ablation_smoothing_{label}",
        config  = {**CFG, "smoothing": smoothing},
        reinit  = True,
    )

    for epoch in range(CFG["num_epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        # Prediction confidence on a small slice of val data
        confidence = compute_prediction_confidence(
            model, val_loader, device, num_batches=5
        )

        print(f"  Epoch {epoch:02d}  train={train_loss:.4f}  "
              f"val={val_loss:.4f}  confidence={confidence:.4f}")

        wandb.log({
            "epoch"                            : epoch,
            f"train_loss/{label}"              : train_loss,
            f"val_loss/{label}"                : val_loss,
            f"prediction_confidence/{label}"   : confidence,
            # Also log perplexity (exp of cross-entropy loss)
            f"val_perplexity/{label}"          : math.exp(min(val_loss, 20)),
        })

    # Final test BLEU
    test_bleu = evaluate_bleu(
        model, test_loader, tgt_vocab,
        device=device, max_len=CFG["max_len"]
    )
    print(f"\n  [{pretty}] Test BLEU: {test_bleu:.2f}")
    wandb.log({f"test_bleu/{label}": test_bleu})

    run.finish()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading dataset …")
    train_ds = Multi30kDataset(split="train")
    train_loader, val_loader, test_loader = train_ds.get_dataloaders(
        batch_size=CFG["batch_size"]
    )
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    # ε = 0.1  (paper setting)
    run_variant(0.1, device, train_loader, val_loader, test_loader,
                src_vocab, tgt_vocab)

    # ε = 0.0  (standard cross-entropy — no smoothing)
    run_variant(0.0, device, train_loader, val_loader, test_loader,
                src_vocab, tgt_vocab)

    print("\nDone. In W&B overlay:")
    print("  prediction_confidence/smoothing_0_1  vs  prediction_confidence/smoothing_0_0")
    print("  val_loss/smoothing_0_1               vs  val_loss/smoothing_0_0")
    print("  val_perplexity/smoothing_0_1         vs  val_perplexity/smoothing_0_0")
    print("  test_bleu/smoothing_0_1              vs  test_bleu/smoothing_0_0")


if __name__ == "__main__":
    main()