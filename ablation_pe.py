"""
ablation_pe.py  —  Section 2.4: Positional Encoding vs. Learned Embeddings
DA6401 Assignment 3

Trains two identical models:
  1. Sinusoidal PE  (your existing PositionalEncoding)
  2. Learned PE     (nn.Embedding — trainable positional parameters)

Compares validation BLEU after every epoch and logs to W&B.

Run:
    python ablation_pe.py
"""

import math
import torch
import torch.nn as nn
import wandb

from dataset import Multi30kDataset, pad_idx, sos_idx, eos_idx
from model import (
    PositionalEncoding,         # sinusoidal
    PositionwiseFeedForward,
    EncoderLayer, DecoderLayer,
    Encoder, Decoder,
    make_src_mask, make_tgt_mask,
)
from train import LabelSmoothingLoss, run_epoch, evaluate_bleu
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
#  Learned Positional Encoding
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """
    Replaces the sinusoidal table with a trainable nn.Embedding.
    Positions 0 … max_len-1 are looked up and added to the token embeddings.
    """

    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 256) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pos_embed.weight, mean=0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len  = x.size(1)
        device   = x.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
        x = x + self.pos_embed(positions)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  Transformer with swappable PE
# ══════════════════════════════════════════════════════════════════════

class FlexTransformer(nn.Module):
    """
    Same architecture as your Transformer but accepts a `pe_type` flag:
      'sinusoidal' → PositionalEncoding   (fixed, not trained)
      'learned'    → LearnedPositionalEncoding  (trainable)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:  int   = 256,
        N:        int   = 3,
        num_heads: int  = 8,
        d_ff:     int   = 512,
        dropout:  float = 0.1,
        pe_type:  str   = "sinusoidal",   # 'sinusoidal' | 'learned'
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        def _make_pe():
            if pe_type == "learned":
                return LearnedPositionalEncoding(d_model, dropout)
            return PositionalEncoding(d_model, dropout)

        self.src_pe = _make_pe()
        self.tgt_pe = _make_pe()

        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.fc_out(self.decoder(x, memory, src_mask, tgt_mask))

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


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
    smoothing    = 0.1,
    max_len      = 100,
)


def run_variant(pe_type: str, device: str,
                train_loader, val_loader, test_loader,
                src_vocab, tgt_vocab) -> None:

    label = pe_type   # 'sinusoidal' or 'learned'
    print(f"\n{'='*60}\n  PE variant: {label}\n{'='*60}")

    model = FlexTransformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model  = CFG["d_model"],
        N        = CFG["N"],
        num_heads = CFG["num_heads"],
        d_ff     = CFG["d_ff"],
        dropout  = CFG["dropout"],
        pe_type  = pe_type,
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
        smoothing  = CFG["smoothing"],
    )

    run = wandb.init(
        project = "da6401-a3",
        name    = f"ablation_pe_{label}",
        config  = {**CFG, "pe_type": pe_type},
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

        # Compute val BLEU every 2 epochs to save time
        val_bleu = None
        if epoch % 2 == 0 or epoch == CFG["num_epochs"] - 1:
            val_bleu = evaluate_bleu(
                model, val_loader, tgt_vocab,
                device=device, max_len=CFG["max_len"]
            )

        print(f"  Epoch {epoch:02d}  train={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}"
              + (f"  val_bleu={val_bleu:.2f}" if val_bleu is not None else ""))

        log_dict = {
            "epoch"                        : epoch,
            f"train_loss/{label}"          : train_loss,
            f"val_loss/{label}"            : val_loss,
        }
        if val_bleu is not None:
            log_dict[f"val_bleu/{label}"]  = val_bleu
        wandb.log(log_dict)

    # Final test BLEU
    test_bleu = evaluate_bleu(
        model, test_loader, tgt_vocab,
        device=device, max_len=CFG["max_len"]
    )
    print(f"\n  [{label}] Test BLEU: {test_bleu:.2f}")
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

    run_variant("sinusoidal", device, train_loader, val_loader, test_loader,
                src_vocab, tgt_vocab)
    run_variant("learned",    device, train_loader, val_loader, test_loader,
                src_vocab, tgt_vocab)

    print("\nDone. In W&B overlay:")
    print("  val_bleu/sinusoidal  vs  val_bleu/learned")
    print("  test_bleu/sinusoidal vs  test_bleu/learned")


if __name__ == "__main__":
    main()