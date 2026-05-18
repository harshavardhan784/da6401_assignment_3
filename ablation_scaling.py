"""
ablation_scaling.py  —  Section 2.2: The Scaling Factor 1/√dk
DA6401 Assignment 3

Trains two model variants and logs Q/K gradient norms for the first
1,000 steps to show the effect of removing the scaling factor.

Run:
    python ablation_scaling.py
"""

import math
import copy
import os
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# ──────────────────────────────────────────────────────────────────────
#  Patched attention — adds a `scale` toggle
# ──────────────────────────────────────────────────────────────────────

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scale: bool = True,          # <-- ablation knob
) -> Tuple[torch.Tensor, torch.Tensor]:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    if use_scale:
        scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ──────────────────────────────────────────────────────────────────────
#  MHA — forwards use_scale down to the attention fn
# ──────────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 dropout: float = 0.1, use_scale: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.use_scale = use_scale

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        B, _, S, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask,
                                                   use_scale=self.use_scale)
        return self.W_o(self._merge_heads(attn_out))


# ──────────────────────────────────────────────────────────────────────
#  Minimal Transformer re-using your existing building blocks
#  (import the rest from model.py to keep things DRY)
# ──────────────────────────────────────────────────────────────────────

from model import (
    PositionalEncoding,
    PositionwiseFeedForward,
    Encoder, Decoder,
    make_src_mask, make_tgt_mask,
)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, use_scale):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, use_scale):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class AblationTransformer(nn.Module):
    """Transformer with a togglable scaling factor in every attention layer."""

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, N=3, num_heads=8, d_ff=512,
                 dropout=0.1, use_scale=True):
        super().__init__()
        self.d_model = d_model
        self.use_scale = use_scale

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.src_pe    = PositionalEncoding(d_model, dropout)
        self.tgt_pe    = PositionalEncoding(d_model, dropout)

        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout, use_scale)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout, use_scale)
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


# ──────────────────────────────────────────────────────────────────────
#  Gradient norm tracker
# ──────────────────────────────────────────────────────────────────────

def collect_qk_grad_norms(model: AblationTransformer) -> dict:
    """
    After loss.backward(), collect the L2 gradient norms for every
    W_q and W_k weight matrix in the encoder's self-attention layers.
    Returns a flat dict suitable for wandb.log().
    """
    norms = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            for param_name in ("W_q", "W_k"):
                param = getattr(module, param_name).weight
                if param.grad is not None:
                    norm_val = param.grad.norm(2).item()
                    key = f"grad_norm/{name}.{param_name}"
                    norms[key] = norm_val
    return norms


# ──────────────────────────────────────────────────────────────────────
#  Training loop (first LOG_STEPS steps only for the ablation)
# ──────────────────────────────────────────────────────────────────────

from train import LabelSmoothingLoss
from dataset import Multi30kDataset, pad_idx
from lr_scheduler import NoamScheduler

LOG_STEPS = 1000   # number of steps to capture grad norms


def run_ablation(use_scale: bool, device: str) -> None:
    label = "with_scale" if use_scale else "without_scale"
    print(f"\n{'='*60}")
    print(f"  Running ablation: {label}")
    print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────
    train_ds = Multi30kDataset(split="train")
    train_loader, val_loader, _ = train_ds.get_dataloaders(batch_size=128)
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    # ── Model ─────────────────────────────────────────────────────────
    model = AblationTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256, N=3, num_heads=8, d_ff=512,
        dropout=0.1, use_scale=use_scale,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
    loss_fn   = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab), pad_idx=pad_idx, smoothing=0.1
    )

    # ── W&B run ────────────────────────────────────────────────────────
    run = wandb.init(
        project="da6401-a3",
        name=f"ablation_scaling_{label}",
        config={
            "use_scale": use_scale,
            "d_model": 256, "N": 3, "num_heads": 8, "d_ff": 512,
            "log_steps": LOG_STEPS,
        },
        reinit=True,
    )

    model.train()
    global_step = 0

    # Iterate over batches; stop after LOG_STEPS for the gradient-norm
    # portion, then continue for one full epoch of loss logging.
    for epoch in range(10):
        epoch_loss = 0.0
        epoch_tokens = 0

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
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

            optimizer.zero_grad()
            loss.backward()

            # ── Gradient norm logging (first LOG_STEPS steps only) ────
            if global_step < LOG_STEPS:
                grad_norms = collect_qk_grad_norms(model)
                # Also log a single aggregate for easy charting
                if grad_norms:
                    avg_norm = sum(grad_norms.values()) / len(grad_norms)
                    grad_norms[f"grad_norm/avg_QK_{label}"] = avg_norm
                wandb.log({"step": global_step, "train_loss_step": loss.item(),
                           **grad_norms})

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            non_pad = (tgt_out != pad_idx).sum().item()
            epoch_loss   += loss.item() * non_pad
            epoch_tokens += non_pad
            global_step  += 1

        avg_loss = epoch_loss / max(epoch_tokens, 1)

        # Validation loss
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                src_mask = make_src_mask(src, pad_idx)
                tgt_mask = make_tgt_mask(tgt_in, pad_idx)
                logits = model(src, tgt_in, src_mask, tgt_mask)
                B, T, V = logits.shape
                l = loss_fn(logits.contiguous().view(B*T, V),
                            tgt_out.contiguous().view(B*T))
                np_ = (tgt_out != pad_idx).sum().item()
                val_loss   += l.item() * np_
                val_tokens += np_
        model.train()

        val_avg = val_loss / max(val_tokens, 1)
        print(f"  Epoch {epoch:02d}  train={avg_loss:.4f}  val={val_avg:.4f}")
        wandb.log({"epoch": epoch,
                   f"train_loss_{label}": avg_loss,
                   f"val_loss_{label}": val_avg})

    run.finish()


# ──────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Run both variants sequentially
    run_ablation(use_scale=True,  device=device)
    run_ablation(use_scale=False, device=device)

    print("\nBoth ablation runs complete. Check your W&B dashboard for:")
    print("  • grad_norm/avg_QK_with_scale    vs")
    print("  • grad_norm/avg_QK_without_scale")
    print("  • train_loss_with_scale          vs")
    print("  • train_loss_without_scale")