"""
ablation_attention.py  —  Section 2.3: Attention Rollout & Head Specialization
DA6401 Assignment 3

Loads your best checkpoint, picks one German sentence, runs the encoder,
and extracts the attention weights from the LAST encoder layer.
Logs one heatmap per head to W&B and prints a brief head-specialization
analysis.

Run:
    python ablation_attention.py --ckpt checkpoints/checkpoint_epoch9.pt
"""

import argparse
import math
import copy
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – saves PNGs for W&B
import matplotlib.pyplot as plt
import wandb

from model import (
    Transformer,
    PositionalEncoding,
    PositionwiseFeedForward,
    Encoder, Decoder,
    make_src_mask, make_tgt_mask,
)
from dataset import Multi30kDataset, pad_idx, sos_idx, eos_idx


# ══════════════════════════════════════════════════════════════════════
#  Instrumented MHA — stores last attention weights as an attribute
# ══════════════════════════════════════════════════════════════════════

def _sdpa_returning_weights(Q, K, V, mask=None):
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_w = F.softmax(scores, dim=-1)
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    return torch.matmul(attn_w, V), attn_w


class InstrumentedMHA(nn.Module):
    """MHA that caches attention weights after every forward pass."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

        self.last_attn_weights: Optional[torch.Tensor] = None  # (B, H, S, S)

    def _split(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):
        B, _, S, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, S, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self._split(self.W_q(query))
        K = self._split(self.W_k(key))
        V = self._split(self.W_v(value))
        out, weights = _sdpa_returning_weights(Q, K, V, mask)
        self.last_attn_weights = weights.detach().cpu()   # cache
        return self.W_o(self._merge(out))


# ══════════════════════════════════════════════════════════════════════
#  Instrumented EncoderLayer — uses InstrumentedMHA
# ══════════════════════════════════════════════════════════════════════

class InstrumentedEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = InstrumentedMHA(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


# ══════════════════════════════════════════════════════════════════════
#  Build an instrumented Transformer from a checkpoint
# ══════════════════════════════════════════════════════════════════════

def load_instrumented_model(ckpt_path: str, device: str):
    """
    Load weights from checkpoint and replace the LAST encoder layer's
    self-attention with InstrumentedMHA so we can read attention maps.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    sd    = state.get("model_state_dict", state)
    cfg   = state.get("model_config", {})

    src_vocab_size = sd["src_embed.weight"].shape[0]
    tgt_vocab_size = sd["tgt_embed.weight"].shape[0]
    d_model   = cfg.get("d_model",    sd["src_embed.weight"].shape[1])
    N         = cfg.get("N",          3)
    num_heads = cfg.get("num_heads",  8)
    d_ff      = cfg.get("d_ff",       512)
    dropout   = cfg.get("dropout",    0.1)

    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model        = d_model,
        N              = N,
        num_heads      = num_heads,
        d_ff           = d_ff,
        dropout        = dropout,
    )
    model.load_state_dict(sd)

    # Swap the last encoder layer's self_attn with an instrumented one
    last_layer = model.encoder.layers[-1]
    instr = InstrumentedMHA(d_model, num_heads, dropout)
    # Copy weights from the original W_q, W_k, W_v, W_o
    instr.W_q.weight.data = last_layer.self_attn.W_q.weight.data.clone()
    instr.W_q.bias.data   = last_layer.self_attn.W_q.bias.data.clone()
    instr.W_k.weight.data = last_layer.self_attn.W_k.weight.data.clone()
    instr.W_k.bias.data   = last_layer.self_attn.W_k.bias.data.clone()
    instr.W_v.weight.data = last_layer.self_attn.W_v.weight.data.clone()
    instr.W_v.bias.data   = last_layer.self_attn.W_v.bias.data.clone()
    instr.W_o.weight.data = last_layer.self_attn.W_o.weight.data.clone()
    instr.W_o.bias.data   = last_layer.self_attn.W_o.bias.data.clone()
    last_layer.self_attn  = instr

    # Attach vocab look-up from checkpoint
    src_stoi = state.get("src_vocab", {})
    tgt_stoi = state.get("tgt_vocab", {})
    model._src_stoi  = src_stoi
    model._tgt_itos  = {i: t for t, i in tgt_stoi.items()}
    model._vocabs_loaded = True

    model.to(device).eval()
    return model, src_stoi, d_model, num_heads


# ══════════════════════════════════════════════════════════════════════
#  Tokenize a German sentence using the saved vocab
# ══════════════════════════════════════════════════════════════════════

def tokenize_sentence(sentence: str, src_stoi: dict, device: str):
    tokens  = sentence.lower().split()
    ids     = [sos_idx] + [src_stoi.get(t, 0) for t in tokens] + [eos_idx]
    src     = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)
    token_labels = ["<sos>"] + tokens + ["<eos>"]
    return src, src_mask, token_labels


# ══════════════════════════════════════════════════════════════════════
#  Plot one heatmap per head
# ══════════════════════════════════════════════════════════════════════

def plot_head_heatmaps(
    attn_weights: torch.Tensor,   # (1, num_heads, S, S)
    token_labels: List[str],
    num_heads: int,
) -> List:
    """Returns a list of wandb.Image objects, one per head."""
    imgs = []
    attn = attn_weights[0].numpy()   # (H, S, S)

    cols = min(4, num_heads)
    rows = math.ceil(num_heads / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4, rows * 3.5),
                             squeeze=False)

    for h in range(num_heads):
        r, c = divmod(h, cols)
        ax   = axes[r][c]
        im   = ax.imshow(attn[h], vmin=0, vmax=attn[h].max(),
                         cmap="Blues", aspect="auto")
        ax.set_title(f"Head {h+1}", fontsize=9)
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(token_labels)))
        ax.set_yticklabels(token_labels, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused axes
    for h in range(num_heads, rows * cols):
        r, c = divmod(h, cols)
        axes[r][c].set_visible(False)

    plt.suptitle("Last Encoder Layer — Attention Weights per Head",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    path = "attention_heads.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    imgs.append(wandb.Image(path, caption="All heads — last encoder layer"))
    return imgs


# ══════════════════════════════════════════════════════════════════════
#  Head-specialization analysis (printed summary)
# ══════════════════════════════════════════════════════════════════════

def analyze_heads(attn: np.ndarray, token_labels: List[str]) -> dict:
    """
    For each head compute:
      - diagonal_score   : how much each token attends to itself
      - next_token_score : how much each token attends to the next one
      - entropy          : average attention entropy (low = sharp / specialized)
    """
    results = {}
    S = len(token_labels)
    for h in range(attn.shape[0]):
        A = attn[h]                                         # (S, S)
        diag  = float(np.mean(np.diag(A)))
        nxt   = float(np.mean([A[i, i+1] for i in range(S-1)])) if S > 1 else 0.0
        # Shannon entropy per row, averaged
        ent   = float(-np.mean(np.sum(A * np.log(A + 1e-9), axis=-1)))
        results[h] = dict(diagonal=diag, next_token=nxt, entropy=ent)
        print(f"  Head {h+1:2d}  diag={diag:.3f}  next={nxt:.3f}  entropy={ent:.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

# A fixed German sentence from the Multi30k test set
SAMPLE_DE = "ein mann mit einem orangefarbenen hut , der etwas anstarrt ."

def main(ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    model, src_stoi, d_model, num_heads = load_instrumented_model(ckpt_path, device)

    src, src_mask, token_labels = tokenize_sentence(SAMPLE_DE, src_stoi, device)
    print(f"\nSentence: {SAMPLE_DE}")
    print(f"Tokens  : {token_labels}")

    # Forward pass — attention weights stored inside the instrumented layer
    with torch.no_grad():
        _ = model.encode(src, src_mask)

    attn_weights = model.encoder.layers[-1].self_attn.last_attn_weights
    # attn_weights: (1, num_heads, S, S)

    run = wandb.init(
        project = "da6401-a3",
        name    = "ablation_attention_heads",
        config  = {"d_model": d_model, "num_heads": num_heads,
                   "sentence": SAMPLE_DE},
        reinit  = True,
    )

    print(f"\nHead specialization (last encoder layer):")
    head_stats = analyze_heads(attn_weights[0].numpy(), token_labels)

    # Log per-head scalar metrics
    for h, stats in head_stats.items():
        wandb.log({
            f"head_analysis/head{h+1}_diagonal"   : stats["diagonal"],
            f"head_analysis/head{h+1}_next_token" : stats["next_token"],
            f"head_analysis/head{h+1}_entropy"    : stats["entropy"],
        })

    # Log heatmap grid
    imgs = plot_head_heatmaps(attn_weights, token_labels, num_heads)
    wandb.log({"attention_heatmaps": imgs})
    print(f"\nHeatmap saved and logged to W&B.")

    # Also log individual per-head images for drill-down
    attn_np = attn_weights[0].numpy()
    for h in range(num_heads):
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(attn_np[h], vmin=0, vmax=attn_np[h].max(),
                       cmap="Blues", aspect="auto")
        ax.set_title(f"Head {h+1} — entropy={head_stats[h]['entropy']:.3f}")
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(token_labels)))
        ax.set_yticklabels(token_labels, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        p = f"head_{h+1}.png"
        plt.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        wandb.log({f"head_{h+1}": wandb.Image(p,
                   caption=f"Head {h+1} | diag={head_stats[h]['diagonal']:.3f} "
                           f"next={head_stats[h]['next_token']:.3f}")})

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", default="checkpoints/checkpoint_epoch19.pt",
        help="Path to your best saved checkpoint"
    )
    args = parser.parse_args()
    main(args.ckpt)