# ablation_2_3.py
"""
Ablation 2.3 — Attention Rollout & Head Specialization
DA6401 Assignment 3: "Attention Is All You Need"

What this script does:
  1. Loads a trained checkpoint.
  2. Picks one German sentence from the test set.
  3. Runs a forward pass through the encoder to capture attention weights
     from EVERY encoder layer (all heads).
  4. Logs a per-head heatmap for the LAST encoder layer to W&B.
  5. Computes Attention Rollout across all encoder layers and logs it.
  6. Prints a short head-specialization analysis (next-token, diagonal,
     long-range, redundancy) to stdout and logs it as a W&B Table.

Usage:
    python ablation_2_3.py --checkpoint checkpoints/<run>_epoch<N>.pt
                           [--sentence "Ein Mann sitzt auf einer Bank ."]
                           [--wandb_project da6401-a3]
                           [--wandb_run    attn-rollout]
"""

import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — saves PNGs, does not open a window
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

import wandb

from dataset import Multi30kDataset, sos_idx, eos_idx, pad_idx
from model   import Transformer, make_src_mask


# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD MODEL FROM CHECKPOINT
# ══════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: str) -> Transformer:
    """Reconstruct model from a checkpoint saved by save_checkpoint()."""
    state = torch.load(ckpt_path, map_location="cpu")
    cfg   = state["model_config"]

    model = Transformer(
        src_vocab_size = cfg["src_vocab_size"],
        tgt_vocab_size = cfg["tgt_vocab_size"],
        d_model        = cfg["d_model"],
        N              = cfg["N"],
        num_heads      = cfg["num_heads"],
        d_ff           = cfg["d_ff"],
        dropout        = cfg["dropout"],
        pos_encoding   = cfg.get("pos_encoding", "sinusoidal"),
        use_scale      = cfg.get("use_scale", True),
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model.to(device)


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — TOKENISE ONE SENTENCE & RUN ENCODER
# ══════════════════════════════════════════════════════════════════════

def encode_sentence(
    model:     Transformer,
    sentence:  str,
    src_vocab,
    device:    str,
) -> tuple:
    """
    Tokenise a German sentence, run it through the encoder,
    and return (tokens, src_tensor, all_layer_attn_weights).

    Returns
    -------
    tokens          : list[str]  — tokenised source words (no BOS/EOS)
    src_ids         : Tensor [1, S]
    layer_attn      : list of Tensor [num_heads, S, S]
                      one entry per encoder layer, last-layer first is
                      index [-1].
    """
    import spacy
    nlp    = spacy.load("de_core_news_sm")
    tokens = [tok.text.lower() for tok in nlp.tokenizer(sentence)]

    ids    = (
        [sos_idx]
        + [src_vocab.lookup_index(t) for t in tokens]
        + [eos_idx]
    )
    src      = torch.tensor([ids], dtype=torch.long, device=device)
    src_mask = make_src_mask(src, pad_idx=pad_idx).to(device)

    # Forward pass — attention weights collected inside MHA via self.attn_weights
    with torch.no_grad():
        _ = model.encode(src, src_mask)

    # Harvest weights from every encoder layer  [num_heads, S, S]
    layer_attn = [
        layer.self_attn.attn_weights[0].cpu()   # remove batch dim
        for layer in model.encoder.layers
    ]

    # Tokens shown on axes: add BOS/EOS labels so positions match
    display_tokens = ["<sos>"] + tokens + ["<eos>"]
    return display_tokens, src, layer_attn


# ══════════════════════════════════════════════════════════════════════
#  STEP 3 — PER-HEAD HEATMAPS  (last encoder layer)
# ══════════════════════════════════════════════════════════════════════

def plot_head_heatmaps(
    attn:   torch.Tensor,   # [num_heads, S, S]
    tokens: list,
    layer_idx: int,
) -> plt.Figure:
    """
    Draw one subplot per attention head for a single encoder layer.

    Returns a matplotlib Figure (caller logs it to W&B).
    """
    num_heads = attn.shape[0]
    ncols     = min(4, num_heads)
    nrows     = math.ceil(num_heads / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 4 * nrows),
        squeeze=False,
    )
    fig.suptitle(f"Encoder Layer {layer_idx} — Per-Head Attention", fontsize=14)

    for h in range(num_heads):
        row, col = divmod(h, ncols)
        ax       = axes[row][col]
        weights  = attn[h].numpy()          # [S, S]

        im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"Head {h}", fontsize=9)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(tokens, fontsize=6)
        ax.set_xlabel("Key position", fontsize=7)
        ax.set_ylabel("Query position", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused subplots
    for h in range(num_heads, nrows * ncols):
        row, col = divmod(h, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ══════════════════════════════════════════════════════════════════════
#  STEP 4 — ATTENTION ROLLOUT
# ══════════════════════════════════════════════════════════════════════

def attention_rollout(layer_attn: list) -> np.ndarray:
    """
    Compute Attention Rollout (Abnar & Zuidema, 2020) across all
    encoder layers.

    For each layer we average over heads, add the identity (residual),
    re-normalise, then multiply through all layers.

    Returns:
        rollout : np.ndarray [S, S]  — propagated attention from each
                  token to every other token after all layers.
    """
    result = None
    for attn in layer_attn:                      # attn: [num_heads, S, S]
        avg   = attn.mean(dim=0).numpy()         # [S, S]  — average heads
        I     = np.eye(avg.shape[0])
        joint = avg + I                          # add residual connection
        joint = joint / joint.sum(axis=-1, keepdims=True)   # re-normalise
        result = joint if result is None else np.matmul(joint, result)
    return result


def plot_rollout(rollout: np.ndarray, tokens: list) -> plt.Figure:
    """Plot the rolled-out attention as a single heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(rollout, cmap="Purples", vmin=0, aspect="auto")
    ax.set_title("Attention Rollout (all encoder layers)", fontsize=12)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Source token (key)", fontsize=9)
    ax.set_ylabel("Source token (query)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
#  STEP 5 — HEAD SPECIALIZATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyse_heads(
    attn:   torch.Tensor,   # [num_heads, S, S]  — last encoder layer
    tokens: list,
) -> list:
    """
    Heuristically categorise each head in the last encoder layer:

    • "next-token"   — peak attention consistently one position ahead
    • "diagonal"     — peak on the diagonal (self-attention / copy)
    • "long-range"   — average attention distance > S/2
    • "uniform"      — entropy close to log(S) — attends everywhere equally
    • "other"        — no dominant pattern

    Returns a list of dicts, one per head, for a W&B Table.
    """
    S        = attn.shape[-1]
    log_S    = math.log(S) if S > 1 else 1.0
    rows     = []

    for h in range(attn.shape[0]):
        w          = attn[h].numpy()              # [S, S]
        peak_cols  = w.argmax(axis=-1)            # [S] — argmax key for each query

        # next-token: peak_col == query_pos + 1
        next_tok   = float((peak_cols == np.arange(S) + 1).mean())

        # diagonal: peak_col == query_pos
        diag_frac  = float((peak_cols == np.arange(S)).mean())

        # average attention distance
        positions  = np.arange(S)
        dist       = np.abs(
            w * (positions[None, :] - positions[:, None])
        ).sum(axis=-1).mean()
        long_range = float(dist > S / 2)

        # entropy (higher = more uniform)
        w_safe  = np.clip(w, 1e-9, 1.0)
        entropy = float(-(w_safe * np.log(w_safe)).sum(axis=-1).mean())
        uniform = entropy / log_S             # normalised 0-1

        # assign label
        if next_tok > 0.4:
            label = "next-token"
        elif diag_frac > 0.5:
            label = "diagonal (copy)"
        elif long_range:
            label = "long-range"
        elif uniform > 0.85:
            label = "uniform (redundant?)"
        else:
            label = "other"

        rows.append({
            "head":           h,
            "next_tok_frac":  round(next_tok, 3),
            "diagonal_frac":  round(diag_frac, 3),
            "avg_dist":       round(float(dist), 3),
            "norm_entropy":   round(uniform, 3),
            "pattern":        label,
        })

    return rows


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Ablation 2.3 — Attention Rollout & Head Specialization")
    p.add_argument("--checkpoint",     type=str, required=True,
                   help="Path to a trained checkpoint (.pt)")
    p.add_argument("--sentence",       type=str,
                   default="Ein Mann sitzt auf einer Bank .",
                   help="German sentence to visualise")
    p.add_argument("--layer",          type=int, default=-1,
                   help="Encoder layer index for per-head heatmaps (-1 = last)")
    p.add_argument("--wandb_project",  type=str, default="da6401-a3")
    p.add_argument("--wandb_run",      type=str, default="ablation-2-3-attn-rollout")
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── W&B ───────────────────────────────────────────────────────────
    wandb.init(
        project = args.wandb_project,
        name    = args.wandb_run,
        config  = {"checkpoint": args.checkpoint, "sentence": args.sentence},
    )

    # ── Load model & vocab ────────────────────────────────────────────
    print("Loading model …")
    model = load_model(args.checkpoint, device)

    # Reconstruct src_vocab from the checkpoint
    state     = torch.load(args.checkpoint, map_location="cpu")
    from dataset import Vocab
    src_vocab = Vocab(state["src_vocab"])

    # ── Encode sentence & collect attention weights ───────────────────
    print(f'Encoding: "{args.sentence}"')
    tokens, src_ids, layer_attn = encode_sentence(
        model, args.sentence, src_vocab, device
    )
    num_layers = len(layer_attn)
    layer_idx  = args.layer if args.layer >= 0 else num_layers + args.layer
    print(f"Collected attention from {num_layers} encoder layers.")
    print(f"Tokens ({len(tokens)}): {tokens}")

    # ── Per-head heatmaps (chosen layer) ─────────────────────────────
    print(f"Plotting per-head heatmaps for encoder layer {layer_idx} …")
    fig_heads = plot_head_heatmaps(layer_attn[layer_idx], tokens, layer_idx)
    fig_heads.savefig("attn_heads.png", dpi=150, bbox_inches="tight")
    wandb.log({"attention/per_head_heatmap": wandb.Image("attn_heads.png")})
    plt.close(fig_heads)

    # ── Attention Rollout ─────────────────────────────────────────────
    print("Computing attention rollout …")
    rollout    = attention_rollout(layer_attn)          # [S, S]
    fig_roll   = plot_rollout(rollout, tokens)
    fig_roll.savefig("attn_rollout.png", dpi=150, bbox_inches="tight")
    wandb.log({"attention/rollout_heatmap": wandb.Image("attn_rollout.png")})
    plt.close(fig_roll)

    # ── Head specialization table ─────────────────────────────────────
    print("Analysing head specialization …")
    head_rows = analyse_heads(layer_attn[layer_idx], tokens)

    print(f"\n{'Head':>4}  {'Pattern':<22}  {'next_tok':>8}  {'diag':>6}  {'avg_dist':>8}  {'entropy':>7}")
    print("-" * 65)
    for r in head_rows:
        print(
            f"{r['head']:>4}  {r['pattern']:<22}  "
            f"{r['next_tok_frac']:>8.3f}  {r['diagonal_frac']:>6.3f}  "
            f"{r['avg_dist']:>8.3f}  {r['norm_entropy']:>7.3f}"
        )

    # Log as W&B Table
    table = wandb.Table(
        columns=["head", "pattern", "next_tok_frac", "diagonal_frac", "avg_dist", "norm_entropy"]
    )
    for r in head_rows:
        table.add_data(
            r["head"], r["pattern"],
            r["next_tok_frac"], r["diagonal_frac"],
            r["avg_dist"],      r["norm_entropy"],
        )
    wandb.log({"attention/head_specialization": table})

    # ── Also log per-layer averaged heatmaps for completeness ─────────
    print("Logging averaged heatmaps for all layers …")
    for i, attn in enumerate(layer_attn):
        avg_attn = attn.mean(dim=0).numpy()   # [S, S]
        fig, ax  = plt.subplots(figsize=(6, 5))
        im = ax.imshow(avg_attn, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"Encoder Layer {i} — Head-Averaged Attention", fontsize=10)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        fname = f"attn_layer{i}_avg.png"
        fig.savefig(fname, dpi=120, bbox_inches="tight")
        wandb.log({f"attention/layer{i}_avg": wandb.Image(fname)})
        plt.close(fig)

    wandb.finish()
    print("\nDone. All plots logged to W&B.")


if __name__ == "__main__":
    main()