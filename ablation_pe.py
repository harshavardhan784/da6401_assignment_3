"""
ablation_pe.py  —  Section 2.4: Positional Encoding vs. Learned Embeddings
DA6401 Assignment 3

Trains two identical models:
  1. Sinusoidal PE  (PositionalEncoding in model.py)
  2. Learned PE     (nn.Embedding — trainable positional parameters)

Compares validation BLEU after every epoch and logs to W&B.

NOTE: This is now a thin wrapper around run_training_experiment() in
      train.py which already accepts the learned_pos_enc flag.  No code
      duplication required.

Run:
    python ablation_pe.py
"""

import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from train import run_training_experiment

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
    max_len      = 100,
    use_scaling  = True,
    fixed_lr     = None,
)


def main():
    # ── Variant 1: Sinusoidal PE (baseline) ───────────────────────────
    run_training_experiment(
        **CFG,
        learned_pos_enc = False,
        run_name        = "ablation_pe_sinusoidal",
    )

    # ── Variant 2: Learned PE ─────────────────────────────────────────
    run_training_experiment(
        **CFG,
        learned_pos_enc = True,
        run_name        = "ablation_pe_learned",
    )

    print("\nDone. In W&B overlay:")
    print("  val_bleu  for sinusoidal  vs  learned")
    print("  val_loss  for sinusoidal  vs  learned")


if __name__ == "__main__":
    main()