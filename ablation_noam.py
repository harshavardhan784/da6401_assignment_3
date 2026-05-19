"""
ablation_noam.py  —  Section 2.1: The Necessity of the Noam Scheduler
DA6401 Assignment 3

Trains two identical models:
  1. With    Noam LR schedule  (linear warmup → inverse-sqrt decay)
  2. Without Noam  →  fixed LR = 1e-4

Logs train loss, val loss, and current LR to W&B for both runs so you
can overlay them in a single W&B report panel.

Run:
    python ablation_noam.py

NOTE: This is a thin wrapper around run_training_experiment() in train.py.
      No code duplication — all ablation options are flags in that function.
"""

import wandb
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
)


def main():
    # ── Variant 1: Noam scheduler ─────────────────────────────────────
    run_training_experiment(
        **CFG,
        fixed_lr    = None,       # None → Noam schedule
        use_scaling = True,
        run_name    = "ablation_noam_noam_scheduler",
    )

    # ── Variant 2: Fixed LR (no warmup) ──────────────────────────────
    run_training_experiment(
        **CFG,
        fixed_lr    = 1e-4,       # constant LR, no warmup
        use_scaling = True,
        run_name    = "ablation_noam_fixed_lr",
    )

    print("\nDone. In W&B, add a Line Chart panel and plot:")
    print("  train_loss (both runs overlaid)")
    print("  val_loss   (both runs overlaid)")
    print("  lr         (shows warmup curve for Noam run)")


if __name__ == "__main__":
    main()