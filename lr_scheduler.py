# lr_scheduler.py
"""
Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           https://arxiv.org/abs/1706.03762

Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


# ══════════════════════════════════════════════════════════════════════
#  NOAM SCHEDULER  (ablation 2.1 baseline)
# ══════════════════════════════════════════════════════════════════════

class NoamScheduler(LRScheduler):
    """
    Noam learning rate schedule from "Attention Is All You Need".

    Linear warm-up for `warmup_steps` steps, then inverse-square-root decay.

    Formula:
        lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Usage:
        optimizer = Adam(model.parameters(), lr=1.0, ...)
        scheduler = NoamScheduler(optimizer, d_model=256, warmup_steps=4000)
        # After each batch: optimizer.step(); scheduler.step()

    Args:
        optimizer    : Wrapped optimizer. Set lr=1.0 so base_lr scales as 1.
        d_model      : Model dimensionality (embedding size).
        warmup_steps : Steps before decay begins.
        last_epoch   : Index of last epoch (default: -1).
    """

    def __init__(
        self,
        optimizer:    optim.Optimizer,
        d_model:      int,
        warmup_steps: int,
        last_epoch:   int = -1,
    ) -> None:
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def _get_lr_scale(self) -> float:
        # last_epoch starts at -1 and is incremented to 0 on the first
        # scheduler.step() call, so effective step = last_epoch + 1.
        step  = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5),
        )
        return scale

    def get_lr(self) -> list:
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


# ══════════════════════════════════════════════════════════════════════
#  FIXED-LR SCHEDULER  (ablation 2.1 comparison)
# ══════════════════════════════════════════════════════════════════════

class FixedLRScheduler(LRScheduler):
    """
    No-op scheduler: keeps the optimizer's learning rate exactly constant.
    Used in the Noam vs Fixed-LR ablation (Part 2.1).
    """

    def __init__(self, optimizer: optim.Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list:
        return list(self.base_lrs)


# ══════════════════════════════════════════════════════════════════════
#  UTILITY — simulate and return LR history
# ══════════════════════════════════════════════════════════════════════

def get_lr_history(
    d_model:      int,
    warmup_steps: int,
    total_steps:  int,
) -> list:
    """
    Simulate the Noam schedule for `total_steps` steps and return the
    per-step learning rate as a list.
    """
    dummy_model = torch.nn.Linear(1, 1)
    optimizer   = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler   = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return history


# ══════════════════════════════════════════════════════════════════════
#  QUICK VISUALISATION
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL      = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS  = 20_000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("noam_lr_schedule.png", dpi=150)
    plt.show()