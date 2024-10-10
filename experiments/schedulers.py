# Straight from code implementation for MP-JEPA, Laird et al.

import numpy as np

class CosineDecayScheduler:
    def __init__(
        self,
        max_val,
        min_val,
        warmup_steps,
        total_steps,
        momentum_scheduling=False,
    ):
        self.max_val = max_val
        self.min_val = min_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.momentum_scheduling = momentum_scheduling

    def get(self, step):
        if step < self.warmup_steps:
            if self.momentum_scheduling:
                return 0.0
            # Linear warmup from 0 to max_val
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            # Cosine decay from max_val to min_val
            cosine_decay = 0.5 * (
                1
                + np.cos(
                    (step - self.warmup_steps)
                    * np.pi
                    / (self.total_steps - self.warmup_steps)
                )
            )
            return self.min_val + (self.max_val - self.min_val) * cosine_decay
        else:
            raise ValueError(
                "Step ({}) > total number of steps ({}).".format(
                    step, self.total_steps
                )
            )


class CosineDecayWithRestartsScheduler:
    def __init__(
        self,
        max_val,
        min_val,
        warmup_steps,
        total_steps,
        restart_period,
        dampening=True,
        momentum_scheduling=False,
    ):
        self.max_val = max_val
        self.min_val = min_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.restart_period = restart_period
        self.dampening = dampening
        self.num_restarts = total_steps // restart_period
        self.momentum_scheduling = momentum_scheduling

    def get(self, step):
        if step < self.warmup_steps:
            if self.momentum_scheduling:
                return 0.0
            # Linear warmup from 0 to max_val
            return self.max_val * step / self.warmup_steps
        elif step <= self.total_steps:
            # Calculate the current cycle number
            cycle = step // self.restart_period
            # Calculate step position within the current cycle
            cycle_step = step % self.restart_period

            if cycle_step == 0 and cycle != 0:
                cycle_step = self.restart_period

            # Calculate decay base for the current cycle
            base = self.min_val + (self.max_val - self.min_val) * (
                0.5**cycle if self.dampening else 1
            )

            # Cosine decay from current base to min_val
            cosine_decay = 0.5 * (
                1 + np.cos(np.pi * cycle_step / self.restart_period)
            )
            return self.min_val + (base - self.min_val) * cosine_decay
        else:
            raise ValueError(
                f"Step ({step}) > total number of steps ({self.total_steps})."
            )