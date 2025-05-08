import numpy as np
from Logger import logger


class ALRScheduler:
    def __init__(self, optimizer, base_lr, min_lr, max_lr, smoothing=0.25):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.smoothing = smoothing
        self.current_lr = base_lr

        # Track running statistics for normalization
        self.grad_norm_avg = 1.0  # Initial guess
        self.act_norm_avg = 1.0  # Initial guess
        self.norm_smoothing = 0.9  # For running average of norms

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def update_learning_rate(self, grad_norm, act_norm):
        """
        Simplified hybrid LR update using moving averages and sigmoid scaling.
        """
        # Validate inputs
        if not (np.isfinite(grad_norm) and np.isfinite(act_norm)):
            logger.warning("Invalid norms detected, skipping learning rate update")
            return

        # Update moving averages
        self.grad_norm_avg = self.norm_smoothing * self.grad_norm_avg + (
            1 - self.norm_smoothing
        ) * max(grad_norm, 1e-6)
        self.act_norm_avg = self.norm_smoothing * self.act_norm_avg + (
            1 - self.norm_smoothing
        ) * max(act_norm, 1e-6)

        # Normalize by historical average
        grad_norm_norm = grad_norm / max(self.grad_norm_avg, 1e-6)
        act_norm_norm = act_norm / max(self.act_norm_avg, 1e-6)

        # Sigmoid-bounded scores
        s_g = self.sigmoid(grad_norm_norm)
        s_a = self.sigmoid(act_norm_norm)

        # Simple average of both scores
        combined_score = 0.98 * s_g + 0.02 * s_a

        # Inverse scale: higher combined_score -> lower lr
        lr_scale = 1.0 / (1.0 + combined_score)

        # Compute raw LR and clamp
        raw_lr = self.base_lr * lr_scale
        raw_lr = max(self.min_lr, min(self.max_lr, raw_lr))

        # Smooth the learning rate update
        self.current_lr = (
            1 - self.smoothing
        ) * self.current_lr + self.smoothing * raw_lr

        # Apply to optimizer
        for group in self.optimizer.param_groups:
            group["lr"] = self.current_lr

        logger.debug(
            f"LR update: {self.current_lr:.6f} | "
            f"grad_norm={grad_norm:.4f}, act_norm={act_norm:.4f}, "
            f"s_g={s_g:.4f}, s_a={s_a:.4f}, "
            f"grad_avg={self.grad_norm_avg:.4f}, act_avg={self.act_norm_avg:.4f}"
        )
