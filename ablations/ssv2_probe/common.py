import logging
import math

import torch.nn as nn


def get_logger(log_path):
    logger = logging.getLogger(f"ssv2_ablation.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    """Multiplicative LR factor: linear warmup to 1.0, then cosine decay to `min_lr_ratio`."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


class MeanPoolProbe(nn.Module):
    """Ablation readout: mean-pool backbone features over time and tokens, then classify."""

    def __init__(self, dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):          # x: (bs, t, n, dim)
        x = x.mean(dim=(1, 2))     # (bs, dim) -- global avg over time and space
        return self.head(self.norm(x))
