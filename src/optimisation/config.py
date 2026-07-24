from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class TrainerConfig:
    """Hyperparameters for `Trainer`. Pass an instance (or overrides via kwargs) in."""

    # runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    seed: Optional[int] = None

    # optimizer -- AdamW with the betas/wd used by MAE-style ViT recipes
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0

    # schedule: linear warmup -> cosine decay, stepped every optimizer step
    num_epochs: int = 100
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # loss -- kwargs forwarded to RVM.loss(); paper default is plain L2 over all pixels
    masked_only: bool = False
    norm_pix: bool = False

    # misc
    amp: bool = False
    log_interval: int = 50
