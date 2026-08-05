from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TrainerConfig:
    """Hyperparameters for `Trainer`. Pass an instance (or overrides via kwargs) in.

    `num_epochs` is a count of full passes over `train_loader`; `warmup_ratio` is a
    fraction of the resulting epochs*steps_per_epoch budget. `save_every_steps` and
    `eval_every_steps` are counted in optimizer steps, not epochs.
    """

    # runtime
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    seed: Optional[int] = None

    # optimizer -- AdamW, two param groups: the DINO vision encoder gets a much
    # lower lr than the recurrent core / decoder, which are trained from scratch.
    lr: float = 1.5e-4          # recurrent core + decoder + repr_head
    encoder_lr: float = 1.5e-5  # DINO vision encoder (backbone)
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0

    # schedule: linear warmup -> cosine decay to `min_lr_ratio` * lr, per group
    num_epochs: int = 5
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.01

    # EMA teacher: exponential moving average of the vision encoder, used to
    # produce the target representation for masked target patches.
    momentum_decay: float = 0.999
    momentum_warmup_steps: int = 2000
    normalize_target: bool = True  # layernorm (no affine) the EMA representation

    # cadence, in optimizer steps. During the first epoch eval runs every step;
    # after that it runs every 2 * eval_every_steps (see Trainer.train).
    save_every_steps: int = 1000
    eval_every_steps: int = 1000

    # misc
    amp: bool = False
    log_interval: int = 50  # console/log-file print frequency, in steps

    # wandb
    wandb_project: str = "rvm_dinov2"
    wandb_name: Optional[str] = None
