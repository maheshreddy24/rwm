from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TrainerConfig:
    """Hyperparameters for `Trainer`. Pass an instance (or overrides via kwargs) in.

    Mirrors `src/optimisation/config.py::TrainerConfig`. There's no `device`/`amp` field:
    JAX picks up whatever `jax.devices()` returns, and mixed precision is controlled by
    `RVMConfig.dtype` (the model computes in that dtype while params/grads stay fp32),
    so no autocast/GradScaler-equivalent is needed.
    """

    # runtime
    checkpoint_dir: str = "checkpoints"
    seed: Optional[int] = None
    init_params_path: Optional[str] = None  # .npz of pretrained params (see optimizer.load_pretrained_params)

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

    # loss -- kwargs forwarded to rvm_loss(); paper default is plain L2 over all pixels
    masked_only: bool = False
    norm_pix: bool = False

    # misc
    log_interval: int = 50
    eval_interval: int = 1000

    # wandb
    wandb_project: str = "rvm"
    wandb_name: Optional[str] = None
