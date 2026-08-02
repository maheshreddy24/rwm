from dataclasses import dataclass, field
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
    checkpoint_dir: str = "checkpoints_ema_jax"
    seed: Optional[int] = None
    init_params_path: Optional[str] = None  # .npz of pretrained params (see optimizer.load_pretrained_params)

    # optimizer -- AdamW with the betas/wd used by MAE-style ViT recipes
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    grad_clip_norm: Optional[float] = 1.0

    # schedule: linear warmup -> cosine decay, stepped every optimizer step
    num_epochs: int = 2
    warmup_steps: int = 500
    min_lr: float = 1e-6

    # loss -- kwargs forwarded to rvm_loss(); paper default is plain L2 over all pixels
    masked_only: bool = False
    norm_pix: bool = False

    # EMA target encoder (optimizer_ema.py only): momentum update `ema = m*ema + (1-m)*params`,
    # applied every `ema_update_every` optimizer steps.
    ema_momentum: float = 0.998
    ema_update_every: int = 1

    # misc
    log_interval: int = 50
    eval_interval: int = 1000

    # wandb
    wandb_project: str = "rvm"
    wandb_name: Optional[str] = None


@dataclass
class EMATrainerConfig:
    """Hyperparameters for the EMA-teacher `Trainer` in `optimizer_ema.py`.

    Unlike `TrainerConfig`, budget and schedule are expressed in steps (dataloader
    batches), not epochs: `total_steps` is the number of batches to consume, and
    `warmup_ratio`/`ema_ramp` are fractions of that budget. `grad_accum` groups
    every `grad_accum` batches into one optimizer update (via `optax.MultiSteps`),
    so the LR/EMA schedules — which tick once per optimizer update — see
    `total_steps // grad_accum` steps, not `total_steps`.
    """

    # runtime
    checkpoint_dir: str = "checkpoints_ema_jax"
    seed: Optional[int] = None
    init_params_path: Optional[str] = None

    # budget
    total_steps: int = 30000
    grad_accum: int = 1

    # optimizer -- AdamW with the betas/wd used by MAE-style ViT recipes
    lr_peak: float = 5.0e-5
    lr_min: float = 5.0e-7
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    weight_decay: float = 0.05
    wd_skip: Tuple[str, ...] = field(default_factory=lambda: ("bias", "norm", "cls_token", "pos_embed"))
    grad_clip: Optional[float] = 1.0

    # schedule: linear warmup -> cosine decay, as a fraction of total_steps
    warmup_ratio: float = 0.05

    # EMA target encoder: momentum ramps from `ema_momentum` to `ema_momentum_end`
    # following `ema_ramp` ("cosine" or "constant") over the full run.
    ema_momentum: float = 0.999
    ema_momentum_end: float = 0.9999
    ema_ramp: str = "cosine"
    ema_dtype: str = "float32"

    # loss -- kwargs forwarded to representation_loss()
    loss_on_masked_only: bool = True

    # misc
    log_interval: int = 50
    eval_interval: int = 1000
    collapse_interval: int = 500

    # wandb
    wandb_project: str = "rvm"
    wandb_name: Optional[str] = None
