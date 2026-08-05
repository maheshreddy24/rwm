import glob
import logging
import math
import os
import time
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from .config import TrainerConfig


def get_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(f"rvm_trainer.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(fmt)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def _warmup_cosine_factor(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    """Multiplicative LR factor, shared across param groups: linear warmup to 1.0,
    then cosine decay to `min_lr_ratio`."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _unpack_batch(batch):
    """Batch -> (source, target, target_deltas). Accepts a dict or a 3-tuple/list."""
    if isinstance(batch, dict):
        return batch["source"], batch["target"], batch["target_deltas"]
    return batch


class EMAAvgFn:
    """multi_avg_fn for `AveragedModel`: decay ramps up to `target_decay` over
    `warmup_steps`, then holds constant."""

    def __init__(self, target_decay: float = 0.999, warmup_steps: int = 2000):
        self.target_decay = target_decay
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def __call__(self, ema_param_list, current_param_list, num_averaged):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            decay = min(self.target_decay, (1 + self.step_count) / (10 + self.step_count))
        else:
            decay = self.target_decay
        torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)


class Trainer:
    """Trains RVM with a representation-space loss: the decoder predicts, for
    masked target patches, the representation an EMA copy of the vision encoder
    produces from the *unmasked* target frame."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
    ):
        self.config = config or TrainerConfig()
        self.device = self.config.device
        self.model = model.to(self.device)

        # EMA teacher for the vision encoder only -- produces representation targets.
        self.ema_fn = EMAAvgFn(self.config.momentum_decay, self.config.momentum_warmup_steps)
        self.ema_model = AveragedModel(self.model.encoder, multi_avg_fn=self.ema_fn)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"exp_{time.time()}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.checkpoint_dir, "train.log"))

        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(enabled=self.config.amp and self.device == "cuda")
        self.epoch = 0
        self.global_step = 0

        self.init_optim()

    def init_optim(self):
        encoder_params = list(self.model.encoder.parameters())
        other_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder.")]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": float(self.config.encoder_lr)},
                {"params": other_params, "lr": float(self.config.lr)},
            ],
            weight_decay=float(self.config.weight_decay),
            betas=tuple(float(b) for b in self.config.betas),
            eps=float(self.config.eps),
        )

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * int(self.config.num_epochs)
        warmup_steps = int(round(total_steps * float(self.config.warmup_ratio)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine_factor(
                step, warmup_steps, total_steps, float(self.config.min_lr_ratio)
            ),
        )

        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            config=asdict(self.config),
        )

    def _log_params(self):
        self.logger.info("=== TrainerConfig ===")
        for k, v in asdict(self.config).items():
            self.logger.info(f"  {k}: {v}")
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"model params: {n_params:,} total, {n_trainable:,} trainable")

    def save_checkpoint(self):
        name = f"model_epoch{self.epoch}_step{self.global_step}.pth"
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "ema_step_count": self.ema_fn.step_count,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "config": asdict(self.config),
            },
            path,
        )
        self.logger.info(f"saved checkpoint: {path}")
        return path

    def load_checkpoint(self, path: str, load_optim: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if ckpt.get("ema_model") is not None:
            self.ema_model.load_state_dict(ckpt["ema_model"])
        self.ema_fn.step_count = ckpt.get("ema_step_count", 0)
        if load_optim and ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        return ckpt

    def resume(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "model_epoch*_step*.pth"))
        if not checkpoints:
            return False
        self.load_checkpoint(max(checkpoints, key=os.path.getmtime))
        return True

    def train(self, num_epochs: Optional[int] = None):
        num_epochs = num_epochs or int(self.config.num_epochs)
        self._log_params()
        for _ in tqdm(range(num_epochs), total=num_epochs, leave=False):
            self.model.train()
            for batch in tqdm(self.train_loader, total=len(self.train_loader), leave=True):
                loss = self._train_step(batch)
                self.global_step += 1

                if self.global_step % int(self.config.log_interval) == 0:
                    lr = self.optimizer.param_groups[-1]["lr"]
                    self.logger.info(f"epoch {self.epoch} step {self.global_step} loss {loss:.4f} lr {lr:.2e}")
                    wandb.log({"train/loss": loss, "train/lr": lr, "epoch": self.epoch}, step=self.global_step)

                if self.global_step % int(self.config.save_every_steps) == 0:
                    self.save_checkpoint()

                if self.eval_loader is not None and self.global_step % int(self.config.eval_every_steps) == 0:
                    self._evaluate_and_log()

            self.epoch += 1

    def _evaluate_and_log(self):
        eval_loss = self.eval()
        self.logger.info(f"epoch {self.epoch} step {self.global_step} eval_loss {eval_loss:.4f}")
        wandb.log({"eval/loss": eval_loss, "epoch": self.epoch}, step=self.global_step)
        self.model.train()

    def _teacher_representation(self, target: torch.Tensor) -> torch.Tensor:
        """EMA-encoder representation of the *unmasked* target frames.
        target: (B, Tt, 3, H, W) -> (B, Tt, 1+N, D)."""
        B, Tt = target.shape[:2]
        flat = target.reshape(B * Tt, *target.shape[2:])
        repr_ = self.ema_model(flat).last_hidden_state  # (B*Tt, 1+N, D)
        if self.config.normalize_target:
            repr_ = F.layer_norm(repr_, repr_.shape[-1:])
        return repr_.view(B, Tt, -1, repr_.shape[-1])

    @staticmethod
    def representation_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked MSE between predicted and target per-token representations.
        pred/target: (B, Tt, N, D), mask: (B, Tt, N), 1 = masked (supervised)."""
        per_token = (pred.float() - target.float()).pow(2).mean(dim=-1)
        return (per_token * mask).sum() / mask.sum().clamp(min=1.0)

    def _train_step(self, batch):
        source, target, target_deltas = _unpack_batch(batch)
        source = source.to(self.device)   # (B, Ts, C, H, W)
        target = target.to(self.device)   # (B, Tt, C, H, W)
        target_deltas = target_deltas.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device, enabled=self.config.amp):
            out = self.model(source, target, target_deltas)
            with torch.no_grad():
                teacher = self._teacher_representation(target)
            loss = self.representation_loss(
                out["decoded_representation"][..., 1:, :],
                teacher[..., 1:, :],
                out["mask"],
            )
            # print(f"loss computed {loss.item()}")

        self.scaler.scale(loss).backward()
        if self.config.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.ema_model.update_parameters(self.model.encoder)
        # print('loss.step()')
        return loss.item()

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for batch in tqdm(self.eval_loader, total = len(self.eval_loader), leave = True):
            # print("inside the eval function")
            source, target, target_deltas = _unpack_batch(batch)
            source = source.to(self.device)
            target = target.to(self.device)
            target_deltas = target_deltas.to(self.device)

            out = self.model(source, target, target_deltas)
            teacher = self._teacher_representation(target)
            loss = self.representation_loss(
                out["decoded_representation"][..., 1:, :],
                teacher[..., 1:, :],
                out["mask"],
            )
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
