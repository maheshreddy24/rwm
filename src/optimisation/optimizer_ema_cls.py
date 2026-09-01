"""Trainer for RecurrentWorldModelCLS. Same optimisation recipe as
`optimizer_ema.Trainer` (AdamW + warmup/cosine schedule over everything but
the frozen encoder), trimmed to the one batch format
`RecurrentWorldModelCLS.forward` accepts (`context`/`frame_times`) and with no
pixel-reconstruction eval-image path, since the CLS-only model has no
`recon_head`."""

import glob
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from .optimizer_ema import get_logger, _warmup_cosine_factor


class Trainer:
    """Trains RecurrentWorldModelCLS with the representation-space loss it
    computes internally in `forward` (see `RecurrentWorldModelCLS.loss`)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: dict,
    ):
        self.config = config["trainer"]
        self.device = self.config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.min_context = self.config.get("min_context", 10)
        print(f"Device used is {self.device} ===========================")

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.checkpoint_dir = os.path.join(self.config["checkpoint_dir"], f"exp_{time.time()}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.checkpoint_dir, "train.log"))
        self.logger.info(config["trainer"])
        self.logger.info(config["model"])

        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(enabled=self.config["amp"] and self.device == "cuda")
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.init_optim()

    def init_optim(self):
        other_params = [p for n, p in self.model.named_parameters() if not n.startswith("encoder.")]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": float(self.config["lr"])},
            ],
            weight_decay=float(self.config["weight_decay"]),
            betas=tuple(float(b) for b in self.config["betas"]),
            eps=float(self.config["eps"]),
        )

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * int(self.config["num_epochs"])
        warmup_steps = int(round(total_steps * float(self.config["warmup_ratio"])))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine_factor(
                step, warmup_steps, total_steps, float(self.config["min_lr_ratio"])
            ),
        )

        wandb.init(
            project=self.config["wandb_project"],
            name=self.config["wandb_name"],
            config=self.config,
        )

    def _log_params(self):
        self.logger.info("=== trainer config ===")
        for k, v in self.config.items():
            self.logger.info(f"  {k}: {v}")

        def count(module: nn.Module) -> tuple[int, int]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        n_params, n_trainable = count(self.model)
        self.logger.info(f"model params: {n_params:,} total, {n_trainable:,} trainable")
        for name, module in self.model.named_children():
            m_params, m_trainable = count(module)
            self.logger.info(f"  {name}: {m_params:,} total, {m_trainable:,} trainable")

    def save_checkpoint(self):
        name = f"model_epoch{self.epoch}_step{self.global_step}.pth"
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "config": self.config,
                "local_step": self.local_step,
            },
            path,
        )
        self.logger.info(f"saved checkpoint: {path}")
        return path

    def load_checkpoint(self, path: str, load_optim: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if load_optim and ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.local_step = ckpt.get("local_step", 0)
        return ckpt

    def resume(self, path: Optional[str] = None):
        if path is None:
            checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "model_epoch*_step*.pth"))
            if not checkpoints:
                return False
            path = max(checkpoints, key=os.path.getmtime)
        self.load_checkpoint(path)
        self.checkpoint_dir = os.path.dirname(path)
        self.logger = get_logger(os.path.join(self.checkpoint_dir, "train.log"))
        return True

    def train(self, num_epochs: Optional[int] = None):
        num_epochs = num_epochs or int(self.config["num_epochs"])
        self._log_params()

        # local_step is the last batch index completed in self.epoch, so a resumed
        # run continues that same epoch from the next batch instead of skipping it.
        resume_step = self.local_step if self.global_step > 0 else -1
        print(f"Will resume from {resume_step}")
        if self.epoch >= num_epochs:
            self.logger.info(f"resume epoch {self.epoch} >= num_epochs {num_epochs}, nothing to train")
            return

        for _ in tqdm(range(self.epoch, num_epochs), total=num_epochs - self.epoch, leave=False):
            self.model.train()
            for step, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True):
                if step <= resume_step:
                    continue
                loss = self._train_step(batch)
                self.global_step += 1
                self.local_step = step

                if self.global_step % int(self.config["log_interval"]) == 0:
                    lr = self.optimizer.param_groups[-1]["lr"]
                    self.logger.info(f"epoch {self.epoch} step {self.global_step} loss {loss:.4f} lr {lr:.2e}")
                    wandb.log({"train/loss": loss, "train/lr": lr, "epoch": self.epoch}, step=self.global_step)

                if self.global_step % int(self.config["save_every_steps"]) == 0:
                    self.save_checkpoint()

                count = 2 if self.epoch > 1 else 1
                eval_every = int(self.config["eval_every_steps"]) * count
                if self.eval_loader is not None and self.global_step % eval_every == 0:
                    self._evaluate_and_log()

            self.epoch += 1
            resume_step = -1

    def _grad_norm(self, module: nn.Module) -> float:
        """L2 norm of the gradients currently held on `module`'s parameters.
        Called from _evaluate_and_log, before the next zero_grad(), so this
        reflects the gradients from the most recent training step."""
        total_sq = 0.0
        for p in module.parameters():
            if p.grad is not None:
                total_sq += p.grad.data.float().norm(2).item() ** 2
        return total_sq ** 0.5

    def _log_gradient_norms(self):
        core_norm = self._grad_norm(self.model.core)
        decoder_norm = self._grad_norm(self.model.decoder)
        self.logger.info(
            f"epoch {self.epoch} step {self.global_step} grad_norm/core {core_norm:.4f} grad_norm/decoder {decoder_norm:.4f}"
        )
        wandb.log(
            {"grad_norm/core": core_norm, "grad_norm/decoder": decoder_norm, "epoch": self.epoch},
            step=self.global_step,
        )

    def _evaluate_and_log(self):
        self._log_gradient_norms()
        eval_loss = self.eval()
        self.logger.info(f"epoch {self.epoch} step {self.global_step} eval_loss {eval_loss:.4f}")
        wandb.log({"eval/loss": eval_loss, "epoch": self.epoch}, step=self.global_step)
        self.model.train()

    def _forward_batch(self, batch):
        context = batch["context"].to(self.device, non_blocking=True)                       # (B, N, C, H, W)
        frame_times = batch["frame_times"].to(self.device, non_blocking=True).float()        # seconds
        target_idx = torch.arange(self.min_context, context.shape[1], device=self.device)
        out = self.model(context, target_idx, frame_times)
        return out, out["loss"]  # the loss is computed inside RecurrentWorldModelCLS

    def _train_step(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device, enabled=self.config["amp"]):
            _, loss = self._forward_batch(batch)

        self.scaler.scale(loss).backward()
        if self.config["grad_clip_norm"] is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["grad_clip_norm"]))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return loss.item()

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for step, batch in tqdm(enumerate(self.eval_loader), total=len(self.eval_loader), leave=True):
            _, loss = self._forward_batch(batch)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
