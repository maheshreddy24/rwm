import math
import os
import time
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
import wandb

from src.models.rvm import patchify
from .config import TrainerConfig


def _warmup_cosine(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    """Multiplicative LR factor (relative to `base_lr`): linear warmup, then cosine decay to `min_lr`."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = min_lr + (base_lr - min_lr) * cosine
    return lr / base_lr


def _unpack_batch(batch):
    """Batch -> (source, target, target_deltas). Accepts a dict or a 3-tuple/list."""
    if isinstance(batch, dict):
        return batch["source"], batch["target"], batch["target_deltas"]
    source, target, target_deltas = batch
    return source, target, target_deltas


class Trainer:
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
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.checkpoint_dir = os.path.join(self.config.checkpoint_dir, f'exp_{time.time()}')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(enabled=self.config.amp and self.device == "cuda")
        self.epoch = 0
        self.global_step = 0    

        self.init_optim()

    def init_optim(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
            betas=tuple(float(b) for b in self.config.betas),
            eps=float(self.config.eps),
        )

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * int(float(self.config.num_epochs))
        warmup_steps = steps_per_epoch * int(float(self.config.warmup_epochs))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine(
                step, warmup_steps, total_steps, float(self.config.lr), float(self.config.min_lr)
            ),
        )

        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            config=asdict(self.config),
        )

    def save_checkpoint(self, name: str = "last.pt"):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "config": asdict(self.config),
            },
            path,
        )
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
        return ckpt

    def resume(self, name: str = "last.pt"):
        path = os.path.join(self.checkpoint_dir, name)
        if not os.path.exists(path):
            return False
        self.load_checkpoint(path)
        return True

    def train(self, num_epochs: Optional[int] = None):
        num_epochs = num_epochs or int(float(self.config.num_epochs))
        for _ in range(num_epochs):
            self.model.train()
            for batch in self.train_loader:
                loss = self._train_step(batch)
                self.global_step += 1
                if self.global_step % int(float(self.config.log_interval)) == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"epoch {self.epoch} step {self.global_step} loss {loss:.4f} lr {lr:.2e}")
                    wandb.log({"train/loss": loss, "train/lr": lr, "epoch": self.epoch}, step=self.global_step)

                if self.eval_loader is not None and self.global_step % int(float(self.config.eval_interval)) == 0:
                    self._evaluate_and_log()
                    self.model.train()

            self.epoch += 1
            self.save_checkpoint()

            if self.eval_loader is not None:
                self._evaluate_and_log()

    def _evaluate_and_log(self):
        eval_loss = self.eval()
        print(f"epoch {self.epoch} step {self.global_step} eval_loss {eval_loss:.4f}")
        wandb.log({"eval/loss": eval_loss, "epoch": self.epoch}, step=self.global_step)
        self._save_patch_visualizations()

    def _train_step(self, batch):
        source, target, target_deltas = _unpack_batch(batch)
        source = source.to(self.device)
        target = target.to(self.device)
        target_deltas = target_deltas.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device, enabled=self.config.amp):
            out = self.model(source, target, target_deltas)
            loss = self.model.loss(
                out, target, masked_only=self.config.masked_only, norm_pix=self.config.norm_pix
            )

        self.scaler.scale(loss).backward()
        if self.config.grad_clip_norm is not None:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        for batch in self.eval_loader:
            source, target, target_deltas = _unpack_batch(batch)
            source = source.to(self.device)
            target = target.to(self.device)
            target_deltas = target_deltas.to(self.device)

            out = self.model(source, target, target_deltas)
            loss = self.model.loss(
                out, target, masked_only=self.config.masked_only, norm_pix=self.config.norm_pix
            )
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _save_patch_visualizations(self):
        """Save original vs. reconstructed crops for a random sample of masked patches.

        Picks patches from the first frame of the first eval batch's first sample,
        un-normalizing predictions with the ground-truth patch stats when
        `norm_pix` is enabled (the model regresses to normalized patches in that mode).
        """
        if self.eval_loader is None:
            return

        was_training = self.model.training
        self.model.eval()

        batch = next(iter(self.eval_loader))
        source, target, target_deltas = _unpack_batch(batch)
        source = source.to(self.device)
        target = target.to(self.device)
        target_deltas = target_deltas.to(self.device)

        out = self.model(source, target, target_deltas)

        patch = self.model.patch
        gt = patchify(target[0, 0].unsqueeze(0), patch)[0]      # (N, patch*patch*3)
        pred = out["pred"][0, 0].float()                        # (N, patch*patch*3)
        mask = out["mask"][0, 0]                                 # (N,)  1 = masked

        if self.config.norm_pix:
            mu = gt.mean(dim=-1, keepdim=True)
            var = gt.var(dim=-1, keepdim=True)
            pred = pred * (var + 1e-6).sqrt() + mu

        masked_idx = (mask > 0.5).nonzero(as_tuple=True)[0]
        if masked_idx.numel() == 0:
            if was_training:
                self.model.train()
            return

        k = min(int(float(self.config.num_vis_patches)), masked_idx.numel())
        chosen = masked_idx[torch.randperm(masked_idx.numel(), device=masked_idx.device)[:k]]

        originals = [self._patch_to_pil(gt[i], patch) for i in chosen.tolist()]
        reconstructions = [self._patch_to_pil(pred[i], patch) for i in chosen.tolist()]
        grid = self._make_comparison_grid(originals, reconstructions, patch)

        vis_dir = os.path.join(self.checkpoint_dir, "image_vis")
        os.makedirs(vis_dir, exist_ok=True)
        grid.save(os.path.join(vis_dir, f"{self.epoch}_{self.global_step}.png"))

        if was_training:
            self.model.train()

    @staticmethod
    def _patch_to_pil(patch_vec: torch.Tensor, patch: int, scale: int = 8) -> Image.Image:
        arr = patch_vec.reshape(patch, patch, 3).clamp(0, 1).mul(255).byte().cpu().numpy()
        img = Image.fromarray(arr, mode="RGB")
        return img.resize((patch * scale, patch * scale), Image.NEAREST)

    @staticmethod
    def _make_comparison_grid(originals, reconstructions, patch: int, pad: int = 4) -> Image.Image:
        cell = originals[0].width
        n = len(originals)
        width = n * cell + (n + 1) * pad
        height = 2 * cell + 3 * pad
        canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
        for i, img in enumerate(originals):
            canvas.paste(img, (pad + i * (cell + pad), pad))
        for i, img in enumerate(reconstructions):
            canvas.paste(img, (pad + i * (cell + pad), 2 * pad + cell))
        return canvas
