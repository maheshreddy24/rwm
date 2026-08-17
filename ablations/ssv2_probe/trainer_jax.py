"""SSv2 action-recognition ablation: RVM (jax VideoSiamMAE) frozen backbone + MeanPoolProbe readout.

Usage:
    python ablations/ssv2_probe/trainer_jax.py --config ablations/ssv2_probe/config_jax.yaml
"""

import argparse
import dataclasses
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torch and everything that dlopens LLVM must load before jax: libtriton.so and
# jaxlib each bundle their own, and whichever loads second binds the wrong
# symbols and segfaults. `import torch` alone is NOT enough -- torch._dynamo and
# triton load lazily, and get dragged in later by torchvision (via
# ssv2_inf_dataset), i.e. after jax. So force them now.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision  # noqa: F401  -- pulls torch._dynamo -> triton
import torch._dynamo  # noqa: F401  -- belt and braces

import jax
import jax.numpy as jnp
import numpy as np
import wandb
import yaml
from tqdm import tqdm

from common import MeanPoolProbe, get_logger, warmup_cosine_factor, CNNProbe
from rvm_jax import build_model
from ssv2_inf_dataset import SSv2


def recover_tree(flat_dict):
    """Un-flatten a `{'a/b/c': array}` dict (as saved in the restored npz) into the
    nested dict pytree flax expects for `apply({'params': ...})`."""
    tree = {}
    for k, v in flat_dict.items():
        parts = k.split("/")
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = v
    return tree


def _build_forward_fn(rvm_model):
    """Jitted, pure wrapper around the frozen RVM's `reconstruct` method.

    `rvm_model` is closed over (a flax Module is a frozen dataclass, not a jit
    argument), so the only traced inputs are params/source/target/deltas/rng.
    """

    def _forward(params, source, target, deltas, rng):
        out = rvm_model.apply(
            {"params": params},
            source, target, deltas,
            method=rvm_model.reconstruct,
            rngs={"default": rng},
        )
        return out["features"][..., 1:, :].astype(jnp.float32)  # drop CLS token: (B, Ts, N, F)

    return jax.jit(_forward)


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda:0"

    dataset_config_path: str = ""
    restored_params_path: str = ""

    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    num_epochs: int = 2
    warmup_ratio: float = 0.05
    min_lr: float = 1e-7

    batch_size: int = 32
    num_workers: int = 16
    amp: bool = True

    log_interval: int = 200
    eval_interval: int = 1000
    checkpoint_dir: str = "ckpts/ssv2_meanpool_probe_jax"

    wandb_project: str = "rvm-action-recog"
    wandb_name: str = "ssv2-meanpool-probe-jax"

    num_classes: int = 174
    feature_dim: int = 384

    def __post_init__(self):
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.betas = tuple(float(b) for b in self.betas)
        self.eps = float(self.eps)
        self.min_lr = float(self.min_lr)
        self.num_epochs = int(self.num_epochs)
        self.warmup_ratio = float(self.warmup_ratio)
        self.batch_size = int(self.batch_size)
        self.num_workers = int(self.num_workers)
        self.log_interval = int(self.log_interval)
        self.eval_interval = int(self.eval_interval)
        self.seed = int(self.seed)
        self.num_classes = int(self.num_classes)
        self.feature_dim = int(self.feature_dim)


def load_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in {path}: {sorted(unknown)}")
    return TrainConfig(**raw)


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader],
        config: TrainConfig,
        resume: str = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.device_type = self.device.type
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.checkpoint_dir = os.path.join(
            config.checkpoint_dir, f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.checkpoint_dir, "training.log"))

        self.rng_key = jax.random.PRNGKey(config.seed)
        self.backbone = build_model()
        restored = recover_tree(np.load(config.restored_params_path, allow_pickle=False))
        self.backbone_params = jax.tree_util.tree_map(jnp.asarray, restored)
        self._forward_fn = _build_forward_fn(self.backbone)
        count = sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(self.backbone_params))
        self.logger.info(f"backbone params: {count:,}")

        # self.model = MeanPoolProbe(dim=config.feature_dim, num_classes=config.num_classes).to(self.device)
        num_patches = 16 * 256
        self.model = CNNProbe(inp_channels=num_patches, num_classes=config.num_classes, dim = config.feature_dim).to(self.device)
        self.logger.info(f"probe params: {sum(p.numel() for p in self.model.parameters()):,}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(enabled=config.amp and self.device_type == "cuda")
        self.epoch = 0
        self.global_step = 0
        self.init_optim()

        if resume is not None:
            self.load_checkpoint(resume)

    def _extract_features(self, frames_np: np.ndarray) -> torch.Tensor:
        """frames_np: (B, T, C, H, W) -> (B, T, N, F) frozen backbone features, CLS dropped."""
        frames_np = np.transpose(frames_np, (0, 1, 3, 4, 2))  # (B, T, H, W, C), channels-last for jax
        B = frames_np.shape[0]
        source = frames_np
        target = frames_np[:, -1:]
        deltas = np.zeros((B, 1), dtype=np.int32)

        self.rng_key, step_key = jax.random.split(self.rng_key)
        features = self._forward_fn(self.backbone_params, source, target, deltas, step_key)
        return torch.from_numpy(np.asarray(features))

    def init_optim(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )
        total_steps = self.config.num_epochs * len(self.train_loader)
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        min_lr_ratio = self.config.min_lr / self.config.lr
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_cosine_factor(step, warmup_steps, total_steps, min_lr_ratio),
        )
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            config=dataclasses.asdict(self.config),
        )

    def save_checkpoint(self, name: str):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "config": dataclasses.asdict(self.config),
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
        if ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt.get("epoch", -1) + 1
        self.global_step = ckpt.get("global_step", 0)
        return ckpt

    def train(self):
        for epoch in tqdm(range(self.epoch, self.config.num_epochs), desc="Epochs"):
            self.epoch = epoch
            self.model.train()
            running_loss, num_batches = 0.0, 0

            for frames, labels in tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False):
                features = self._extract_features(frames.detach().cpu().numpy()).to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast(device_type=self.device_type, enabled=self.config.amp):
                    logits = self.model(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                running_loss += loss.item()
                num_batches += 1
                self.global_step += 1

                if self.global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    wandb.log({"train/loss_step": loss.item(), "train/lr": lr, "epoch": epoch}, step=self.global_step)
                    self.logger.info(f"epoch {epoch} step {self.global_step} loss {loss.item():.4f} lr {lr:.2e}")

                if self.eval_loader is not None and self.global_step % self.config.eval_interval == 0:
                    self.evaluate()
                    self.model.train()

            epoch_loss = running_loss / max(1, num_batches)
            wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch}, step=self.global_step)
            self.logger.info(f"epoch {epoch} average train loss: {epoch_loss:.4f}")

            self.save_checkpoint(name=f"epoch_{epoch}.pt")
            self.evaluate()

        self.save_checkpoint(name="final.pt")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for frames, labels in tqdm(self.eval_loader, desc="Eval", leave=False):
            features = self._extract_features(frames.detach().cpu().numpy()).to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast(device_type=self.device_type, enabled=self.config.amp):
                logits = self.model(features)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * frames.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += frames.size(0)

        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        wandb.log({"eval/loss": avg_loss, "eval/accuracy": accuracy}, step=self.global_step)
        self.logger.info(f"epoch {self.epoch} eval loss {avg_loss:.4f} acc {accuracy:.4f}")
        return avg_loss, accuracy


def build_dataloaders(config: TrainConfig):
    train_set = SSv2(config.dataset_config_path, split="train")
    eval_set = SSv2(config.dataset_config_path, split="validation")

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_set, batch_size=config.batch_size//4, shuffle=False,
        num_workers=config.num_workers//2,
    )
    return train_loader, eval_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_jax.yaml"),
    )
    parser.add_argument("--resume_path", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_loader, eval_loader = build_dataloaders(config)
    trainer = Trainer(train_loader, eval_loader, config, args.resume_path)
    trainer.train()


if __name__ == "__main__":
    main()
