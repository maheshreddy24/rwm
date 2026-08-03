import argparse
import dataclasses
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from icecream import ic

# torch (and anything importing torchvision) must load before jax:
# libtriton.so and jaxlib each bundle their own LLVM, and whichever
# dlopens second binds the wrong symbols and segfaults.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import jax
import jax.numpy as jnp
import numpy as np
import wandb
import yaml
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.readout_head import Readout
from models.rvm_jax import build_model
from ssv2_inf_dataset import SSv2


#! params from the 4D scaling paper (Carreira et al.), optimization fixed across all tasks/models:
#! 1.28M training examples, batch size 32 -> 40k steps, AdamW, wd 1e-4,
#! LR swept over {1e-4, 3e-4, 1e-3}, 1k-step linear warmup, cosine decay to 1e-7.
#! here the same budget is expressed in epochs: num_epochs * len(train_loader)
#! optimizer updates, with warmup_ratio of those spent on linear warmup.


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


def _warmup_cosine(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    """Multiplicative LR factor (relative to `base_lr`): linear warmup, then cosine decay to `min_lr`.

    Still called once per optimizer update -- the LR schedule stays smooth within
    an epoch; only logging/eval/checkpointing are driven by the epoch counter.
    """
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = min_lr + (base_lr - min_lr) * cosine
    return lr / base_lr


def _build_forward_fn(rvm_model):
    """Jitted, pure wrapper around the frozen RVM's `reconstruct` method.

    `rvm_model` is closed over (a flax Module is a frozen dataclass, not a jit
    argument), so the only traced inputs are params/source/target/deltas/rng --
    this is what makes jitting safe here. Jitting the bound `Trainer.forward`
    method directly (the previous approach) doesn't work: `self` would be
    passed as the first traced argument, and a `Trainer` instance isn't a
    registered pytree.
    """

    def _forward(params, source, target, deltas, rng):
        out = rvm_model.apply(
            {"params": params},
            source, target, deltas,
            method=rvm_model.reconstruct,
            rngs={"default": rng},
        )
        # cast on-device: bfloat16 arrays don't cross the jax->numpy->torch
        # boundary cleanly, so land on a real float32 before returning.
        return out["features"].astype(jnp.float32)

    return jax.jit(_forward)


@dataclass
class TrainConfig:
    device: str = "cuda:0"
    seed: int = 42

    dataset_config_path: str = ""
    restored_params_path: str = ""

    lr: float = 3e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    num_epochs: int = 20
    warmup_ratio: float = 0.025  # 1k warmup steps out of a 40k-step budget
    min_lr: float = 1e-7

    batch_size: int = 32
    num_workers: int = 8
    amp: bool = True

    log_interval: int = 1      # in epochs
    eval_interval: int = 1     # in epochs
    checkpoint_dir: str = "checkpoints_ssv2_acr"

    wandb_project: str = "rvm-action-recog"
    wandb_name: str = "ssv2-readout"

    readout_in_dim: int = 384
    readout_dim: int = 768
    readout_num_heads: int = 12
    readout_num_classes: int = 174

    def __post_init__(self):
        # PyYAML parses bare exponent floats (e.g. "3e-4") as strings, not floats --
        # coerce explicitly rather than relying on the yaml file's formatting.
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
        eval_loader: Optional[DataLoader] = None,
        config: TrainConfig = None,
        resume: str = None
    ):
        self.config = config
        self.device = torch.device(config.device)
        self.device_type = self.device.type
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.checkpoint_dir = os.path.join(
            self.config.checkpoint_dir, f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- frozen RVM (jax): encoder/core/decoder weights never change ---
        self.rng_key = jax.random.PRNGKey(self.config.seed)
        self.rvm_encoder = build_model()
        restored = recover_tree(
            np.load(self.config.restored_params_path, allow_pickle=False)
        )
        # upload once so every training step reuses device arrays instead of
        # re-transferring host numpy arrays on every call.
        self.restored_params = jax.tree_util.tree_map(jnp.asarray, restored)
        self._forward_fn = _build_forward_fn(self.rvm_encoder)
        self.epochs = self.config.num_epochs

        count = sum(np.prod(v.shape) for v in jax.tree_util.tree_leaves(self.restored_params))
        print(f"number of params for RVM encoder = {count}")

        # --- readout head (torch): the only thing being optimized ---
        self.model = Readout(
            in_dim=self.config.readout_in_dim,
            dim=self.config.readout_dim,
            num_heads=self.config.readout_num_heads,
            num_frames=16,  # fixed by the SSv2 protocol (paper Table 6), matches dataset num_frames
            num_classes=self.config.readout_num_classes,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(enabled=self.config.amp and self.device_type == "cuda")
        self.current_epoch = 0
        self.init_optim()

        if resume is not None:
            self.load_checkpoint(resume)
            self.epochs += 2

    def _extract_representation(self, frames_np: np.ndarray) -> torch.Tensor:
        """frames_np: (B, T, H, W, C) clip -> (B, T, N, 384) frozen backbone features."""
        B, T, H, W, C = frames_np.shape
        source = frames_np                                       # (B, T, H, W, C)
        target = frames_np[:, -1:, :, :, :]                      # (B, 1, H, W, C)  <- keep the axis
        deltas = np.zeros((B, 1), dtype=np.int32)                # (B, 1), matches Tt=1

        self.rng_key, step_key = jax.random.split(self.rng_key)
        representation = self._forward_fn(
            self.restored_params, source, target, deltas, step_key
        )
        return torch.from_numpy(np.asarray(representation))

    def init_optim(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
            eps=self.config.eps,
        )
        # full optimizer budget = every batch of every epoch
        total_steps = self.epochs * len(self.train_loader)
        warmup_steps = int(self.config.warmup_ratio * total_steps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine(
                step,
                warmup_steps,
                total_steps,
                self.config.lr,
                self.config.min_lr,
            ),
        )

        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            config=dataclasses.asdict(self.config),
        )

    def save_checkpoint(self, epoch: int, name: str = "last.pt"):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
                "config": dataclasses.asdict(self.config),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str, load_optim: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if load_optim and ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        # resume from the epoch after the one that was saved
        self.current_epoch = ckpt.get("epoch", -1) + 1
        return ckpt

    def train(self):
        self.model.train()

        global_step = self.current_epoch * len(self.train_loader)

        for epoch in tqdm(range(self.current_epoch, self.epochs), desc="Epochs", leave=True):
            self.current_epoch = epoch

            running_loss = 0.0
            num_batches = 0

            for step, (frames, labels) in tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch}",
                leave=False,
            ):
                representation = self._extract_representation(
                    frames.detach().cpu().numpy()
                ).to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast(
                    device_type=self.device_type,
                    enabled=self.config.amp,
                ):
                    logits = self.model(representation)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                running_loss += loss.item()
                num_batches += 1
                global_step += 1

                # Step-based logging
                if global_step % self.config.log_interval == 0:
                    wandb.log(
                        {
                            "train/loss_step": loss.item(),
                            "train/lr": self.scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                    print(
                        f"[epoch {epoch} step {global_step}] "
                        f"loss {loss.item():.4f}"
                    )

                # Step-based evaluation
                if (
                    self.eval_loader is not None
                    and global_step % self.config.eval_interval == 0
                ):
                    self.evaluate()
                    self.model.train()

            # Epoch statistics
            epoch_loss = running_loss / max(1, num_batches)

            wandb.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "epoch": epoch,
                },
                step=global_step,
            )

            print(f"[epoch {epoch}] average train loss: {epoch_loss:.4f}")

            # Save checkpoint every epoch
            self.save_checkpoint(epoch=epoch, name=f"epoch_{epoch}.pt")

        self.save_checkpoint(epoch=self.epochs - 1, name="final.pt")
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for frames, labels in tqdm(self.eval_loader, desc="Eval", leave=False):
            representation = self._extract_representation(
                frames.detach().cpu().numpy()
            ).to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast(device_type=self.device_type, enabled=self.config.amp):
                logits = self.model(representation)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * frames.size(0)
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_samples += frames.size(0)

        avg_loss = total_loss / max(1, total_samples)
        accuracy = total_correct / max(1, total_samples)
        wandb.log(
            {"eval/loss": avg_loss, "eval/accuracy": accuracy, "epoch": self.current_epoch},
            step=self.current_epoch,
        )
        print(f"[epoch {self.current_epoch}] eval loss {avg_loss:.4f} acc {accuracy:.4f}")
        return avg_loss, accuracy


def build_dataloaders(config: TrainConfig):
    train_set = SSv2(config.dataset_config_path, split="train")
    eval_set = SSv2(config.dataset_config_path, split="validation")

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, eval_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_config.yaml"),
    )
    parser.add_argument(
        "--resume_path",
        type = str,
        default = None,
        required = False
    )
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