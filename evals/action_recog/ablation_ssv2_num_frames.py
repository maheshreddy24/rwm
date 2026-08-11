import argparse
import dataclasses
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# torch and everything that dlopens LLVM must load before jax: libtriton.so and
# jaxlib each bundle their own, and whichever loads second binds the wrong
# symbols and segfaults. `import torch` alone is NOT enough -- torch._dynamo and
# triton load lazily, and get dragged in later by torchvision (via
# ssv2_inf_dataset / readout_head), i.e. after jax. So force them now.
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import torchvision  # noqa: F401  -- pulls torch._dynamo -> triton
# import torch._dynamo  # noqa: F401  -- belt and braces

import numpy as np
import wandb
import yaml
from icecream import ic
from tqdm import tqdm

from models.readout_head import Readout
from models.rvm_torch import RVM # I guess all the parameters are hardcodeed.
from ssv2_inf_dataset import SSv2

#! params from the 4D scaling paper (Carreira et al.), optimization fixed across all tasks/models:
#! 1.28M training examples, batch size 32 -> 40k steps, AdamW, wd 1e-4,
#! LR swept over {1e-4, 3e-4, 1e-3}, 1k-step linear warmup, cosine decay to 1e-7.
#! here the same budget is expressed in epochs: num_epochs * len(train_loader)
#! optimizer updates, with warmup_ratio of those spent on linear warmup.

import logging

def get_logger(log_path):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# per-epoch checkpoints are named "nf{num_frames}_epoch_{epoch}.pt" (see
# Trainer.train); resuming parses this to know which ablation setting and
# epoch to pick up from.
CHECKPOINT_NAME_RE = re.compile(r"nf(?P<nf>\d+)_epoch_(?P<epoch>\d+)\.pt$")


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


@dataclass
class TrainConfig:
    device: str = "cuda:0"
    seed: int = 42

    dataset_config_path: str = ""
    restored_params_path: str = ""     # JAX checkpoint (.npz), used by the JAX trainer
    rvm_weights_path: str = ""         # torch state_dict (.pt) for models.rvm_torch.RVM

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
    checkpoint_dir: str = "checkpoints_ssv2_acr_rvm_dino"

    wandb_project: str = "rvm-dino-action-recog"
    wandb_name: str = "ssv2-readout"

    readout_in_dim: int = 384 # dino embedding dim varies acc to the variant. 
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
        # so the number of frames we want to retain after the encoding,
        # this will give us an idea of how good the recurrent state is.
        self.num_frames = [16, 13, 10, 7, 4, 1]
        self.abl_iterations = len(self.num_frames)

        # resuming continues inside the same experiment dir as the checkpoint,
        # instead of starting a fresh exp_<timestamp> dir
        self.resume_path = resume
        self.resume_nf = None
        if resume is not None:
            self.checkpoint_dir = os.path.dirname(os.path.abspath(resume))
            match = CHECKPOINT_NAME_RE.search(os.path.basename(resume))
            if not match:
                raise ValueError(
                    f"can't parse num_frames/epoch from checkpoint name: {resume} "
                    f"(expected nf<N>_epoch_<E>.pt)"
                )
            self.resume_nf = int(match.group("nf"))
        else:
            self.checkpoint_dir = os.path.join(
                self.config.checkpoint_dir, f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
            )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(os.path.join(self.checkpoint_dir, "training.log"))

        self.rd_encoder = RVM(encoder_name="facebook/dinov2-small")
        if self.config.rvm_weights_path:
            state_dict = torch.load(self.config.rvm_weights_path, map_location="cpu")
            self.rd_encoder.load_state_dict(state_dict['model'])
            self.logger.info(f"loaded RVM weights from {self.config.rvm_weights_path}")
        # freeze the backbone
        for p in self.rd_encoder.parameters():
            p.requires_grad = False
        self.rd_encoder = self.rd_encoder.to(self.device)
        self.rd_encoder = self.rd_encoder.eval()
        self.epochs = self.config.num_epochs

        count = sum(p.numel() for p in self.rd_encoder.parameters())
        self.logger.info(f"number of params for RVM encoder = {count}")

        # ! this has to be inti in the loop.
        # --- readout head (torch): the only thing being optimized ---
        # self.model = Readout(
        #     in_dim=self.config.readout_in_dim,
        #     dim=self.config.readout_dim,
        #     num_heads=self.config.readout_num_heads,
        #     num_frames=16,  # fixed by the SSv2 protocol (paper Table 6), matches dataset num_frames
        #     num_classes=self.config.readout_num_classes,
        # ).to(self.device)
        # count = sum(p.numel() for p in self.model.parameters())
        # self.logger.info(f"number of params for readout head = {count}")


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(enabled=self.config.amp and self.device_type == "cuda")
        self.current_epoch = 0
        # self.init_optim()
        # per ablation setting: accuracy logged at each eval step, and the
        # global step it was logged at (same length/step-grid for every
        # setting since epochs and the train_loader are shared across them)
        self.plot_acc_data = [[] for _ in range(self.abl_iterations)]
        self.plot_step_data = [[] for _ in range(self.abl_iterations)]

    def _extract_representation(self, frames: torch.Tensor, retain_idx: int) -> torch.Tensor:
        bs, t, c, h, w = frames.shape
        source = frames.to(self.device)
        target = frames[:, -1:, :, :, :].to(self.device)  # (B, 1, C, H, W)
        deltas = torch.zeros((bs, 1), dtype=torch.int64, device=self.device)  # (B, 1), matches Tt=1

        # with torch.no_grad():
        #     output = self.rd_encoder(source, target, deltas)
        with torch.no_grad(), torch.amp.autocast(device_type=self.device_type, enabled=self.config.amp):
            output = self.rd_encoder(source, target, deltas)

        memory = output["memory"][..., 1:, :]  # (B, T, N, 384), CLS token dropped
        # so based on the retain_idx we can retain M out of N last frames
        return memory[:, memory.shape[1]-retain_idx:, ...] # we return x .., end

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
            name=f"{self.config.wandb_name}-nf{self.current_nf}",
            config=dataclasses.asdict(self.config),
            reinit=True,
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

    def train_loop(self):
        # settings ordered before the resumed one are assumed already finished
        # in a prior run (their own logs/checkpoints exist there already)
        resume_idx = self.num_frames.index(self.resume_nf) if self.resume_nf is not None else None

        for idx, nf in enumerate(tqdm(self.num_frames, desc="frame ablation")):
            if resume_idx is not None and idx < resume_idx:
                self.logger.info(f"skipping num_frames = {nf} (already completed before resume)")
                continue

            self.logger.info(f'')
            self.logger.info(f"New iteration")
            self.current_nf = nf
            self.current_epoch = 0

            self.model = Readout(
                in_dim=self.config.readout_in_dim,
                dim=self.config.readout_dim,
                num_heads=self.config.readout_num_heads,
                num_frames=nf,  # so we ablate on number of frames
                num_classes=self.config.readout_num_classes,
            ).to(self.device)

            n_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"num_frames = {nf}, iteration = {idx}, readout params = {n_params}")

            self.init_optim()

            if idx == resume_idx:
                self.load_checkpoint(self.resume_path)
                self.logger.info(
                    f"resumed num_frames = {nf} from {self.resume_path}, "
                    f"starting at epoch {self.current_epoch}"
                )

            self.train(idx)
            wandb.finish()

        self._save_ablation_results()

    def _save_ablation_results(self):
        # called after every eval point, so curves across settings are rarely
        # the same length (in progress vs. finished vs. not-yet-started) --
        # keep one array per num_frames setting instead of stacking them.

        # 1) num_frames (x) vs latest eval accuracy (y)
        latest_accs = [
            self.plot_acc_data[i][-1] if self.plot_acc_data[i] else float("nan")
            for i in range(self.abl_iterations)
        ]
        frames_vs_acc = np.array(list(zip(self.num_frames, latest_accs)))
        np.save(os.path.join(self.checkpoint_dir, "frames_vs_acc.npy"), frames_vs_acc)

        # 2) step (x) vs accuracy (y) curve per num_frames setting
        curves = {}
        for i, nf in enumerate(self.num_frames):
            curves[f"nf{nf}_steps"] = np.array(self.plot_step_data[i])
            curves[f"nf{nf}_acc"] = np.array(self.plot_acc_data[i])
        np.savez(os.path.join(self.checkpoint_dir, "steps_vs_acc.npz"), **curves)

    def train(self, indx: int):
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
                    frames, retain_idx=self.num_frames[indx]
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
                    self.logger.info(
                        f"[epoch {epoch} step {global_step}] "
                        f"loss {loss.item():.4f}"
                    )

                # Step-based evaluation
                if (
                    self.eval_loader is not None
                    and global_step % self.config.eval_interval == 0
                ):
                    _, acc = self.evaluate(retain_idx=self.num_frames[indx])
                    self.plot_acc_data[indx].append(acc)
                    self.plot_step_data[indx].append(global_step)
                    self._save_ablation_results()
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

            self.logger.info(f"[epoch {epoch}] average train loss: {epoch_loss:.4f}")

            # Save checkpoint every epoch
            self.save_checkpoint(epoch=epoch, name=f"nf{self.num_frames[indx]}_epoch_{epoch}.pt")
            self.evaluate(retain_idx=self.num_frames[indx])

        self.save_checkpoint(epoch=self.epochs - 1, name=f"nf{self.num_frames[indx]}_final.pt")
    @torch.no_grad()
    def evaluate(self, retain_idx: int):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for frames, labels in tqdm(self.eval_loader, desc="Eval", leave=False):
            representation = self._extract_representation(
                frames, retain_idx=retain_idx  # bs, t, c, h, w
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
            {"eval/loss": avg_loss, "eval/accuracy": accuracy},
            # step=self.current_epoch + 1,
        )
        self.logger.info(f"[epoch {self.current_epoch}] eval loss {avg_loss:.4f} acc {accuracy:.4f}")
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
    trainer.train_loop()


if __name__ == "__main__":
    main()