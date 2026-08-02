import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional, Tuple
import jax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.rvm_jax import build_model
from models.readout_head import Readout
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np


#! use these params from the 4d scaling paper. 
# Optimization is fixed across all tasks and models: 1.28M training examples, batch size 32 → 40k steps, AdamW, wd 1e-4, 
# LR swept over {1e-4, 3e-4, 1e-3}, 1k-step linear warmup, cosine decay to 1e-7.
def recover_tree(flat_dict):
  tree = {}
  for k, v in flat_dict.items():
    parts = k.split("/")
    node = tree
    for part in parts[:-1]:
      if part not in node:
        node[part] = {}
      node = node[part]
    node[parts[-1]] = v
  return tree

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
    """Batch -> (inputs, labels). Accepts a dict or a 2-tuple/list."""
    if isinstance(batch, dict):
        return batch["input"], batch["label"]
    inputs, labels = batch
    return inputs, labels


class Trainer:
    def __init__(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        config = None,
    ):
        self.config = config 
        self.device = self.config.device
        # self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.checkpoint_dir = os.path.join(self.config.checkpoint_dir, f"exp_{time.time()}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)


        rng_seed = 0
        self.rng_key = {'default': jax.random.PRNGKey(rng_seed)}
        self.rvm_encoder = build_model() # this will return the model
        self.restored_params = recover_tree(np.load(self.config['restored_params_path'], allow_pickle=False)) # weights of the pretrained model

        count = sum([np.prod(v.shape) for v in jax.tree_util.tree_leaves(self.restored_params)])
        print(f'number of params for RVM encoder = {count}')

        # ssv2
        self.model = Readout(in_dim = 384, dim = 768, num_heads = 12, num_frames = 16, num_classes = 174) #in_dim, dim, num_heads, num_frames, num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.scaler = torch.amp.GradScaler(enabled=self.config.amp and self.device == "cuda")
        self.epoch = 0
        self.global_step = 0

        self.init_optim()


    @jax.jit
    def forward(self, params, source, target, target_deltas):
        # 'representation': representation,   # (B, Tt, N, C) 
        return self.rvm_encoder.apply(
            {'params': params},
            source, target, target_deltas,
            method=self.rvm_encoder.reconstruct,
            rngs=self.rng_key,
        )
        
    def init_optim(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
            betas=tuple(float(b) for b in self.config.betas),
            eps=float(self.config.eps),
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _warmup_cosine(
                step,
                int(float(self.config.warmup_steps)),
                int(float(self.config.total_steps)),
                float(self.config.lr),
                float(self.config.min_lr),
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

    def train(self):
        # pass
        for epoch in tqdm(range(self.epoch, self.config.epochs), desc = f'Epochs', leave = True):
            for step, frames, labels in tqdm(enumerate(self.train_loader), desc = f'Batches', leave = True): #! check i/o shapes of rvm & model
                self.model.train()
                target_frames = frames[:, 1, :, :, :]
                target_deltas = np.array([17])
                output = self.forward(self.restored_params, frames.detach().numpy(), target_frames.detach().numpy(), target_deltas)
                representation = output['representation'] # (B, Tt, N, C)
                representation = torch.from_numpy(representation).float().to(self.device) # convert to torch tensor
                logits = self.model(representation) # (B, num_classes)
                loss = self.criterion(logits, labels.to(self.device))

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                if step % self.config.log_interval == 0:
                    wandb.log({"train/loss": loss.item(), "train/step": self.global_step})
                    print(f"Epoch [{epoch}/{self.config.epochs}] Step [{step}/{len(self.train_loader)}] Loss: {loss.item():.4f}")

                if step % self.config.eval_interval == 0 and self.eval_loader is not None:
                    self.evaluate()

            self.save_checkpoint(name=f"epoch_{epoch}.pt")


    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for step, frames, labels in tqdm(enumerate(self.eval_loader), desc = f'Eval Batches', leave = True):
                target_frames = frames[:, 1, :, :, :]
                target_deltas = np.array([17])
                output = self.forward(self.restored_params, frames.detach().numpy(), target_frames.detach().numpy(), target_deltas)
                representation = output['representation'] # (B, Tt, N, C)
                representation = torch.from_numpy(representation).float().to(self.device) # convert to torch tensor
                logits = self.model(representation) # (B, num_classes)
                loss = self.criterion(logits, labels.to(self.device))

                total_loss += loss.item() * frames.size(0)
                _, predicted = torch.max(logits.data, 1)
                total_correct += (predicted == labels.to(self.device)).sum().item()
                total_samples += frames.size(0)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        wandb.log({"eval/loss": avg_loss, "eval/accuracy": accuracy, "eval/step": self.global_step})
        print(f"Eval Loss: {avg_loss:.4f}, Eval Accuracy: {accuracy:.4f}")

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
                




