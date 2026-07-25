"""Entry point: wires RVMDataset -> VideoSiamMAE -> Trainer and runs training (JAX/Flax).

Usage:
    python src_jax/trainer.py --config configs/train_jax.yaml [--resume]
"""

import argparse
import random
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_jax.datasets.rvm_dataset import RVMDataset
from src_jax.models.rvm_jax import RVMConfig
from src_jax.optimisation.config import TrainerConfig
from src_jax.optimisation.optimizer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def build_dataloader(dataset_config, dataloader_config, shuffle: bool):
    if dataset_config is None:
        return None
    dataset = RVMDataset(dataset_config)
    return DataLoader(
        dataset,
        batch_size=dataloader_config.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=dataloader_config.get("num_workers", 4),
        pin_memory=False,  # pinned memory is a CUDA/torch-tensor concept, unused once we go to numpy/JAX
        drop_last=shuffle,
        collate_fn=RVMDataset.collate_fn,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_jax.yaml")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint_dir/last.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config.get("seed") is not None:
        set_seed(config["seed"])

    train_loader = build_dataloader(config["dataset"]["train"], config["dataloader"], shuffle=True)
    eval_loader = build_dataloader(config["dataset"].get("eval"), config["dataloader"], shuffle=False)

    model_kwargs = dict(config.get("model", {}))
    if isinstance(model_kwargs.get("dtype"), str):
        model_kwargs["dtype"] = getattr(jnp, model_kwargs["dtype"])
    model_config = RVMConfig(**model_kwargs)
    trainer_config = TrainerConfig(**config.get("trainer", {}))

    dataset_max_delta = config["dataset"]["train"].get("max_delta", 64)
    assert dataset_max_delta <= 64, (
        f"dataset max_delta={dataset_max_delta} > 64: the JAX model's delta embedding "
        "one-hots to a fixed depth of 64 (see VideoSiamMAE.reconstruct)."
    )

    trainer = Trainer(model_config, train_loader, eval_loader, trainer_config)

    if args.resume:
        resumed = trainer.resume()
        print("resumed from checkpoint" if resumed else "no checkpoint found, starting fresh")

    trainer.train()


if __name__ == "__main__":
    main()
