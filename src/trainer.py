"""Entry point: wires RVMDataset -> RVM -> Trainer and runs training.

Usage:
    python src/trainer.py --config configs/train.yaml [--resume]
"""

import argparse
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.rvm_dataset import RVMDataset, set_seed
from src.models.rvm import RVM
from src.optimisation.config import TrainerConfig
from src.optimisation.optimizer import Trainer
import torch
import random
import numpy as np
import os
import logging


def get_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



def build_dataloader(dataset_config, dataloader_config, shuffle: bool):
    if dataset_config is None:
        return None
    dataset = RVMDataset(dataset_config)
    return DataLoader(
        dataset,
        batch_size=dataloader_config.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=dataloader_config.get("num_workers", 4),
        pin_memory=dataloader_config.get("pin_memory", True),
        drop_last=shuffle,
        collate_fn=RVMDataset.collate_fn,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint_dir/last.pt")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config.get("seed") is not None:
        set_seed(config["seed"])

    train_loader = build_dataloader(config["dataset"]["train"], config["dataloader"], shuffle=True)
    eval_loader = build_dataloader(config["dataset"].get("eval"), config["dataloader"], shuffle=False)

    model = RVM(**config.get("model", {}))
    trainer_config = TrainerConfig(**config.get("trainer", {}))
    trainer = Trainer(model, train_loader, eval_loader, trainer_config)

    if args.resume:
        resumed = trainer.resume()
        print("resumed from checkpoint" if resumed else "no checkpoint found, starting fresh")

    trainer.train()


if __name__ == "__main__":
    main()
