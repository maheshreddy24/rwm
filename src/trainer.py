"""Entry point: wires RVMDataset -> RVM -> Trainer and runs training.

Usage:
python src/trainer.py --config configs/train_ema.yaml --resume /home/rvm/checkpoints_ema_dino_backbone/exp_1785924245.177395/model_epoch2_step10000.pth
"""

import argparse
import sys
from pathlib import Path

import cv2
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    
from src.datasets.rvm_dataset import RVMDataset
from src.models.rvm import RVM
from src.optimisation.optimizer_ema import Trainer
import torch
import random
import numpy as np



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id):
    # Each of the 16 worker processes otherwise runs cv2 with its own
    # full-core thread pool, oversubscribing the CPU 16x and starving the GPU.
    cv2.setNumThreads(1)


def _worker_init_fn(worker_id):
    # Each of the 16 worker processes otherwise runs cv2 with its own
    # full-core thread pool, oversubscribing the CPU 16x and starving the GPU.
    cv2.setNumThreads(1)


def build_dataloader(dataset_config, dataloader_config, shuffle: bool):
    if dataset_config is None:
        return None
    dataset = RVMDataset(dataset_config)
    num_workers = dataloader_config.get("num_workers", 4)
    return DataLoader(
        dataset,
        batch_size=dataloader_config.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=dataloader_config.get("pin_memory", True),
        drop_last=shuffle,
        collate_fn=RVMDataset.collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=dataloader_config.get("prefetch_factor", 4) if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_ema.yaml")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="__latest__",
        default=None,
        metavar="CKPT_PATH",
        help="resume training; pass a checkpoint path to resume from it, "
        "or omit the value to resume from the latest checkpoint in checkpoint_dir",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config.get("seed") is not None:
        print("config is set")
        set_seed(config["seed"])

    train_loader = build_dataloader(config["dataset"]["train"], config["dataloader"], shuffle=True)
    eval_loader = build_dataloader(config["dataset"].get("eval"), config["dataloader"], shuffle=False)

    model = RVM(**config.get("model", {}))
    trainer = Trainer(model, train_loader, eval_loader, config["trainer"])

    if args.resume is not None:
        ckpt_path = None if args.resume == "__latest__" else args.resume
        resumed = trainer.resume(ckpt_path)
        print(f"resumed from checkpoint: {trainer.checkpoint_dir}" if resumed else "no checkpoint found, starting fresh")

    trainer.train()


if __name__ == "__main__":
    main()
