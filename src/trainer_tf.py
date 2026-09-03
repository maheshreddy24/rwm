import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.rvm_dataset_tf import RVMDataset
from src.models.rvm_tf import RecurrentWorldModel
from src.models.utils.variants import resolve_variant
from src.optimisation.optimizer_ema import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Full determinism would also require torch.use_deterministic_algorithms(True);
    # without it cudnn.deterministic only buys slower convs, not reproducibility.
    torch.backends.cudnn.benchmark = True


def _worker_init_fn(worker_id):
    # Without this, each worker process runs cv2 with its own full-core
    # thread pool, oversubscribing the CPU and starving the GPU.
    cv2.setNumThreads(1)


def build_dataloader(dataset_config, dataloader_config, shuffle: bool, persistent_workers: bool, num_workers=None):
    if dataset_config is None:
        return None
    dataset = RVMDataset(dataset_config)
    if num_workers is None:
        num_workers = dataloader_config.get("num_workers", 4)
    return DataLoader(
        dataset,
        batch_size=dataloader_config.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=dataloader_config.get("pin_memory", True),
        drop_last=shuffle,
        collate_fn=RVMDataset.collate_fn,
        persistent_workers=persistent_workers and num_workers > 0,
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

    cv2.setNumThreads(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if config.get("seed") is not None:
        set_seed(config["seed"])

    dataloader_config = config["dataloader"]
    train_loader = build_dataloader(
        config["dataset"]["train"], dataloader_config, shuffle=True, persistent_workers=True
    )
    eval_loader = build_dataloader(
        config["dataset"].get("eval"),
        dataloader_config,
        shuffle=False,
        persistent_workers=False,
        num_workers=dataloader_config.get("eval_num_workers", min(4, dataloader_config.get("num_workers", 4))),
    )

    model = RecurrentWorldModel(**resolve_variant(config.get("model", {})))

    trainer = Trainer(model, train_loader, eval_loader, config)

    if args.resume is not None:
        ckpt_path = None if args.resume == "__latest__" else args.resume
        resumed = trainer.resume(ckpt_path)
        print(f"resumed from checkpoint: {trainer.checkpoint_dir}" if resumed else "no checkpoint found, starting fresh")

    trainer.train()


if __name__ == "__main__":
    main()