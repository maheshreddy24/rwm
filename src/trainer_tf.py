# import argparse
# import random
# import sys
# from pathlib import Path

# import numpy as np
# import torch
# import cv2
# import yaml
# from torch.utils.data import DataLoader

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# from src.datasets.rvm_dataset_tf import RVMDataset
# from src.models.rvm_tf import RecurrentWorldModel
# from src.optimisation.optimizer_ema import Trainer


# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     # Full determinism would also require torch.use_deterministic_algorithms(True);
#     # without it cudnn.deterministic only buys slower convs, not reproducibility.
#     torch.backends.cudnn.benchmark = True


# def _worker_init_fn(worker_id):
#     # Without this, each worker process runs cv2 with its own full-core
#     # thread pool, oversubscribing the CPU and starving the GPU.
#     cv2.setNumThreads(1)


# def build_dataloader(dataset_config, dataloader_config, shuffle: bool, persistent_workers: bool, num_workers=None):
#     if dataset_config is None:
#         return None
#     dataset = RVMDataset(dataset_config)
#     if num_workers is None:
#         num_workers = dataloader_config.get("num_workers", 4)
#     return DataLoader(
#         dataset,
#         batch_size=dataloader_config.get("batch_size", 8),
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=dataloader_config.get("pin_memory", True),
#         drop_last=shuffle,
#         collate_fn=RVMDataset.collate_fn,
#         persistent_workers=persistent_workers and num_workers > 0,
#         prefetch_factor=dataloader_config.get("prefetch_factor", 4) if num_workers > 0 else None,
#         worker_init_fn=_worker_init_fn if num_workers > 0 else None,
#     )


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", default="configs/train_ema.yaml")
#     parser.add_argument(
#         "--resume",
#         nargs="?",
#         const="__latest__",
#         default=None,
#         metavar="CKPT_PATH",
#         help="resume training; pass a checkpoint path to resume from it, "
#         "or omit the value to resume from the latest checkpoint in checkpoint_dir",
#     )
#     args = parser.parse_args()

#     cv2.setNumThreads(1)

#     with open(args.config, "r") as f:
#         config = yaml.safe_load(f)

#     if config.get("seed") is not None:
#         set_seed(config["seed"])

#     dataloader_config = config["dataloader"]
#     train_loader = build_dataloader(
#         config["dataset"]["train"], dataloader_config, shuffle=True, persistent_workers=True
#     )
#     eval_loader = build_dataloader(
#         config["dataset"].get("eval"),
#         dataloader_config,
#         shuffle=False,
#         persistent_workers=False,
#         num_workers=dataloader_config.get("eval_num_workers", min(4, dataloader_config.get("num_workers", 4))),
#     )

#     model = RecurrentWorldModel(**config.get("model", {}))

#     trainer = Trainer(model, train_loader, eval_loader, config)

#     if args.resume is not None:
#         ckpt_path = None if args.resume == "__latest__" else args.resume
#         resumed = trainer.resume(ckpt_path)
#         print(f"resumed from checkpoint: {trainer.checkpoint_dir}" if resumed else "no checkpoint found, starting fresh")

#     trainer.train()


# if __name__ == "__main__":
#     main()

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
from src.optimisation.optimizer_ema import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Full determinism would also require:
    # torch.use_deterministic_algorithms(True)
    #
    # We intentionally keep benchmark=True because the goal here
    # is training/evaluation throughput rather than full determinism.
    torch.backends.cudnn.benchmark = True


def _worker_init_fn(worker_id):
    """
    Prevent each DataLoader worker from creating its own OpenCV
    thread pool and oversubscribing the CPU.
    """
    cv2.setNumThreads(1)


def build_dataloader(
    dataset_config,
    dataloader_config,
    shuffle: bool,
    persistent_workers: bool,
    num_workers=None,
    prefetch_factor=None,
):
    if dataset_config is None:
        return None

    dataset = RVMDataset(dataset_config)

    if num_workers is None:
        num_workers = dataloader_config.get("num_workers", 4)

    if prefetch_factor is None:
        prefetch_factor = dataloader_config.get("prefetch_factor", 4)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=dataloader_config.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=dataloader_config.get("pin_memory", True),
        drop_last=shuffle,
        collate_fn=RVMDataset.collate_fn,
        persistent_workers=persistent_workers and num_workers > 0,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )

    # prefetch_factor is only valid when num_workers > 0.
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/train_ema.yaml",
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        const="__latest__",
        default=None,
        metavar="CKPT_PATH",
        help=(
            "resume training; pass a checkpoint path to resume from it, "
            "or omit the value to resume from the latest checkpoint "
            "in checkpoint_dir"
        ),
    )

    args = parser.parse_args()

    # Keep OpenCV single-threaded in the main process as well.
    cv2.setNumThreads(1)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------------
    if config.get("seed") is not None:
        set_seed(config["seed"])

    dataloader_config = config["dataloader"]

    # ------------------------------------------------------------------
    # TRAIN DATALOADER
    #
    # IMPORTANT:
    # Training configuration is intentionally left unchanged.
    # ------------------------------------------------------------------
    train_loader = build_dataloader(
        config["dataset"]["train"],
        dataloader_config,
        shuffle=True,
        persistent_workers=True,
    )

    # ------------------------------------------------------------------
    # EVAL DATALOADER
    #
    # These settings are intentionally HARD-CODED here.
    #
    # We do NOT use the eval_num_workers from YAML.
    # We do NOT change the training DataLoader.
    # ------------------------------------------------------------------

    EVAL_NUM_WORKERS = 15
    EVAL_PREFETCH_FACTOR = 2
    EVAL_PIN_MEMORY = True
    EVAL_PERSISTENT_WORKERS = True

    eval_dataset_config = config["dataset"].get("eval")

    eval_loader = None

    if eval_dataset_config is not None:
        eval_dataset = RVMDataset(eval_dataset_config)

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=dataloader_config.get("batch_size", 8),
            shuffle=False,
            num_workers=EVAL_NUM_WORKERS,
            pin_memory=EVAL_PIN_MEMORY,
            drop_last=False,
            collate_fn=RVMDataset.collate_fn,
            persistent_workers=EVAL_PERSISTENT_WORKERS,
            prefetch_factor=EVAL_PREFETCH_FACTOR,
            worker_init_fn=_worker_init_fn,
        )

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    model = RecurrentWorldModel(
        **config.get("model", {})
    )

    # ------------------------------------------------------------------
    # TRAINER
    # ------------------------------------------------------------------
    trainer = Trainer(
        model,
        train_loader,
        eval_loader,
        config,
    )

    # ------------------------------------------------------------------
    # RESUME
    # ------------------------------------------------------------------
    if args.resume is not None:
        ckpt_path = None if args.resume == "__latest__" else args.resume

        resumed = trainer.resume(ckpt_path)

        print(
            f"resumed from checkpoint: {trainer.checkpoint_dir}"
            if resumed
            else "no checkpoint found, starting fresh"
        )

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------
    trainer.train()


if __name__ == "__main__":
    main()
