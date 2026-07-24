import csv
import os
import random

import cv2
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from icecream import ic
from torch.utils.data import Dataset

FRAME_SIZE = (256, 256)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RVMDataset(Dataset):
    """Samples (source frames, target frames, target_deltas) triples for `RVM.forward`.

    Config keys:
        video_csv:          path to a csv with columns [path, num_frames, fps, duration_sec]
        num_source_frames:  Ts, number of context frames fed to the recurrent core
        num_target_frames:  Tt, number of future frames to reconstruct
        max_delta:          upper bound (exclusive) on the source->target frame gap,
                             must match RVM(max_delta=...) since deltas index an embedding table
        frame_size:         (H, W) to resize decoded frames to, default (256, 256)
        min_duration_sec:   drop videos shorter than this, default 4
    """

    def __init__(self, config):
        self.config = config
        self.video_csv = config["video_csv"]
        self.num_source_frames = config["num_source_frames"]
        self.num_target_frames = config["num_target_frames"]
        self.max_delta = config.get("max_delta", 64)
        self.frame_size = tuple(config.get("frame_size", FRAME_SIZE))
        min_duration_sec = config.get("min_duration_sec", 4)

        data_paths = []
        with open(self.video_csv, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i > 0 and float(row[-1]) > min_duration_sec:
                    data_paths.append(row)

        self.data_paths = data_paths
        print(f"total samples: {len(self.data_paths)}")

    def _open_video(self, fname):
        if not os.path.exists(fname):
            return None
        try:
            return VideoReader(fname, num_threads=1, ctx=cpu(0))
        except Exception:
            return None

    def _sample_indices(self, total_frames: int):
        """Ts source indices (increasing, random stride) + Tt target indices sampled
        at random gaps after the last source frame. Returns (source_idx, target_idx, deltas)."""
        Ts, Tt = self.num_source_frames, self.num_target_frames

        last_src_idx = random.randint(Ts - 1, max(Ts - 1, total_frames - 2))
        last_src_idx = min(last_src_idx, max(total_frames - 2, 0))

        max_stride = max(1, last_src_idx // max(Ts - 1, 1))
        stride = random.randint(1, max_stride)
        source_idx = last_src_idx - stride * np.arange(Ts - 1, -1, -1)
        source_idx = np.clip(source_idx, 0, total_frames - 1).astype(np.int64)

        max_gap = max(min(self.max_delta - 1, total_frames - 1 - last_src_idx), 1)
        deltas = np.sort(np.random.randint(1, max_gap + 1, size=Tt)).astype(np.int64)
        target_idx = np.clip(last_src_idx + deltas, 0, total_frames - 1).astype(np.int64)

        return source_idx, target_idx, deltas

    def _load_frames(self, fname):
        vr = self._open_video(fname)
        if vr is None or len(vr) < 2:
            return None

        total_frames = len(vr)
        source_idx, target_idx, deltas = self._sample_indices(total_frames)

        vr.seek(0)
        all_idx = np.concatenate([source_idx, target_idx])
        buffer = vr.get_batch(all_idx).asnumpy()  # [Ts+Tt, H, W, 3] uint8
        buffer = np.stack(
            [cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR) for frame in buffer]
        )

        frames = torch.from_numpy(buffer).float() / 255.0  # [Ts+Tt, H, W, 3]
        frames = frames.permute(0, 3, 1, 2).contiguous()  # [Ts+Tt, 3, H, W]

        Ts = self.num_source_frames
        source = frames[:Ts]
        target = frames[Ts:]
        return source, target, torch.from_numpy(deltas)

    def __getitem__(self, index):
        video_path = self.data_paths[index][0]

        sample = self._load_frames(video_path)
        if sample is None:
            # Invalid sample, retry with a random one.
            return self.__getitem__(random.randrange(len(self)))

        source, target, target_deltas = sample
        return {
            "source": source,
            "target": target,
            "target_deltas": target_deltas,
        }

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def collate_fn(batch):
        return {
            "source": torch.stack([item["source"] for item in batch], dim=0),        # [B, Ts, 3, H, W]
            "target": torch.stack([item["target"] for item in batch], dim=0),        # [B, Tt, 3, H, W]
            "target_deltas": torch.stack([item["target_deltas"] for item in batch], dim=0),  # [B, Tt]
        }


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "dataset.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = RVMDataset(config)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=RVMDataset.collate_fn
    )

    batch = next(iter(loader))
    ic(batch["source"].shape)
    ic(batch["target"].shape)
    ic(batch["target_deltas"].shape)
