import json
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
from decord import VideoReader, cpu
from multiprocessing import Pool
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _count_frames(video_path):
    """Frame count for one video, or -1 if it can't be opened/decoded.

    Module-level (not a method) so it can be pickled for the Pool used by
    SSv2._frame_counts.
    """
    try:
        return len(VideoReader(video_path, ctx=cpu(0), num_threads=1))
    except Exception:
        return -1


class SSv2(Dataset):
    """Something-Something v2 clips as (T, H, W, C) float tensors in [0, 1].

    Every item has exactly `num_frames` frames. Videos too short for the
    configured `frame_stride` fall back to the largest stride that still
    yields `num_frames` samples (n // num_frames), so only videos with fewer
    than `num_frames` frames are dropped.
    """

    def __init__(self, config_path, split="train"):
        super().__init__()

        self.config = yaml.safe_load(open(config_path, "r"))
        self.split = split

        self.num_frames = int(self.config["num_frames"])
        self.stride = int(self.config["frame_stride"])

        with open(self.config[f"{split}_json"], "r") as f:
            self.data = json.load(f)

        with open(self.config["labels_json"], "r") as f:
            self.labels = json.load(f)

        video_paths = []
        class_idx = []

        for item in self.data:
            video_paths.append(f"{self.config['videos_dir']}/{item['id']}.webm")
            label = item["template"].replace("[", "").replace("]", "")
            class_idx.append(int(self.labels[label]))

        counts = self._frame_counts(video_paths)

        # largest stride that still yields num_frames samples is n // num_frames;
        # clamp the configured stride down to it rather than dropping the video.
        keep, strides = [], []
        for i, n in enumerate(counts):
            if n < 0:
                continue
            s = min(self.stride, n // self.num_frames)
            if s >= 1:
                keep.append(i)
                strides.append(s)

        n_total = len(video_paths)
        n_kept = len(keep)
        n_bad = sum(1 for n in counts if n < 0)
        n_short = n_total - n_kept - n_bad
        n_reduced = sum(1 for s in strides if s < self.stride)

        self.video_paths = [video_paths[i] for i in keep]
        self.class_idx = [class_idx[i] for i in keep]
        self.video_strides = strides

        print(
            f"[{split}] kept {n_kept}/{n_total} videos "
            f"({100.0 * (n_total - n_kept) / max(1, n_total):.2f}% dropped: "
            f"{n_short} under {self.num_frames} frames, {n_bad} unreadable); "
            f"{n_reduced} ({100.0 * n_reduced / max(1, n_kept):.2f}%) "
            f"use a reduced stride"
        )

        if split == "train":
            self.transform = transforms.Compose([
                transforms.Resize(239),
                transforms.RandomCrop(224),
                transforms.RandomApply(
                    [transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1,
                    )],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.1),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(239),
                transforms.CenterCrop(224),
            ])

    def _frame_counts(self, video_paths):
        """Frame count per video, cached on disk -- the scan opens every file."""
        cache_path = os.path.join(
            self.config["videos_dir"], f".frame_counts_{self.split}.json"
        )

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached = json.load(f)
            # counts are stride-independent, so the cache survives config changes;
            # only a change in the video list invalidates it.
            if len(cached) == len(video_paths):
                return cached
            print(f"[{self.split}] frame-count cache is stale, rescanning")

        print(f"[{self.split}] scanning {len(video_paths)} videos for frame counts...")
        with Pool(processes=int(self.config.get("scan_workers", 16))) as pool:
            counts = pool.map(_count_frames, video_paths, chunksize=64)

        try:
            with open(cache_path, "w") as f:
                json.dump(counts, f)
        except OSError as e:
            print(f"[{self.split}] could not write frame-count cache: {e}")

        return counts

    def _load_video(self, video_path, stride):
        # num_threads=1: decord's threaded ffmpeg decoder throws EAGAIN on some
        # of the vp9/webm files. get_batch decodes in one pass instead of one
        # seek_accurate per frame.
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

        indices = list(range(0, self.num_frames * stride, stride))
        frames_np = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8

        frames = []
        for arr in frames_np:
            img = self.transform(Image.fromarray(arr))
            # HWC float tensor in [0, 1]
            img = torch.from_numpy(np.array(img)).float() / 255.0
            frames.append(img)

        return torch.stack(frames, dim=0).permute(0, 3, 1, 2)  # (T, H, W, C)

    def __getitem__(self, idx):
        frames = self._load_video(self.video_paths[idx], self.video_strides[idx])
        return frames, self.class_idx[idx]

    def __len__(self):
        return len(self.video_paths)


if __name__ == "__main__":
    datas = SSv2("dataset_config.yaml")
    loader = DataLoader(datas, batch_size=8)
    frames, labels = next(iter(loader))
    print(frames.shape, labels.shape)