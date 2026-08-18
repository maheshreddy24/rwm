import csv
import os

import cv2
import numpy as np
import torch
import yaml
from decord import VideoReader, cpu
from icecream import ic
from torch.utils.data import Dataset

FRAME_SIZE = (256, 256)

RRC_SCALE = (0.3, 1.0)
RRC_RATIO = (0.75, 1.25)


class RVMDataset(Dataset):
    """Samples (source frames, target frames, sampled_indices) triples for `RVM.forward`.

    Config keys:
        video_csv:          path to a csv with a `path` column (extra columns ignored)
        num_source_frames:  Ts, number of context frames fed to the recurrent core
        num_target_frames:  Tt, number of future frames to reconstruct
        max_delta:          inclusive upper bound on the source->target frame gap,
                             must match RVM(max_delta=...) since deltas index an embedding table
        frame_size:         (H, W) to resize decoded frames to, default (256, 256)

    Source frames are always Ts consecutive frames; target frames are sampled at
    random (unsorted) gaps of [4, max_delta] after the last source frame. A single
    RandomResizedCrop + optional horizontal flip is applied per-video, shared across
    every source and target frame, so the crop doesn't jitter between frames.
    """

    def __init__(self, config):
        self.video_csv = config["video_csv"]
        self.roll_out = config.get("roll_out", 16) # inspired from V-JEPA 2.1
        self.max_stride = config.get("max_stride", 3) # 12 fps, 3 stride --> 3 * 16 = 48, 
        self.min_duration = config.get("min_duration_frames", 48)
        self.frame_size = tuple(config.get("frame_size", FRAME_SIZE))

        self.rng = np.random.default_rng()

        self.data_paths = []

        with open(self.video_csv, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row["path"].strip()
                if path:
                    self.data_paths.append(path)

        print(f"Loaded {len(self.data_paths)} videos.")

    def _open_video(self, fname):
        if not os.path.exists(fname):
            return None
        try:
            return VideoReader(fname, num_threads=1, ctx=cpu(0))
        except Exception as e:
            print(f"[RVMDataset] failed to open {fname}: {e}")
            return None


    def _random_resized_crop_params(self, height: int, width: int):
        area = height * width
        log_ratio = np.log(RRC_RATIO)
        for _ in range(10):
            target_area = area * self.rng.uniform(RRC_SCALE[0], RRC_SCALE[1])
            aspect_ratio = np.exp(self.rng.uniform(log_ratio[0], log_ratio[1]))
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            if 0 < w <= width and 0 < h <= height:
                top = int(self.rng.integers(0, height - h + 1))
                left = int(self.rng.integers(0, width - w + 1))
                return top, left, h, w

        # Fallback: center crop clamped to the ratio bounds.
        in_ratio = width / height
        if in_ratio < min(RRC_RATIO):
            w = width
            h = int(round(w / min(RRC_RATIO)))
        elif in_ratio > max(RRC_RATIO):
            h = height
            w = int(round(h * max(RRC_RATIO)))
        else:
            w, h = width, height
        top = (height - h) // 2
        left = (width - w) // 2
        return top, left, h, w

    def _augment(self, buffer: np.ndarray) -> np.ndarray:
        """Apply one shared RandomResizedCrop + optional hflip across all frames.
        buffer: [T, H, W, 3] uint8 -> [T, *frame_size, 3] uint8."""
        T, H, W, _ = buffer.shape
        top, left, crop_h, crop_w = self._random_resized_crop_params(H, W)
        do_flip = self.rng.random() < 0.5

        out = np.empty((T, self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
        for t in range(T):
            frame = buffer[t, top : top + crop_h, left : left + crop_w]
            # cv2.resize takes (W, H); frame_size is (H, W)
            frame = cv2.resize(
                frame,
                (self.frame_size[1], self.frame_size[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            if do_flip:
                frame = frame[:, ::-1]
            out[t] = frame
        return out

    def _load_frames(self, fname):
        vr = self._open_video(fname)
        if vr is None:
            return None

        total_frames = len(vr)
        sampled_indices = np.linspace(0, total_frames, self.roll_out).astype(int)


        buffer = self._augment(buffer)
        frames = buffer.astype(np.float32) / 255.0  # [Ts+Tt, H, W, 3]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()  # [Ts+Tt, C, H, W]

        context = frames[sampled_indices]
        return context, torch.from_numpy(sampled_indices)

    def __getitem__(self, index):
        # Try up to 20 times to find a valid sample.
        for _ in range(20):
            video_path = self.data_paths[index]
            sample = self._load_frames(video_path)

            if sample is not None:
                context, sampled_indices = sample
                return {
                    "context": context,
                    "sampled_indices": sampled_indices,
                }

            index = int(self.rng.integers(len(self)))

        # Fallback: return a random dummy sample.
        print("returning random")

        context = torch.rand(
            self.num_target_frames,
            3,
            self.frame_size[0],
            self.frame_size[1],
            dtype=torch.float32,
        )

        sampled_indices = torch.randint(
            low=4,
            high=self.max_delta + 1,
            size=(self.num_target_frames,),
            dtype=torch.long,
        )

        return {
            "context": context,
            "sampled_indices": sampled_indices,
        }
    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def collate_fn(batch):
        return {
            # "source": torch.stack([item["source"] for item in batch], dim=0),        # [B, Ts, C, H, W]
            "conext": torch.stack([item["conext"] for item in batch], dim=0),        # [B, Tt, C, H, W]
            "sampled_indices": torch.stack([item["sampled_indices"] for item in batch], dim=0),  # [B, Tt]
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
    # ic(batch["source"].shape)
    ic(batch["conext"].shape)
    ic(batch["sampled_indices"].shape)