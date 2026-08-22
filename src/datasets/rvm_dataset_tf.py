import csv
import os
from dataclasses import dataclass

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


@dataclass(slots=True)
class ClipWindow:
    """One sampleable region of one video. `start`/`end` are frame indices, end exclusive."""

    path: str
    start: int
    end: int
    source: str

    @property
    def length(self) -> int:
        return self.end - self.start


class RVMDataset(Dataset):
    """Samples a rolled-out clip of frames for `RecurrentWorldModel.forward`.

    Config:
        datasets:                     list of source specs (see below)
        roll_out:                     frames per clip, V-JEPA 2.1 style (default 16)
        max_stride:                   max frasme stride between sampled frames (default 3)
        min_duration_frames:          windows shorter than this are dropped (default 48)
        frame_size:                   (H, W) to resize decoded frames to, default (256, 256)

    Each entry of `datasets` is:
        name:              tag carried through to the batch
        manifest:          csv from build_manifest.py (path,num_frames,fps,duration,source)
        mode:              "single"  -> one window spanning the whole video (ssv2 / kinetics)
                           "segment" -> one window per `segment_seconds` chunk (ego4d)
        segment_seconds:   chunk length for mode="segment" (default 180 = 3 min)
        max_windows:       optional cap on windows per video, evenly spread
        weight:            optional sampling weight, used by make_weighted_sampler

    Video lengths are read from the manifest, so __init__ never opens a video. A clip
    is `roll_out` frames spaced by a random stride in [1, max_stride], starting at a
    random offset *inside its window*. One RandomResizedCrop + optional hflip is drawn
    per clip and shared across frames, so the crop doesn't jitter.
    """

    def __init__(self, config):
        self.roll_out = config.get("roll_out", 16)  # inspired from V-JEPA 2.1
        self.max_stride = config.get("max_stride", 3)  # 12 fps, 3 stride --> 3 * 16 = 48
        self.min_duration = config.get("min_duration_frames", 48)
        self.frame_size = tuple(config.get("frame_size", FRAME_SIZE))

        # A window must be long enough to hold one stride-1 clip.
        self.min_window = max(self.min_duration, self.roll_out)

        # rng is created lazily so each dataloader worker gets a distinct stream.
        self._rng = None

        specs = config.get("datasets")
        if specs is None:  # backwards compat with the old single-csv config
            specs = [{"name": "default", "manifest": config["video_csv"], "mode": "single"}]

        self.windows: list[ClipWindow] = []
        self.source_weights: dict[str, float] = {}
        for spec in specs:
            name = spec.get("name", os.path.basename(spec["manifest"]))
            self.source_weights[name] = float(spec.get("weight", 1.0))
            before = len(self.windows)
            self.windows.extend(self._windows_for_spec(spec, name))
            n_new = len(self.windows) - before
            print(f"[RVMDataset] {name}: {n_new} windows (mode={spec.get('mode', 'single')})")

        print(f"[RVMDataset] {len(self.windows)} windows total.")


    @property
    def rng(self):
        if self._rng is None:
            # torch.initial_seed() differs per worker and per epoch.
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def _windows_for_spec(self, spec, name):
        mode = spec.get("mode", "single")
        segment_seconds = float(spec.get("segment_seconds", 180.0))
        max_windows = spec.get("max_windows")

        windows = []
        with open(spec["manifest"], newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = row["path"].strip()
                if not path:
                    continue
                num_frames = int(row["num_frames"])
                fps = float(row.get("fps") or 0.0)
                if num_frames < self.min_window:
                    continue

                if mode == "single":
                    windows.append(ClipWindow(path, 0, num_frames, name))
                    continue

                # mode == "segment": chunk the video into equal windows.
                seg = int(round(segment_seconds * fps)) if fps > 0 else num_frames
                seg = max(seg, self.min_window)
                n_seg = max(1, num_frames // seg)
                if max_windows:
                    n_seg = min(n_seg, int(max_windows))

                # Spread windows evenly so the tail is absorbed rather than dropped.
                bounds = np.linspace(0, num_frames, n_seg + 1).round().astype(int)
                for start, end in zip(bounds[:-1], bounds[1:]):
                    if end - start >= self.min_window:
                        windows.append(ClipWindow(path, int(start), int(end), name))
        return windows


    def _open_video(self, fname):
        if not os.path.exists(fname):
            return None
        try:
            return VideoReader(fname, num_threads=1, ctx=cpu(0))
        except Exception as e:
            print(f"[RVMDataset] failed to open {fname}: {e}")
            return None

    def _sample_indices(self, win_start, win_end, total_frames):
        """Pick `roll_out` strided indices inside [win_start, win_end)."""
        win_end = min(win_end, total_frames)
        available = win_end - win_start
        if available < self.roll_out:
            return None

        stride = int(self.rng.integers(1, self.max_stride + 1))
        span = (self.roll_out - 1) * stride
        if span >= available:
            stride = max(1, (available - 1) // (self.roll_out - 1))
            span = (self.roll_out - 1) * stride

        start = win_start + int(self.rng.integers(0, available - span))
        indices = start + np.arange(self.roll_out) * stride
        return np.clip(indices, 0, total_frames - 1)


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

    def _load_frames(self, window: ClipWindow):
        vr = self._open_video(window.path)
        if vr is None:
            return None

        total_frames = len(vr)
        if total_frames < self.min_window:
            return None

        sampled_indices = self._sample_indices(window.start, window.end, total_frames)
        if sampled_indices is None:
            return None

        try:
            buffer = vr.get_batch(sampled_indices).asnumpy()  # [roll_out, H, W, 3]
        except Exception as e:
            print(f"[RVMDataset] decode failed {window.path}[{window.start}:{window.end}]: {e}")
            return None

        buffer = self._augment(buffer)
        frames = buffer.astype(np.float32) / 255.0  # [roll_out, H, W, 3]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()  # [roll_out, C, H, W]

        return frames, torch.from_numpy(sampled_indices.astype(np.int64))


    def __getitem__(self, index):
        # Try up to 20 times to find a valid sample.
        for _ in range(20):
            window = self.windows[index]
            sample = self._load_frames(window)

            if sample is not None:
                context, sampled_indices = sample
                return {
                    "context": context,
                    "sampled_indices": sampled_indices,
                    "source": window.source,
                }

            index = int(self.rng.integers(len(self)))

        # Fallback: return a random dummy sample.
        print("returning random")

        context = torch.rand(
            self.roll_out,
            3,
            self.frame_size[0],
            self.frame_size[1],
            dtype=torch.float32,
        )
        sampled_indices = torch.arange(self.roll_out, dtype=torch.long)

        return {
            "context": context,
            "sampled_indices": sampled_indices,
            "source": "dummy",
        }

    def __len__(self):
        return len(self.windows)

    def make_weighted_sampler(self, num_samples=None):
        """Balance sources so Ego4D's many windows don't drown out ssv2/kinetics.

        Per-source weight is `spec.weight / n_windows_in_source`, so each source's
        total mass equals its configured weight.
        """
        from torch.utils.data import WeightedRandomSampler

        counts = {}
        for window in self.windows:
            counts[window.source] = counts.get(window.source, 0) + 1
        weights = torch.tensor(
            [
                self.source_weights.get(w.source, 1.0) / counts[w.source]
                for w in self.windows
            ],
            dtype=torch.double,
        )
        return WeightedRandomSampler(
            weights, num_samples or len(self.windows), replacement=True
        )

    @staticmethod
    def collate_fn(batch):
        return {
            "context": torch.stack([item["context"] for item in batch], dim=0),         # [B, roll_out, C, H, W]
            "sampled_indices": torch.stack([item["sampled_indices"] for item in batch], dim=0),  # [B, roll_out]
            "source": [item["source"] for item in batch],
        }


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "dataset.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = RVMDataset(config)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        sampler=dataset.make_weighted_sampler(),
        num_workers=2,
        collate_fn=RVMDataset.collate_fn,
    )

    batch = next(iter(loader))
    ic(batch["context"].shape)
    ic(batch["sampled_indices"].shape)
    ic(batch["source"])