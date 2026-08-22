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
    label: int = -1

    @property
    def length(self) -> int:
        return self.end - self.start


class RVMDataset(Dataset):
    """Samples a rolled-out clip of frames for `RecurrentWorldModel.forward`.

    Config:
        manifest:              csv from build_manifest.py
                               (path,label,num_frames,fps,duration_sec,source)
        sources:               per-source windowing rules, keyed by the manifest's
                               `source` column:
                                   mode:            "single"  -> one window per video
                                                    "segment" -> one window per chunk
                                   segment_seconds: chunk length for "segment" (default 180)
                                   max_windows:     optional cap on windows per video
                                   weight:          sampling weight for make_weighted_sampler
                               unlisted sources fall back to `default_source`
        default_source:        rules for any source not in `sources` (default: single, w=1)
        roll_out:              frames per clip, V-JEPA 2.1 style (default 16)
        max_stride:            max frame stride between sampled frames (default 3)
        min_duration_frames:   windows shorter than this are dropped (default 48)
        frame_size:            (H, W) to resize decoded frames to, default (256, 256)
        deterministic:         eval mode - center crop, no flip, fixed stride, window
                               center start. Set True for the test manifest.

    Video lengths come from the manifest, so __init__ never opens a video. A clip is
    `roll_out` frames spaced by a random stride in [1, max_stride], starting at a
    random offset *inside its window*. One RandomResizedCrop + optional hflip is drawn
    per clip and shared across frames, so the crop doesn't jitter.
    """

    DEFAULT_SOURCE = {"mode": "single", "weight": 1.0}

    def __init__(self, config, manifest=None, deterministic=None):
        self.manifest_path = manifest or config["manifest"]
        self.roll_out = config.get("roll_out", 16)  # inspired from V-JEPA 2.1
        self.max_stride = config.get("max_stride", 3)  # 12 fps, 3 stride --> 3 * 16 = 48
        self.min_duration = config.get("min_duration_frames", 48)
        self.frame_size = tuple(config.get("frame_size", FRAME_SIZE))
        self.deterministic = (
            config.get("deterministic", False) if deterministic is None else deterministic
        )

        self.sources = config.get("sources", {}) or {}
        self.default_source = config.get("default_source", self.DEFAULT_SOURCE)

        # A window must be long enough to hold one stride-1 clip.
        self.min_window = max(self.min_duration, self.roll_out)

        # rng is created lazily so each dataloader worker gets a distinct stream.
        self._rng = None

        self.windows: list[ClipWindow] = []
        self._build_index()

    # ------------------------------------------------------------------ index

    @property
    def rng(self):
        if self._rng is None:
            # torch.initial_seed() differs per worker and per epoch.
            self._rng = np.random.default_rng(torch.initial_seed() % (2**32))
        return self._rng

    def _rules_for(self, source):
        return {**self.DEFAULT_SOURCE, **self.default_source, **self.sources.get(source, {})}

    def _build_index(self):
        stats = {}
        with open(self.manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                path = row["path"].strip()
                if not path:
                    continue
                source = (row.get("source") or "default").strip()
                num_frames = int(float(row["num_frames"]))
                fps = float(row.get("fps") or 0.0)
                try:
                    label = int(float(row.get("label", -1)))
                except (TypeError, ValueError):
                    label = -1

                entry = stats.setdefault(source, {"videos": 0, "skipped": 0, "windows": 0})
                if num_frames < self.min_window:
                    entry["skipped"] += 1
                    continue
                entry["videos"] += 1

                rules = self._rules_for(source)
                before = len(self.windows)

                if rules["mode"] == "segment":
                    seconds = float(rules.get("segment_seconds", 180.0))
                    seg = int(round(seconds * fps)) if fps > 0 else num_frames
                    seg = max(seg, self.min_window)
                    n_seg = max(1, num_frames // seg)
                    if rules.get("max_windows"):
                        n_seg = min(n_seg, int(rules["max_windows"]))
                    # Spread evenly so the tail is absorbed rather than orphaned.
                    bounds = np.linspace(0, num_frames, n_seg + 1).round().astype(int)
                    for start, end in zip(bounds[:-1], bounds[1:]):
                        if end - start >= self.min_window:
                            self.windows.append(
                                ClipWindow(path, int(start), int(end), source, label)
                            )
                else:
                    self.windows.append(ClipWindow(path, 0, num_frames, source, label))

                entry["windows"] += len(self.windows) - before

        name = os.path.basename(self.manifest_path)
        mode_tag = "eval" if self.deterministic else "train"
        print(f"[RVMDataset] {name} ({mode_tag})")
        for source in sorted(stats):
            rules = self._rules_for(source)
            entry = stats[source]
            extra = (
                f" @{rules.get('segment_seconds', 180)}s"
                if rules["mode"] == "segment"
                else ""
            )
            print(
                f"  {source:12s} {entry['videos']:7d} videos -> "
                f"{entry['windows']:8d} windows  ({rules['mode']}{extra}, "
                f"{entry['skipped']} too short)"
            )
        print(f"  {'TOTAL':12s} {len(self.windows):8d} windows")

    # ------------------------------------------------------------------ video

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

        if self.deterministic:
            stride = self.max_stride
        else:
            stride = int(self.rng.integers(1, self.max_stride + 1))

        span = (self.roll_out - 1) * stride
        if span >= available:
            stride = max(1, (available - 1) // (self.roll_out - 1))
            span = (self.roll_out - 1) * stride

        if self.deterministic:
            offset = (available - span) // 2  # window center
        else:
            offset = int(self.rng.integers(0, available - span))

        indices = win_start + offset + np.arange(self.roll_out) * stride
        return np.clip(indices, 0, total_frames - 1)

    # ------------------------------------------------------------- augmentation

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
        return self._center_crop_params(height, width)

    def _center_crop_params(self, height: int, width: int):
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
        if self.deterministic:
            top, left, crop_h, crop_w = self._center_crop_params(H, W)
            do_flip = False
        else:
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

    # ------------------------------------------------------------------ dunder

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
                    "label": window.label,
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
            "label": -1,
        }

    def __len__(self):
        return len(self.windows)

    # ------------------------------------------------------------------ helpers

    def make_weighted_sampler(self, num_samples=None):
        """Balance sources so Ego4D's many windows don't drown out ssv2/kinetics.

        Per-window weight is `source_weight / n_windows_in_source`, so each source's
        total probability mass equals its configured weight.
        """
        from torch.utils.data import WeightedRandomSampler

        counts = {}
        for window in self.windows:
            counts[window.source] = counts.get(window.source, 0) + 1
        weights = torch.tensor(
            [
                float(self._rules_for(w.source)["weight"]) / counts[w.source]
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
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),  # [B], -1 = unlabelled
        }


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "dataset.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_set = RVMDataset(config, manifest=config["train_manifest"], deterministic=False)
    test_set = RVMDataset(config, manifest=config["test_manifest"], deterministic=True)

    loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=2,
        sampler=train_set.make_weighted_sampler(),
        num_workers=2,
        collate_fn=RVMDataset.collate_fn,
    )

    batch = next(iter(loader))
    ic(batch["context"].shape)
    ic(batch["sampled_indices"].shape)
    ic(batch["source"])