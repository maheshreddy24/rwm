"""How much of the (top-3 PCA) variance in ground-truth DINOv2 embeddings survives
in the model's teacher-forced predictions, and how does that change as the
target-frame stride (seconds between predicted frames) grows?

For each stride in STRIDES:
  - sample N_VIDEOS videos from the eval CSV, each with a random start point
  - build a clip of C context frames (gaps drawn from [CTX_MIN_STRIDE, CTX_MAX_STRIDE],
    same distribution the model was trained on) followed by S target frames spaced
    exactly `stride` seconds apart
  - run `teacher_forcing_rollout_repr` to get pred/gt patch-token embeddings (D=384)
  - for every target frame, fit PCA(3) separately on that frame's gt patch tokens and
    on its pred patch tokens, and take the fraction of variance the top 3 components
    explain
  - average that fraction over the S frames of a video, then over all videos

Produces one curve for gt and one for pred, variance-explained vs. stride.
Run from `src/` (same working-directory convention as inference.ipynb) so the
`from models.rvm_tf import RecurrentWorldModel` import resolves.
"""

import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from decord import VideoReader, cpu
from sklearn.decomposition import PCA
from tqdm import tqdm

from models.rvm_tf import RecurrentWorldModel

# ----------------------------------------------------------------------------- config

CSV_PATH = "/home/ego4d/data/v2/video_540ss_splits/test_ego4d.csv"
CONFIG_PATH = "/home/rvm/configs/train_ema.yaml"
CKPT_PATH = "/home/rvm/checkpoints/checkpoints_rwm/full_context/model_epoch1_step3900.pth"

N_VIDEOS = 10
STRIDES = [0.2, 0.4, 0.6, 1.0]  # seconds between target frames, swept

C = 5  # context frames
S = 5  # target (predicted) frames
CTX_MIN_STRIDE = 0.2  # seconds, context-frame gaps -- same range the model trained on
CTX_MAX_STRIDE = 0.4

FRAME_SIZE = (252, 252)  # must match encoder patch_size=14 -> 18x18=324 patch grid
SEED = 42
N_PCA_COMPONENTS = 3
POOL_MULTIPLIER = 3  # candidate pool = min(len(csv), POOL_MULTIPLIER * N_VIDEOS), to
                      # absorb videos that are too short / fail to decode at a given stride

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- model

def load_model():
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    model = RecurrentWorldModel(**config.get("model", {}))
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(DEVICE).eval()
    return model


# ----------------------------------------------------------------------------- data

def sample_indices(num_frames, fps, stride_sec, rng):
    """C context frames (gaps ~ [CTX_MIN_STRIDE, CTX_MAX_STRIDE]) followed by S
    target frames spaced exactly `stride_sec` apart, starting at a random offset.
    Returns (C+S,) frame indices, or None if the video is too short."""

    def to_frames(seconds):
        return max(1, round(seconds * fps))

    ctx_lo = to_frames(CTX_MIN_STRIDE)
    ctx_hi = max(ctx_lo, to_frames(CTX_MAX_STRIDE))
    tgt_gap = to_frames(stride_sec)

    ctx_gaps = rng.integers(ctx_lo, ctx_hi + 1, size=C - 1)
    tgt_gaps = np.full(S, tgt_gap, dtype=np.int64)
    gaps = np.concatenate([ctx_gaps, tgt_gaps])

    span = int(gaps.sum())
    if span >= num_frames:
        return None
    offset = int(rng.integers(0, num_frames - span))
    indices = offset + np.concatenate([[0], np.cumsum(gaps)])
    return indices.astype(np.int64)


def load_clip(path, indices):
    """(C+S,) frame indices -> float32 tensor (C+S, 3, H, W) in [0, 1], RGB."""
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    buffer = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) uint8 RGB
    out = np.empty((buffer.shape[0], FRAME_SIZE[0], FRAME_SIZE[1], 3), dtype=np.uint8)
    for t in range(buffer.shape[0]):
        out[t] = cv2.resize(buffer[t], (FRAME_SIZE[1], FRAME_SIZE[0]), interpolation=cv2.INTER_AREA)
    frames = out.astype(np.float32) / 255.0
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()  # (T, 3, H, W)
    return frames


# ----------------------------------------------------------------------------- pca

def pca_top3_ratio(points: np.ndarray) -> float:
    """points: (num_tokens, D). Fraction of total variance the top-3 PCs explain."""
    return float(PCA(n_components=N_PCA_COMPONENTS).fit(points).explained_variance_ratio_.sum())


# ----------------------------------------------------------------------------- main

def run_stride(model, df, stride_sec):
    gt_scores, pred_scores = [], []
    n_success = 0

    pbar = tqdm(df.iterrows(), total=min(len(df), N_VIDEOS * POOL_MULTIPLIER), desc=f"stride={stride_sec}s")
    for row_idx, row in pbar:
        if n_success >= N_VIDEOS:
            break

        path = str(row["path"]).strip()
        if not path or "drawing" in path or not os.path.exists(path):
            continue

        num_frames = int(float(row["num_frames"]))
        fps = float(row.get("fps") or 0.0)
        if fps <= 0:
            continue

        rng = np.random.default_rng(row_idx)  # fixed per video -> same offset/context
                                               # gaps reused across every stride value
        indices = sample_indices(num_frames, fps, stride_sec, rng)
        if indices is None:
            continue

        try:
            clip = load_clip(path, indices)  # (C+S, 3, H, W)
        except Exception as e:
            print(f"  [skip] decode failed {path}: {e}")
            continue

        context = clip[:C].unsqueeze(0).to(DEVICE)        # (1, C, 3, H, W)
        targets = clip[C:].unsqueeze(0).to(DEVICE)        # (1, S, 3, H, W)
        times = torch.from_numpy((indices / fps).astype(np.float32))
        context_times = times[:C].unsqueeze(0).to(DEVICE)
        target_times = times[C:].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model.teacher_forcing_rollout_repr(
                context=context,
                targets=targets,
                context_times=context_times,
                target_times=target_times,
            )

        pred = out["pred"][:, :, 1:, :]          # (1, S, P, D), CLS dropped
        gt = out["target_feat"][:, :, 1:, :]     # (1, S, P, D)

        frame_gt_scores, frame_pred_scores = [], []
        for s in range(S):
            frame_gt_scores.append(pca_top3_ratio(gt[0, s].cpu().numpy()))
            frame_pred_scores.append(pca_top3_ratio(pred[0, s].cpu().numpy()))

        gt_scores.append(float(np.mean(frame_gt_scores)))
        pred_scores.append(float(np.mean(frame_pred_scores)))
        n_success += 1
        pbar.set_postfix(done=n_success, gt=np.mean(gt_scores), pred=np.mean(pred_scores))

    pbar.close()
    return gt_scores, pred_scores


def main():
    df = pd.read_csv(CSV_PATH)
    pool_size = min(len(df), POOL_MULTIPLIER * N_VIDEOS)
    df = df.sample(n=pool_size, random_state=SEED).reset_index(drop=True)

    model = load_model()

    results = {}
    for stride in STRIDES:
        print(f"\n=== stride={stride}s ===")
        gt_scores, pred_scores = run_stride(model, df, stride)
        print(f"  videos used: {len(gt_scores)}/{N_VIDEOS}")
        print(f"  gt   top-3 variance explained: {np.mean(gt_scores):.4f} +- {np.std(gt_scores):.4f}")
        print(f"  pred top-3 variance explained: {np.mean(pred_scores):.4f} +- {np.std(pred_scores):.4f}")
        results[stride] = {
            "n_videos": len(gt_scores),
            "gt_mean": float(np.mean(gt_scores)),
            "gt_std": float(np.std(gt_scores)),
            "pred_mean": float(np.mean(pred_scores)),
            "pred_std": float(np.std(pred_scores)),
            "gt_scores": gt_scores,
            "pred_scores": pred_scores,
        }

    json_path = os.path.join(OUT_DIR, "pca_variance_vs_stride.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved raw results -> {json_path}")

    strides_sorted = sorted(results.keys())
    gt_means = [results[s]["gt_mean"] for s in strides_sorted]
    gt_stds = [results[s]["gt_std"] for s in strides_sorted]
    pred_means = [results[s]["pred_mean"] for s in strides_sorted]
    pred_stds = [results[s]["pred_std"] for s in strides_sorted]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(strides_sorted, gt_means, yerr=gt_stds, marker="o", capsize=3, label="ground truth")
    ax.errorbar(strides_sorted, pred_means, yerr=pred_stds, marker="o", capsize=3, label="prediction")
    ax.set_xlabel("target stride (seconds between predicted frames)")
    ax.set_ylabel("top-3 PCA explained variance ratio")
    ax.set_title(f"top-3 PCA variance: gt vs. pred (N={N_VIDEOS} videos, teacher forcing)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    png_path = os.path.join(OUT_DIR, "pca_variance_vs_stride.png")
    fig.savefig(png_path, dpi=150)
    print(f"saved plot -> {png_path}")


if __name__ == "__main__":
    main()
