"""
IntPhys2 prediction-eval scorer, v2.

Fixes vs v1:
  * pairs within SceneIndex only. `occluder` is the variable that FLIPS
    possibility in IntPhys2, so grouping on it matches same-outcome videos
    together and silently drops them.
  * uses the real `Difficulty` / `Camera` / `condition` columns instead of
    guessing difficulty from `env`.
  * prints label/pairing diagnostics so a broken merge is visible.

Usage:
    python score_intphys2_v2.py --losses losses_10fs_..._rank0.pth \
                                --metadata .../Main/metadata.csv
"""
import argparse
import itertools
import numpy as np
import pandas as pd
import torch


def reduce_surprise(losses):
    """[N, C, W] zero-padded -> per-video scores, padding masked out."""
    losses = losses.float()
    valid = losses > 0

    neg_inf = torch.where(valid, losses, torch.full_like(losses, -float("inf")))
    max_s = neg_inf.max(dim=2).values
    avg_s = (losses * valid).sum(dim=2) / valid.sum(dim=2).clamp(min=1)

    out = {"max_per_ctxt": max_s, "avg_per_ctxt": avg_s}

    pos_inf = torch.where(valid, losses, torch.full_like(losses, float("inf")))
    filt = pos_inf.min(dim=1).values
    fv = torch.isfinite(filt) & (filt > 0)
    out["max_filtered"] = torch.where(
        fv, filt, torch.full_like(filt, -float("inf"))).max(dim=1).values
    out["avg_filtered"] = (torch.where(fv, filt, torch.zeros_like(filt)).sum(1)
                           / fv.sum(1).clamp(min=1))
    return out


def relative_accuracy(df, score_col, pair_cols=("SceneIndex",)):
    """All (possible, impossible) combinations within each scene."""
    correct = total = 0
    for _, g in df.groupby(list(pair_cols)):
        pos = g.loc[g.is_possible == 1, score_col].values
        imp = g.loc[g.is_possible == 0, score_col].values
        for p, i in itertools.product(pos, imp):
            total += 1
            correct += int(p < i)
    return 100.0 * correct / max(total, 1), total


def absolute_accuracy(df, score_col):
    pos = np.sort(df.loc[df.is_possible == 1, score_col].values)
    imp = df.loc[df.is_possible == 0, score_col].values
    if len(pos) == 0 or len(imp) == 0:
        return float("nan"), float("nan")
    idx = min(int(np.ceil(0.90 * len(pos))), len(pos) - 1)
    t = pos[idx]
    acc = 100.0 * ((pos < t).sum() + (imp > t).sum()) / (len(pos) + len(imp))
    grid = np.linspace(pos.min(), imp.max(), 100)
    best = max(((pos < g).sum() + (imp > g).sum()) / (len(pos) + len(imp))
               for g in grid)
    return acc, 100.0 * best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--score", default="max_filtered")
    args = ap.parse_args()

    d = torch.load(args.losses, map_location="cpu", weights_only=False)
    losses, names, ctxts = d["losses"], d["names"], d["context_lengths"]
    print(f"losses {tuple(losses.shape)}  contexts {ctxts}  frame_step {d['frame_step']}")

    # padding sanity check
    valid = (losses > 0).sum(2)
    print(f"valid windows per (video, ctxt): min={valid.min().item()} "
          f"max={valid.max().item()}  (tensor W = {losses.shape[2]})")

    red = reduce_surprise(losses)
    df = pd.DataFrame({"key": [str(n).replace(".mp4", "") for n in names]})
    for i, c in enumerate(ctxts):
        df[f"max_ctxt{c}"] = red["max_per_ctxt"][:, i].numpy()
        df[f"avg_ctxt{c}"] = red["avg_per_ctxt"][:, i].numpy()
    df["max_filtered"] = red["max_filtered"].numpy()
    df["avg_filtered"] = red["avg_filtered"].numpy()

    meta = pd.read_csv(args.metadata)
    namecol = next(c for c in ["name", "file_name", "filename"] if c in meta.columns)
    meta["key"] = meta[namecol].astype(str).str.replace(".mp4", "", regex=False)
    m = df.merge(meta, on="key", how="inner")
    print(f"merged {len(m)} / {len(df)}")

    # ---- diagnostics -----------------------------------------------------
    print("\ntype values:", dict(m["type"].value_counts()))
    m["is_possible"] = (~m["type"].astype(str).str.contains("Impossible")).astype(int)
    print("is_possible counts:", dict(m["is_possible"].value_counts()))
    if "occluder" in m.columns:
        print("\ntype x occluder:\n", pd.crosstab(m["type"], m["occluder"]))
    comp = m.groupby("SceneIndex")["is_possible"].agg(["size", "sum"])
    print("\nscene composition (size, #possible):")
    print(comp.value_counts().head())

    # ---- report ----------------------------------------------------------
    cols = ["max_filtered", "avg_filtered"] + \
           [f"max_ctxt{c}" for c in ctxts] + [f"avg_ctxt{c}" for c in ctxts]
    print("\n=== All (Main set) ===")
    print(f"{'score':<16}{'Relative':>10}{'Absolute':>10}{'Oracle':>10}{'#pairs':>8}")
    for sc in cols:
        rel, n = relative_accuracy(m, sc)
        ab, best = absolute_accuracy(m, sc)
        print(f"{sc:<16}{rel:>10.2f}{ab:>10.2f}{best:>10.2f}{n:>8}")

    sc = args.score
    print(f"\n=== Breakdowns (score = {sc}) ===")
    for col in ["Difficulty", "Camera", "condition", "game_name"]:
        if col not in m.columns:
            continue
        print(f"\n-- {col} --")
        for v, sub in m.groupby(col):
            rel, n = relative_accuracy(sub, sc)
            ab, best = absolute_accuracy(sub, sc)
            print(f"  {str(v):<24} n={len(sub):>4}  rel={rel:6.2f}  "
                  f"abs={ab:6.2f}  oracle={best:6.2f}  pairs={n}")


if __name__ == "__main__":
    main()