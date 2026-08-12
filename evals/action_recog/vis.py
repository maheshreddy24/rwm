#!/usr/bin/env python3
"""
Parse an action-recognition (SSv2) training log and plot:
  1. steps vs eval accuracy, one curve per num_frames setting
  2. steps vs eval loss, one curve per num_frames setting
  3. num_frames vs accuracy (final + best)

Handles:
  - multiple "New iteration" blocks (one per num_frames setting)
  - resumed runs ("skipping num_frames = X", "resumed num_frames = X ...")
    which are merged back into the same setting
  - end-of-epoch evals that have no step number on the line

Usage:
    python plot_ablation.py train.log --outdir figs
"""

import argparse
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- parsing ---

RE_NEW_RUN = re.compile(r"num_frames\s*=\s*(\d+),\s*iteration\s*=\s*(\d+)")
RE_STEP = re.compile(r"\[epoch (\d+) step (\d+)\]\s+loss\s+([\d.]+)")
RE_EVAL = re.compile(r"\[epoch (\d+)\]\s+eval loss\s+([\d.]+)\s+acc\s+([\d.]+)")
RE_TRAIN_AVG = re.compile(r"\[epoch (\d+)\]\s+average train loss:\s+([\d.]+)")
RE_SKIP = re.compile(r"skipping num_frames\s*=\s*(\d+)")


def parse_log(path):
    """Return OrderedDict: num_frames -> dict with train/eval records."""
    runs = OrderedDict()
    cur = None          # record dict for the active num_frames
    last_step = 0       # most recent global step seen in the active run
    step_gap = 200      # inferred logging interval

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if RE_SKIP.search(line):
                continue

            m = RE_NEW_RUN.search(line)
            if m:
                nf = int(m.group(1))
                # If this num_frames was seen before, it's a resume -> reuse it.
                cur = runs.setdefault(
                    nf,
                    {"num_frames": nf,
                     "iteration": int(m.group(2)),
                     "train": [],        # (step, loss)
                     "eval": [],         # (step, epoch, loss, acc)
                     "epoch_avg": []},   # (epoch, avg train loss)
                )
                last_step = max((s for s, _ in cur["train"]), default=0)
                continue

            if cur is None:
                continue

            m = RE_STEP.search(line)
            if m:
                epoch, step, loss = int(m.group(1)), int(m.group(2)), float(m.group(3))
                if last_step and step > last_step:
                    step_gap = min(step_gap, step - last_step) or step_gap
                last_step = step
                cur["train"].append((step, loss))
                continue

            m = RE_EVAL.search(line)
            if m:
                epoch, loss, acc = int(m.group(1)), float(m.group(2)), float(m.group(3))
                # Evals logged right after "average train loss" close out the
                # epoch and sit slightly past the last logged step; nudge them
                # forward so they don't stack on top of the previous eval.
                step = last_step
                if cur["eval"] and cur["eval"][-1][0] == step:
                    step = last_step + step_gap
                cur["eval"].append((step, epoch, loss, acc))
                continue

            m = RE_TRAIN_AVG.search(line)
            if m:
                cur["epoch_avg"].append((int(m.group(1)), float(m.group(2))))

    # Sort + dedupe (a resume can replay an eval point).
    for r in runs.values():
        r["train"] = sorted(dict(r["train"]).items())
        ev = {(s, e): (l, a) for s, e, l, a in r["eval"]}
        r["eval"] = sorted((s, e, l, a) for (s, e), (l, a) in ev.items())
    return runs


# --------------------------------------------------------------- plotting ---

def smooth(ys, k):
    """Simple centred moving average, used only for the noisy train loss."""
    if k <= 1:
        return ys
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - k // 2), min(len(ys), i + k // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def colours(runs):
    cmap = plt.get_cmap("viridis")
    n = max(len(runs) - 1, 1)
    return {nf: cmap(i / n) for i, nf in enumerate(sorted(runs, reverse=True))}


def plot_curves(runs, key, ylabel, title, outfile, marker="o"):
    """key: 3 -> acc, 2 -> eval loss (index into the eval tuple)."""
    col = colours(runs)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for nf in sorted(runs, reverse=True):
        ev = runs[nf]["eval"]
        if not ev:
            continue
        xs = [e[0] for e in ev]
        ys = [e[key] for e in ev]
        ax.plot(xs, ys, marker=marker, ms=4, lw=1.8,
                color=col[nf], label=f"{nf} frames")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, ls="--")
    ax.legend(title="num_frames")
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)
    print(f"wrote {outfile}")


def plot_train_loss(runs, outfile, window=5):
    col = colours(runs)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for nf in sorted(runs, reverse=True):
        tr = runs[nf]["train"]
        if not tr:
            continue
        xs = [s for s, _ in tr]
        ys = smooth([l for _, l in tr], window)
        ax.plot(xs, ys, lw=1.5, color=col[nf], label=f"{nf} frames")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"train loss (moving avg, w={window})")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(title="num_frames")
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)
    print(f"wrote {outfile}")


def plot_frames_vs_acc(runs, outfile):
    nfs = sorted(runs)
    final, best = [], []
    for nf in nfs:
        accs = [e[3] for e in runs[nf]["eval"]]
        final.append(accs[-1] if accs else float("nan"))
        best.append(max(accs) if accs else float("nan"))

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(nfs, best, marker="s", lw=2, label="best eval acc")
    ax.plot(nfs, final, marker="o", lw=2, ls="--", label="final eval acc")
    for x, y in zip(nfs, best):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("num_frames")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title("Accuracy vs number of input frames (SSv2)")
    ax.set_xticks(nfs)
    ax.grid(alpha=0.3, ls="--")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=160)
    plt.close(fig)
    print(f"wrote {outfile}")


def print_summary(runs):
    print(f"\n{'frames':>7} {'evals':>6} {'last step':>10} {'final acc':>10} {'best acc':>9}")
    for nf in sorted(runs, reverse=True):
        ev = runs[nf]["eval"]
        if not ev:
            continue
        accs = [e[3] for e in ev]
        print(f"{nf:>7} {len(ev):>6} {ev[-1][0]:>10} {accs[-1]:>10.4f} {max(accs):>9.4f}")
    print()


# ------------------------------------------------------------------- main ---

def main():
    p = argparse.ArgumentParser()
    p.add_argument("logfile")
    p.add_argument("--outdir", default=".")
    p.add_argument("--csv", help="optional path to dump parsed eval points")
    args = p.parse_args()

    runs = parse_log(args.logfile)
    if not runs:
        raise SystemExit("no runs parsed - check the log format")
    print_summary(runs)

    import os
    os.makedirs(args.outdir, exist_ok=True)
    j = lambda n: os.path.join(args.outdir, n)

    plot_curves(runs, 3, "eval top-1 accuracy",
                "Eval accuracy vs training step", j("steps_vs_acc.png"))
    plot_curves(runs, 2, "eval loss",
                "Eval loss vs training step", j("steps_vs_eval_loss.png"))
    plot_train_loss(runs, j("steps_vs_train_loss.png"))
    plot_frames_vs_acc(runs, j("frames_vs_acc.png"))

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("num_frames,step,epoch,eval_loss,acc\n")
            for nf in sorted(runs, reverse=True):
                for s, e, l, a in runs[nf]["eval"]:
                    f.write(f"{nf},{s},{e},{l},{a}\n")
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()