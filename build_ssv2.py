#!/usr/bin/env python3
"""
Build train.csv / test.csv for Something-Something-V2.

Output format (no header):  <path_to_clip>,<class_index>

Note: the official test.json has no labels, so `validation.json` is used
as the test split (standard practice for SSv2).
"""

import csv
import json
import os
import re
import argparse


def _norm(t):
    """Normalize a template so train.json and labels.json keys line up."""
    t = t.replace("[", "").replace("]", "")
    return re.sub(r"\s+", " ", t).strip().lower()


def _find(labels_dir, *candidates):
    for name in candidates:
        p = os.path.join(labels_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of {candidates} found in {labels_dir}")


def build_csv(labels_dir, video_dir, out_dir, ext=".webm", skip_missing=True):
    os.makedirs(out_dir, exist_ok=True)

    with open(_find(labels_dir, "labels.json",
                    "something-something-v2-labels.json")) as f:
        raw_labels = json.load(f)
    template_to_idx = {_norm(k): int(v) for k, v in raw_labels.items()}
    print(f"{len(template_to_idx)} classes loaded")

    splits = {
        "train": _find(labels_dir, "train.json",
                       "something-something-v2-train.json"),
        "test":  _find(labels_dir, "validation.json",
                       "something-something-v2-validation.json"),
    }

    for split, json_path in splits.items():
        with open(json_path) as f:
            entries = json.load(f)

        rows, missing = [], 0
        for e in entries:
            key = _norm(e["template"])
            if key not in template_to_idx:
                raise KeyError(f"template not in labels.json: {e['template']!r}")

            path = os.path.join(video_dir, e["id"] + ext)
            if not os.path.exists(path):
                missing += 1
                if skip_missing:
                    continue
            rows.append([path, template_to_idx[key]])

        out_path = os.path.join(out_dir, f"{split}.csv")
        with open(out_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)

        print(f"{split}: {len(rows)} clips -> {out_path}"
              + (f"  ({missing} missing on disk)" if missing else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", default="datasets/ssv2/labels")
    ap.add_argument("--video_dir",
                    default="datasets/ssv2/20bn-something-something-v2")
    ap.add_argument("--out_dir", default="datasets/ssv2")
    ap.add_argument("--ext", default=".webm",
                    help="use .mp4 if you transcoded the clips")
    args = ap.parse_args()

    build_csv(args.labels_dir, args.video_dir, args.out_dir, args.ext)
