# #!/usr/bin/env python3
# """
# Build train.csv / test.csv for Something-Something-V2.

# Output format (no header):  <path_to_clip>,<class_index>

# Note: the official test.json has no labels, so `validation.json` is used
# as the test split (standard practice for SSv2).
# """

# import csv
# import json
# import os
# import re
# import argparse


# def _norm(t):
#     """Normalize a template so train.json and labels.json keys line up."""
#     t = t.replace("[", "").replace("]", "")
#     return re.sub(r"\s+", " ", t).strip().lower()


# def _find(labels_dir, *candidates):
#     for name in candidates:
#         p = os.path.join(labels_dir, name)
#         if os.path.exists(p):
#             return p
#     raise FileNotFoundError(f"None of {candidates} found in {labels_dir}")


# def build_csv(labels_dir, video_dir, out_dir, ext=".webm", skip_missing=True):
#     os.makedirs(out_dir, exist_ok=True)

#     with open(_find(labels_dir, "labels.json",
#                     "something-something-v2-labels.json")) as f:
#         raw_labels = json.load(f)
#     template_to_idx = {_norm(k): int(v) for k, v in raw_labels.items()}
#     print(f"{len(template_to_idx)} classes loaded")

#     splits = {
#         "train": _find(labels_dir, "train.json",
#                        "something-something-v2-train.json"),
#         "test":  _find(labels_dir, "validation.json",
#                        "something-something-v2-validation.json"),
#     }

#     for split, json_path in splits.items():
#         with open(json_path) as f:
#             entries = json.load(f)

#         rows, missing = [], 0
#         for e in entries:
#             key = _norm(e["template"])
#             if key not in template_to_idx:
#                 raise KeyError(f"template not in labels.json: {e['template']!r}")

#             path = os.path.join(video_dir, e["id"] + ext)
#             if not os.path.exists(path):
#                 missing += 1
#                 if skip_missing:
#                     continue
#             rows.append([path, template_to_idx[key]])

#         out_path = os.path.join(out_dir, f"{split}.csv")
#         with open(out_path, "w", newline="") as f:
#             csv.writer(f).writerows(rows)

#         print(f"{split}: {len(rows)} clips -> {out_path}"
#               + (f"  ({missing} missing on disk)" if missing else ""))


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--labels_dir", default="datasets/ssv2/labels")
#     ap.add_argument("--video_dir",
#                     default="datasets/ssv2/20bn-something-something-v2")
#     ap.add_argument("--out_dir", default="datasets/ssv2")
#     ap.add_argument("--ext", default=".webm",
#                     help="use .mp4 if you transcoded the clips")
#     args = ap.parse_args()

#     build_csv(args.labels_dir, args.video_dir, args.out_dir, args.ext)
#!/usr/bin/env python3
"""
Merge SSv2 + Kinetics clip lists into unified train/test CSVs.

  train_ssv2_kine.csv  = ssv2 train        + 90% of kinetics train
  test_ssv2_kine.csv   = ssv2 validation   + 10% of kinetics train

The kinetics eval split is stratified per class and seeded, so it is
reproducible and every class keeps representation on both sides.

Kinetics label indices are shifted by +174 (the SSv2 class count) so the
two label spaces do not collide in a single classification head.
"""

import argparse
import csv
import os
import random
from collections import Counter, defaultdict

SSV2_NUM_CLASSES = 174
VIDEO_EXTS = (".webm", ".mp4", ".avi", ".mkv", ".mov")


# ---------------------------------------------------------------- reading

def _sniff(path):
    """Return (delimiter, has_header) for a csv-ish file."""
    with open(path, newline="") as f:
        sample = f.read(8192)
    try:
        delim = csv.Sniffer().sniff(sample, delimiters=",\t; ").delimiter
    except csv.Error:
        delim = ","
    first = sample.splitlines()[0]
    # header if no field on the first line looks like a path or an int
    fields = first.split(delim)
    looks_data = any(
        f.strip().lower().endswith(VIDEO_EXTS) or f.strip().lstrip("-").isdigit()
        for f in fields
    )
    return delim, not looks_data


def _pick_columns(rows):
    """Guess which column holds the path and which holds the label index."""
    ncol = len(rows[0])
    probe = rows[: min(200, len(rows))]

    path_col = None
    for c in range(ncol):
        if all(str(r[c]).strip().lower().endswith(VIDEO_EXTS) for r in probe):
            path_col = c
            break
    if path_col is None:  # fall back: column containing a separator
        for c in range(ncol):
            if all(("/" in str(r[c]) or "\\" in str(r[c])) for r in probe):
                path_col = c
                break
    if path_col is None:
        raise ValueError("could not identify the path column")

    label_col = None
    for c in range(ncol):
        if c == path_col:
            continue
        if all(str(r[c]).strip().lstrip("-").isdigit() for r in probe):
            label_col = c
            break
    return path_col, label_col


def read_clip_csv(path, name):
    """-> list of (clip_path, label_int). Prints what it inferred."""
    delim, has_header = _sniff(path)
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f, delimiter=delim) if r]
    header = rows.pop(0) if has_header else None

    path_col, label_col = _pick_columns(rows)
    print(f"[{name}] {path}")
    print(f"  delimiter={delim!r}  header={header}  rows={len(rows)}")
    print(f"  path_col={path_col}  label_col={label_col}")
    print(f"  sample -> {rows[0]}")

    out = []
    if label_col is None:
        # No label column -> assume the class is the parent directory name,
        # e.g. .../kinetics_dataset/clapping/xxx.mp4  ->  "clapping"
        for r in rows:
            p = r[path_col].strip()
            out.append((p, os.path.basename(os.path.dirname(p))))
        classes = sorted({l for _, l in out})
        print(f"  no label column -> derived from parent dir: "
              f"{len(classes)} classes, e.g. {classes[:3]}")
        if len(classes) < 2:
            raise ValueError(
                f"[{name}] only {len(classes)} class dir(s) found; the paths "
                f"are probably not organised as <class>/<clip>."
            )
    else:
        for r in rows:
            out.append((r[path_col].strip(), int(r[label_col])))
    return out


def encode_labels(items, offset, mapping_out=None):
    """Map string class names -> contiguous ints starting at `offset`."""
    classes = sorted({l for _, l in items if isinstance(l, str)})
    if not classes:
        return items, {}
    name_to_idx = {c: i + offset for i, c in enumerate(classes)}
    if mapping_out:
        with open(mapping_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_name", "index"])
            for c in classes:
                w.writerow([c, name_to_idx[c]])
        print(f"wrote class map -> {mapping_out}")
    return [(p, name_to_idx[l]) for p, l in items], name_to_idx


# ---------------------------------------------------------------- splitting

def stratified_split(items, frac, seed):
    """Hold out `frac` of items per class. -> (kept, held_out)"""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for it in items:
        by_class[it[1]].append(it)

    kept, held = [], []
    for label, group in by_class.items():
        rng.shuffle(group)
        n_hold = max(1, round(len(group) * frac)) if len(group) > 1 else 0
        held.extend(group[:n_hold])
        kept.extend(group[n_hold:])
    rng.shuffle(kept)
    rng.shuffle(held)
    return kept, held


# ---------------------------------------------------------------- reporting

def report(split_name, rows):
    src = Counter(r[2] for r in rows)
    total = len(rows)
    n_cls = len({r[1] for r in rows})
    print(f"\n{split_name}: {total} clips, {n_cls} distinct classes")
    for s, n in sorted(src.items()):
        print(f"  {s:<10} {n:>7}  ({n / total:6.2%})")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssv2_train", default="datasets/ssv2/train.csv")
    ap.add_argument("--ssv2_test", default="datasets/ssv2/test.csv")
    ap.add_argument("--kinetics",
                    default="datasets/kinetics_train_set_clip_paths.csv")
    ap.add_argument("--out_dir", default="datasets")
    ap.add_argument("--eval_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--offset_kinetics", action="store_true", default=True,
                    help=f"add {SSV2_NUM_CLASSES} to kinetics label indices")
    ap.add_argument("--no_offset_kinetics", dest="offset_kinetics",
                    action="store_false")
    ap.add_argument("--source_col", action="store_true", default=True,
                    help="write a third column: ssv2 | kinetics")
    ap.add_argument("--no_source_col", dest="source_col", action="store_false")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ssv2_train = read_clip_csv(args.ssv2_train, "ssv2-train")
    ssv2_test = read_clip_csv(args.ssv2_test, "ssv2-test")
    kinetics = read_clip_csv(args.kinetics, "kinetics")

    offset = SSV2_NUM_CLASSES if args.offset_kinetics else 0

    if any(isinstance(l, str) for _, l in kinetics):
        kinetics, name_to_idx = encode_labels(
            kinetics, offset,
            mapping_out=os.path.join(args.out_dir, "kinetics_class_map.csv"))
    elif offset:
        kinetics = [(p, l + offset) for p, l in kinetics]

    print(f"\nkinetics label range: {min(l for _, l in kinetics)}"
          f"..{max(l for _, l in kinetics)}  (offset +{offset})")
    print(f"total classes in unified space: "
          f"{SSV2_NUM_CLASSES + len({l for _, l in kinetics})}")

    kin_train, kin_eval = stratified_split(kinetics, args.eval_frac, args.seed)

    train = ([(p, l, "ssv2") for p, l in ssv2_train]
             + [(p, l, "kinetics") for p, l in kin_train])
    test = ([(p, l, "ssv2") for p, l in ssv2_test]
            + [(p, l, "kinetics") for p, l in kin_eval])

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(test)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("train_ssv2_kine.csv", train),
                       ("test_ssv2_kine.csv", test)):
        out = os.path.join(args.out_dir, name)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            for p, l, s in rows:
                w.writerow([p, l, s] if args.source_col else [p, l])
        print(f"\nwrote {out}")

    report("TRAIN", train)
    report("TEST", test)

    n_ss = len(ssv2_train) + len(ssv2_test)
    n_kin = len(kinetics)
    print(f"\noverall  ssv2={n_ss} ({n_ss / (n_ss + n_kin):.2%})  "
          f"kinetics={n_kin} ({n_kin / (n_ss + n_kin):.2%})")
    print(f"kinetics eval holdout: {len(kin_eval)}/{n_kin} "
          f"({len(kin_eval) / n_kin:.2%})")

    overlap = {p for p, _, _ in train} & {p for p, _, _ in test}
    print(f"train/test path overlap: {len(overlap)}"
          + ("  <-- LEAKAGE" if overlap else "  (clean)"))


if __name__ == "__main__":
    main()