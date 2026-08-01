"""
Merge SSv2 + Kinetics clip lists into unified train/test CSVs, probing every
clip with decord so the output carries frame/fps/duration metadata.

  train_ssv2_kine.csv  = ssv2 train        + 90% of kinetics train
  test_ssv2_kine.csv   = ssv2 validation   + 10% of kinetics train

Output columns:  path,num_frames,fps,duration_sec,source
(add --with_label to get  path,label,num_frames,fps,duration_sec,source)

The kinetics eval split is stratified per class and seeded, so it is
reproducible and every class keeps representation on both sides.

Kinetics label indices are shifted by +174 (the SSv2 class count) so the
two label spaces do not collide in a single classification head.

Probing is the slow part.  Results are cached to a CSV (--probe_cache) keyed
by path, so re-runs only touch clips that have not been seen yet.  Clips that
decord cannot open are dropped and listed in <out_dir>/failed_clips.txt.
"""

import argparse
import csv
import math
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

SSV2_NUM_CLASSES = 174
VIDEO_EXTS = (".webm", ".mp4", ".avi", ".mkv", ".mov")


# ---------------------------------------------------------------- reading

def normalize_path(p, root):
    """Resolve a clip path to an absolute, normalised path.

    SSv2 lists tend to be relative ("datasets/ssv2/x.webm") while the
    Kinetics list is already absolute; this puts both in the same form so
    the output csv does not depend on the caller's cwd.
    """
    p = os.path.expanduser(str(p).strip())
    if not os.path.isabs(p):
        p = os.path.join(root, p) if root else os.path.abspath(p)
    return os.path.normpath(p)


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


def read_clip_csv(path, name, root=""):
    """-> list of (abs_clip_path, label_int). Prints what it inferred."""
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
            p = normalize_path(r[path_col], root)
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
            out.append((normalize_path(r[path_col], root), int(r[label_col])))

    n_missing = sum(1 for p, _ in out[:50] if not os.path.exists(p))
    print(f"  resolved -> {out[0][0]}")
    if n_missing:
        print(f"  WARNING: {n_missing}/50 sampled paths do not exist on disk "
              f"-- check --path_root")
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


# ---------------------------------------------------------------- probing

_DECORD = None


def _load_decord():
    """Import decord once, with a useful message if it is missing."""
    global _DECORD
    if _DECORD is None:
        try:
            import decord  # noqa: F401
        except ImportError:
            sys.exit("decord is not installed.  pip install decord "
                     "(or decord-gpu / build from source for GPU decode)")
        decord.bridge.set_bridge("native")
        _DECORD = decord
    return _DECORD


def probe_video(full):
    """-> ((num_frames, fps, duration_sec), None) or (None, 'reason')."""
    decord = _load_decord()
    try:
        # num_threads=1: we already parallelise over clips, and decord's
        # internal pool makes per-file memory blow up on big lists.
        vr = decord.VideoReader(full, ctx=decord.cpu(0), num_threads=1)
        num_frames = len(vr)
        fps = float(vr.get_avg_fps())
        del vr
    except Exception as e:  # decord raises bare RuntimeError on bad files
        return None, f"{type(e).__name__}: {e}"

    if num_frames <= 0:
        return None, "0 frames"
    if not fps or fps <= 0 or math.isnan(fps) or math.isinf(fps):
        return None, f"bad fps ({fps})"

    return (num_frames, round(fps, 4), round(num_frames / fps, 4)), None


def load_cache(path, root=""):
    """-> {abs_clip_path: (num_frames, fps, duration_sec)}

    Keys are normalised on load, so a cache written before paths were made
    absolute still hits instead of forcing a full re-probe.
    """
    if not path or not os.path.exists(path):
        return {}
    cache = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        for row in r:
            if len(row) < 4 or row[0] == "path":
                continue
            try:
                cache[normalize_path(row[0], root)] = (
                    int(row[1]), float(row[2]), float(row[3]))
            except ValueError:
                continue
    print(f"probe cache: {len(cache)} entries from {path}")
    return cache


def probe_all(paths, root, workers, cache_path, log_every=2000):
    """Probe every path not already in the cache. -> (meta, failures)."""
    meta = load_cache(cache_path, root)
    todo = [p for p in paths if p not in meta]
    failures = {}
    if not todo:
        print("probe: everything already cached")
        return meta, failures

    _load_decord()  # fail fast, before spawning threads
    print(f"probe: {len(todo)} clips to open with {workers} workers")

    cache_f = None
    writer = None
    lock = threading.Lock()
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        new_file = not os.path.exists(cache_path)
        cache_f = open(cache_path, "a", newline="")
        writer = csv.writer(cache_f)
        if new_file:
            writer.writerow(["path", "num_frames", "fps", "duration_sec"])

    t0 = time.time()
    done = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(probe_video, p): p for p in todo}
            for fut in as_completed(futs):
                p = futs[fut]
                info, err = fut.result()
                done += 1
                if info is None:
                    failures[p] = err
                else:
                    meta[p] = info
                    if writer:
                        with lock:
                            writer.writerow([p, info[0], info[1], info[2]])
                            if done % log_every == 0:
                                cache_f.flush()
                if done % log_every == 0 or done == len(todo):
                    rate = done / max(time.time() - t0, 1e-6)
                    eta = (len(todo) - done) / max(rate, 1e-6)
                    print(f"  {done}/{len(todo)}  {rate:6.1f} clips/s  "
                          f"eta {eta/60:5.1f} min  failed={len(failures)}")
    finally:
        if cache_f:
            cache_f.close()

    return meta, failures


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

def report(split_name, rows, meta):
    src = Counter(r[2] for r in rows)
    total = len(rows)
    n_cls = len({r[1] for r in rows})
    print(f"\n{split_name}: {total} clips, {n_cls} distinct classes")
    for s, n in sorted(src.items()):
        durs = [meta[r[0]][2] for r in rows if r[2] == s]
        frames = [meta[r[0]][0] for r in rows if r[2] == s]
        fpss = [meta[r[0]][1] for r in rows if r[2] == s]
        print(f"  {s:<10} {n:>7}  ({n / total:6.2%})  "
              f"dur {min(durs):5.2f}/{sum(durs)/len(durs):5.2f}/{max(durs):6.2f}s "
              f"frames {min(frames)}/{sum(frames)//len(frames)}/{max(frames)} "
              f"fps {min(fpss):.1f}-{max(fpss):.1f}")
    print("   (min/mean/max)")


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
                    help="write the trailing column: ssv2 | kinetics")
    ap.add_argument("--no_source_col", dest="source_col", action="store_false")

    # probing
    ap.add_argument("--path_root", default="",
                    help="base dir for RELATIVE csv paths (e.g. /home/rvm); "
                         "absolute paths are left alone. default: cwd")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--probe_cache", default=None,
                    help="default: <out_dir>/clip_meta_cache.csv")
    ap.add_argument("--min_frames", type=int, default=0,
                    help="drop clips with fewer decoded frames than this")
    ap.add_argument("--min_duration", type=float, default=0.0)
    ap.add_argument("--with_label", action="store_true",
                    help="insert the label column after path")
    ap.add_argument("--header", action="store_true",
                    help="write a header row in the output csvs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = args.probe_cache or os.path.join(args.out_dir,
                                                  "clip_meta_cache.csv")

    root = os.path.abspath(os.path.expanduser(args.path_root)) \
        if args.path_root else os.getcwd()
    print(f"relative paths resolved against: {root}\n")

    ssv2_train = read_clip_csv(args.ssv2_train, "ssv2-train", root)
    ssv2_test = read_clip_csv(args.ssv2_test, "ssv2-test", root)
    kinetics = read_clip_csv(args.kinetics, "kinetics", root)

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

    # ---- decord probe -------------------------------------------------
    all_paths = sorted({p for p, _, _ in train} | {p for p, _, _ in test})
    print(f"\nprobing {len(all_paths)} unique clips with decord ...")
    meta, failures = probe_all(all_paths, root,
                               args.num_workers, cache_path)

    if failures:
        fail_path = os.path.join(args.out_dir, "failed_clips.txt")
        with open(fail_path, "w") as f:
            for p, why in sorted(failures.items()):
                f.write(f"{p}\t{why}\n")
        print(f"\n{len(failures)} clips could not be opened -> {fail_path}")
        for p, why in list(sorted(failures.items()))[:5]:
            print(f"  {p}  ({why})")

    def keep(row):
        info = meta.get(row[0])
        if info is None:
            return False
        return info[0] >= args.min_frames and info[2] >= args.min_duration

    n_before = len(train) + len(test)
    train = [r for r in train if keep(r)]
    test = [r for r in test if keep(r)]
    dropped = n_before - len(train) - len(test)
    if dropped:
        print(f"dropped {dropped} clips (unreadable / below "
              f"min_frames={args.min_frames}, min_duration={args.min_duration})")
    if not train or not test:
        sys.exit("nothing left after probing; check --path_root")

    # ---- write --------------------------------------------------------
    cols = ["path"] + (["label"] if args.with_label else []) \
        + ["num_frames", "fps", "duration_sec"] \
        + (["source"] if args.source_col else [])

    for name, rows in (("train_ssv2_kine.csv", train),
                       ("test_ssv2_kine.csv", test)):
        out = os.path.join(args.out_dir, name)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            if args.header:
                w.writerow(cols)
            for p, l, s in rows:
                n, fps, dur = meta[p]
                row = [p] + ([l] if args.with_label else []) + [n, fps, dur]
                if args.source_col:
                    row.append(s)
                w.writerow(row)
        print(f"\nwrote {out}  ({len(rows)} rows, cols={','.join(cols)})")

    report("TRAIN", train, meta)
    report("TEST", test, meta)

    n_ss = sum(1 for r in train + test if r[2] == "ssv2")
    n_kin = sum(1 for r in train + test if r[2] == "kinetics")
    print(f"\noverall  ssv2={n_ss} ({n_ss / (n_ss + n_kin):.2%})  "
          f"kinetics={n_kin} ({n_kin / (n_ss + n_kin):.2%})")
    n_kin_eval = sum(1 for r in test if r[2] == "kinetics")
    print(f"kinetics eval holdout: {n_kin_eval}/{n_kin} "
          f"({n_kin_eval / max(n_kin, 1):.2%})")

    total_hours = sum(meta[r[0]][2] for r in train + test) / 3600.0
    print(f"total video: {total_hours:.1f} h")

    overlap = {p for p, _, _ in train} & {p for p, _, _ in test}
    print(f"train/test path overlap: {len(overlap)}"
          + ("  <-- LEAKAGE" if overlap else "  (clean)"))


if __name__ == "__main__":
    main()