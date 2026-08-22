#!/usr/bin/env python3
"""Merge clip lists into ONE manifest per split for RVMDataset.

Designed to sit downstream of make_ssv2_kine_csv.py: if an input csv already
carries num_frames/fps/duration_sec (as train_ssv2_kine.csv does), those values
are passed straight through and the clip is NOT re-probed. Only inputs without
metadata (e.g. a bare ego4d path list) get probed, and results land in the same
`clip_meta_cache.csv` format your merge script uses, so the two share a cache.

Handles headerless csvs (the merge script's default) by inferring the layout:
    path[,label],num_frames,fps,duration_sec[,source]

Each --input is `CSV` or `CSV:SOURCE`. A `source` column in the csv wins;
SOURCE from the cli is the fallback for csvs that have none.

Output (always with header):
    path,label,num_frames,fps,duration_sec,source

Usage:
    python build_manifest.py \
        --input datasets/train_ssv2_kine.csv \
        --input datasets/ego4d_train.csv:ego4d \
        --out datasets/train_manifest.csv \
        --path_root /home/rvm --num_workers 32

    python build_manifest.py \
        --input datasets/test_ssv2_kine.csv \
        --input datasets/ego4d_test.csv:ego4d \
        --out datasets/test_manifest.csv \
        --path_root /home/rvm --num_workers 32
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

VIDEO_EXTS = (".webm", ".mp4", ".avi", ".mkv", ".mov")


# ---------------------------------------------------------------- reading

def normalize_path(p, root):
    """Same contract as make_ssv2_kine_csv.normalize_path."""
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
    fields = first.split(delim)
    looks_data = any(
        f.strip().lower().endswith(VIDEO_EXTS) or f.strip().lstrip("-").isdigit()
        for f in fields
    )
    return delim, not looks_data


def _is_num(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _infer_layout(rows):
    """Guess column indices for a headerless csv.

    Expected writer order: path[,label],num_frames,fps,duration_sec[,source]
    -> dict with keys path/label/num_frames/fps/duration/source (None if absent)
    """
    ncol = len(rows[0])
    probe = rows[: min(200, len(rows))]
    layout = dict.fromkeys(
        ("path", "label", "num_frames", "fps", "duration", "source")
    )

    for c in range(ncol):
        if all(str(r[c]).strip().lower().endswith(VIDEO_EXTS) for r in probe):
            layout["path"] = c
            break
    if layout["path"] is None:
        for c in range(ncol):
            if all(("/" in str(r[c]) or "\\" in str(r[c])) for r in probe):
                layout["path"] = c
                break
    if layout["path"] is None:
        raise ValueError("could not identify the path column")

    for c in range(ncol):
        if c == layout["path"]:
            continue
        if all(str(r[c]).strip() and not _is_num(r[c]) for r in probe):
            layout["source"] = c
            break

    numeric = [
        c for c in range(ncol)
        if c not in (layout["path"], layout["source"])
        and all(_is_num(r[c]) for r in probe)
    ]
    # trailing three numerics are always num_frames, fps, duration_sec
    if len(numeric) >= 4:
        layout["label"] = numeric[-4]
    if len(numeric) >= 3:
        layout["num_frames"], layout["fps"], layout["duration"] = numeric[-3:]
    return layout


def read_input(spec, root):
    """-> list of dicts: path, label, source, and meta (or None if unprobed)."""
    if ":" in spec and not os.path.exists(spec):
        csv_path, _, cli_source = spec.rpartition(":")
    else:
        csv_path, cli_source = spec, None
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    delim, has_header = _sniff(csv_path)
    with open(csv_path, newline="") as f:
        raw = [r for r in csv.reader(f, delimiter=delim) if r]

    if has_header:
        header = [h.strip().lower() for h in raw.pop(0)]
        idx = {name: i for i, name in enumerate(header)}
        layout = {
            "path": idx.get("path"),
            "label": idx.get("label"),
            "num_frames": idx.get("num_frames"),
            "fps": idx.get("fps"),
            "duration": idx.get("duration_sec", idx.get("duration")),
            "source": idx.get("source"),
        }
        if layout["path"] is None:
            raise ValueError(f"{csv_path}: header has no `path` column: {header}")
    else:
        header = None
        layout = _infer_layout(raw)

    has_meta = all(
        layout[k] is not None for k in ("num_frames", "fps", "duration")
    )
    print(f"[{os.path.basename(csv_path)}] rows={len(raw)} delim={delim!r} "
          f"header={'yes' if header else 'inferred'} meta={'yes' if has_meta else 'no'}")
    print(f"  layout={ {k: v for k, v in layout.items() if v is not None} }")

    out = []
    for r in raw:
        path = str(r[layout["path"]]).strip()
        if not path:
            continue
        source = (
            str(r[layout["source"]]).strip() if layout["source"] is not None else ""
        ) or cli_source
        if not source:
            raise ValueError(
                f"{csv_path}: no source column and no cli source. "
                f"Use --input {csv_path}:<source>"
            )
        label = -1
        if layout["label"] is not None:
            try:
                label = int(float(r[layout["label"]]))
            except (ValueError, IndexError):
                label = -1

        meta = None
        if has_meta:
            try:
                num_frames = int(float(r[layout["num_frames"]]))
                fps = float(r[layout["fps"]])
                duration = float(r[layout["duration"]])
                if num_frames > 0 and fps > 0:
                    meta = (num_frames, round(fps, 4), round(duration, 4))
            except (ValueError, IndexError):
                meta = None

        out.append(
            {
                "path": normalize_path(path, root),
                "label": label,
                "source": source,
                "meta": meta,
            }
        )
    print(f"  resolved -> {out[0]['path']}")
    return out


# ---------------------------------------------------------------- probing

_DECORD = None


def _load_decord():
    global _DECORD
    if _DECORD is None:
        try:
            import decord  # noqa: F401
        except ImportError:
            sys.exit("decord is not installed.  pip install decord")
        decord.bridge.set_bridge("native")
        _DECORD = decord
    return _DECORD


def _parse_rate(value):
    """'30000/1001' -> 29.97"""
    if not value or value == "N/A":
        return 0.0
    try:
        if "/" in value:
            num, den = value.split("/")
            den = float(den)
            return float(num) / den if den else 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def probe_ffprobe(path):
    """Header-only read: ~100x faster than decord, which builds a frame index.
    Frame counts can be off by a frame or two on some containers; harmless here
    because the dataset re-checks len(vr) and clamps indices at load time."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,avg_frame_rate,r_frame_rate,duration",
        "-show_entries", "format=duration", "-of", "json", path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return None, f"{type(e).__name__}"
    if proc.returncode != 0:
        return None, "ffprobe failed"
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None, "bad ffprobe json"

    streams = info.get("streams") or []
    if not streams:
        return None, "no video stream"
    s = streams[0]

    fps = _parse_rate(s.get("avg_frame_rate")) or _parse_rate(s.get("r_frame_rate"))
    try:
        duration = float(s.get("duration") or info.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    nb = s.get("nb_frames")
    num_frames = int(nb) if nb not in (None, "N/A") else 0
    if num_frames <= 0 and fps > 0 and duration > 0:
        num_frames = int(round(duration * fps))
    if num_frames <= 0:
        return None, "0 frames"
    if fps <= 0 and duration > 0:
        fps = num_frames / duration
    if not fps or fps <= 0 or math.isnan(fps) or math.isinf(fps):
        return None, f"bad fps ({fps})"
    return (num_frames, round(fps, 4), round(num_frames / fps, 4)), None


def probe_decord(path):
    """Exact, matches what the dataset sees via len(vr). Slow on long videos."""
    decord = _load_decord()
    try:
        vr = decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)
        num_frames = len(vr)
        fps = float(vr.get_avg_fps())
        del vr
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if num_frames <= 0:
        return None, "0 frames"
    if not fps or fps <= 0 or math.isnan(fps) or math.isinf(fps):
        return None, f"bad fps ({fps})"
    return (num_frames, round(fps, 4), round(num_frames / fps, 4)), None


def probe_video(path, backend):
    if not os.path.exists(path):
        return None, "missing on disk"
    if backend == "decord":
        return probe_decord(path)
    info, err = probe_ffprobe(path)
    if info is None and backend == "auto":
        return probe_decord(path)
    return info, err


def load_cache(path, root=""):
    """Same format as make_ssv2_kine_csv's clip_meta_cache.csv."""
    if not path or not os.path.exists(path):
        return {}
    cache = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 4 or row[0] == "path":
                continue
            try:
                cache[normalize_path(row[0], root)] = (
                    int(float(row[1])), float(row[2]), float(row[3]))
            except ValueError:
                continue
    print(f"probe cache: {len(cache)} entries from {path}")
    return cache


def probe_all(paths, root, workers, cache_path, backend, log_every=200):
    meta = load_cache(cache_path, root)
    todo = [p for p in paths if p not in meta]
    failures = {}
    if not todo:
        print("probe: everything already cached")
        return meta, failures

    if backend == "decord":
        _load_decord()  # fail fast before spawning threads
    print(f"probe: {len(todo)} clips with {workers} workers (backend={backend})")

    cache_f = writer = None
    lock = threading.Lock()
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        new_file = not os.path.exists(cache_path)
        cache_f = open(cache_path, "a", newline="")
        writer = csv.writer(cache_f)
        if new_file:
            writer.writerow(["path", "num_frames", "fps", "duration_sec"])

    t0, done = time.time(), 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(probe_video, p, backend): p for p in todo}
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


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True,
                    metavar="CSV[:SOURCE]", help="repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--path_root", default="",
                    help="base dir for RELATIVE csv paths; default cwd")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--probe_cache", default=None,
                    help="default: <out_dir>/clip_meta_cache.csv "
                         "(shared with make_ssv2_kine_csv.py)")
    ap.add_argument("--probe_backend", choices=["auto", "ffprobe", "decord"],
                    default="auto",
                    help="auto = ffprobe, decord fallback. decord = exact but slow")
    ap.add_argument("--min_frames", type=int, default=0)
    ap.add_argument("--min_duration", type=float, default=0.0)
    ap.add_argument("--reprobe", action="store_true",
                    help="ignore metadata already present in the input csvs")
    args = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    cache_path = args.probe_cache or os.path.join(out_dir, "clip_meta_cache.csv")
    root = os.path.abspath(os.path.expanduser(args.path_root)) if args.path_root \
        else os.getcwd()
    print(f"relative paths resolved against: {root}\n")

    entries, seen = [], set()
    for spec in args.input:
        for e in read_input(spec, root):
            if e["path"] in seen:
                continue
            seen.add(e["path"])
            if args.reprobe:
                e["meta"] = None
            entries.append(e)
    print(f"\n{len(entries)} unique clips across {len(args.input)} input(s)")

    need_probe = sorted({e["path"] for e in entries if e["meta"] is None})
    passthrough = len(entries) - len(need_probe)
    print(f"metadata already in csv: {passthrough}   to probe: {len(need_probe)}\n")

    meta, failures = ({}, {})
    if need_probe:
        meta, failures = probe_all(need_probe, root, args.num_workers,
                                   cache_path, args.probe_backend)

    rows, dropped = [], 0
    for e in entries:
        info = e["meta"] or meta.get(e["path"])
        if info is None or info[0] < args.min_frames or info[2] < args.min_duration:
            dropped += 1
            continue
        rows.append({
            "path": e["path"], "label": e["label"], "num_frames": info[0],
            "fps": info[1], "duration_sec": info[2], "source": e["source"],
        })

    if dropped:
        print(f"\ndropped {dropped} clips (unreadable / below min_frames="
              f"{args.min_frames}, min_duration={args.min_duration})")
    if not rows:
        sys.exit("nothing left; check --path_root")

    rows.sort(key=lambda r: (r["source"], r["path"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["path", "label", "num_frames", "fps",
                           "duration_sec", "source"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nwrote {len(rows)} rows -> {args.out}")
    for source, count in sorted(Counter(r["source"] for r in rows).items()):
        sub = [r for r in rows if r["source"] == source]
        hours = sum(r["duration_sec"] for r in sub) / 3600
        durs = [r["duration_sec"] for r in sub]
        print(f"  {source:12s} {count:7d} clips  {hours:9.1f} h   "
              f"dur {min(durs):6.1f}/{sum(durs)/len(durs):7.1f}/{max(durs):8.1f}s "
              f"(min/mean/max)")
    total = sum(r["duration_sec"] for r in rows) / 3600
    print(f"  {'TOTAL':12s} {len(rows):7d} clips  {total:9.1f} h")

    if failures:
        fail_path = args.out + ".failed.txt"
        with open(fail_path, "w") as f:
            for p, why in sorted(failures.items()):
                f.write(f"{p}\t{why}\n")
        print(f"\n{len(failures)} clips could not be opened -> {fail_path}")
        for p, why in list(sorted(failures.items()))[:5]:
            print(f"  {p}  ({why})")


if __name__ == "__main__":
    main()