#!/usr/bin/env python3
"""Build a frame-count manifest for a video CSV. Run this ONCE per dataset, offline.

Reads a csv with a `path` column and writes:
    path,num_frames,fps,duration,source

`ffprobe` only parses container headers, so this is ~1-5 ms per file and
embarrassingly parallel. Falls back to decord for containers whose headers lie
(some Ego4D mp4s report nb_frames=N/A).

Usage:
    python build_manifest.py --video-csv ego4d.csv  --out ego4d_manifest.csv  --source ego4d   --workers 32
    python build_manifest.py --video-csv ssv2_k400.csv --out short_manifest.csv --source short  --workers 32
"""

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


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
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,avg_frame_rate,r_frame_rate,duration",
        "-show_entries", "format=duration",
        "-of", "json", path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if proc.returncode != 0:
        return None

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None

    streams = info.get("streams") or []
    if not streams:
        return None
    stream = streams[0]

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate"))
    duration = stream.get("duration") or info.get("format", {}).get("duration") or 0.0
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0

    nb_frames = stream.get("nb_frames")
    num_frames = int(nb_frames) if nb_frames not in (None, "N/A") else 0
    if num_frames <= 0 and fps > 0 and duration > 0:
        num_frames = int(round(duration * fps))
    if num_frames <= 0:
        return None
    if fps <= 0 and duration > 0:
        fps = num_frames / duration
    if fps <= 0:
        return None

    return num_frames, fps, duration


def probe_decord(path):
    """Exact but slow (builds a frame index). Only used as a fallback."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
        num_frames = len(vr)
        fps = float(vr.get_avg_fps()) or 0.0
        if num_frames <= 0 or fps <= 0:
            return None
        return num_frames, fps, num_frames / fps
    except Exception:
        return None


def probe(path, allow_decord=True):
    if not os.path.exists(path):
        return path, None
    result = probe_ffprobe(path)
    if result is None and allow_decord:
        result = probe_decord(path)
    return path, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", required=True, help="tag written to every row, e.g. ego4d / short")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-decord-fallback", action="store_true")
    args = parser.parse_args()

    paths = []
    with open(args.video_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row["path"].strip()
            if path:
                paths.append(path)
    print(f"probing {len(paths)} videos with {args.workers} workers...")

    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(probe, p, not args.no_decord_fallback) for p in paths
        ]
        for i, future in enumerate(as_completed(futures), 1):
            path, result = future.result()
            if result is None:
                failed.append(path)
            else:
                num_frames, fps, duration = result
                rows.append(
                    {
                        "path": path,
                        "num_frames": num_frames,
                        "fps": round(fps, 6),
                        "duration": round(duration, 3),
                        "source": args.source,
                    }
                )
            if i % 500 == 0:
                print(f"  {i}/{len(paths)}  ok={len(rows)}  failed={len(failed)}")

    rows.sort(key=lambda r: r["path"])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "num_frames", "fps", "duration", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows -> {args.out}")
    if failed:
        fail_path = args.out + ".failed.txt"
        with open(fail_path, "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        print(f"{len(failed)} unreadable videos listed in {fail_path}")


if __name__ == "__main__":
    main()