import argparse
import concurrent.futures
import csv
import os
import random
import subprocess
import tarfile
import zipfile
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

URL_LIST_PATH = os.path.join(BASE_DIR, "k600_train_path.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "kinetics_dataset")
CSV_PATH = os.path.join(BASE_DIR, "kinetics_train_set_clip_paths.csv")


def get_class_name(filename):
    """Extract class name from archive filename."""
    if filename.endswith(".tar.gz"):
        return filename[:-7]
    elif filename.endswith(".tgz"):
        return filename[:-4]
    elif filename.endswith(".zip"):
        return filename[:-4]
    else:
        return Path(filename).stem


def extract_archive(archive_path, extract_dir):
    """Extract .zip, .tar, .tar.gz, or .tgz archives."""
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)

    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)

    elif archive_path.endswith(".tar"):
        with tarfile.open(archive_path, "r") as tf:
            tf.extractall(extract_dir)

    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def download_and_extract(url):
    filename = os.path.basename(url)
    class_name = get_class_name(filename)

    archive_path = os.path.join(OUTPUT_DIR, filename)
    extract_dir = os.path.join(OUTPUT_DIR, class_name)

    os.makedirs(extract_dir, exist_ok=True)

    print(f"Downloading {class_name}...")

    # Download
    subprocess.run(
        [
            "curl",
            "-L",
            "-f",
            "--retry",
            "3",
            "-o",
            archive_path,
            url,
        ],
        check=True,
    )

    print(f"Extracting {class_name}...")

    # Extract archive
    extract_archive(archive_path, extract_dir)

    # Remove downloaded archive
    os.remove(archive_path)

    # Collect all clip paths
    clip_paths = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                clip_paths.append(os.path.join(root, file))

    return class_name, clip_paths


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract Kinetics-600 classes."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=590,
        help="Number of random classes to download.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of parallel downloads.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(URL_LIST_PATH, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    random.seed(args.seed)
    random.shuffle(urls)

    selected_urls = urls[: args.num_classes]

    print(f"Downloading {len(selected_urls)} classes...\n")

    all_clip_paths = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:

        future_to_url = {
            executor.submit(download_and_extract, url): url
            for url in selected_urls
        }

        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                class_name, clip_paths = future.result()
                all_clip_paths.extend(clip_paths)
                print(f"✓ {class_name}: {len(clip_paths)} clips")
            except Exception as e:
                print(f"✗ Failed: {url}")
                print(f"  {e}")

    all_clip_paths.sort()

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_path"])
        for path in all_clip_paths:
            writer.writerow([path])

    print("\nDone!")
    print(f"Downloaded clips: {len(all_clip_paths)}")
    print(f"CSV saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()