import argparse
import concurrent.futures
import csv
import os
import random
import subprocess
import zipfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

URL_LIST_PATH = os.path.join(BASE_DIR, "k600_train_path.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "kinetics_dataset")
CSV_PATH = os.path.join(BASE_DIR, "kinetics_train_set_clip_paths.csv")


def download_and_extract(url):
    filename = os.path.basename(url)
    class_name = os.path.splitext(filename)[0]

    zip_path = os.path.join(OUTPUT_DIR, filename)
    extract_dir = os.path.join(OUTPUT_DIR, class_name)

    os.makedirs(extract_dir, exist_ok=True)

    # Download
    subprocess.run(
        ["curl", "-L", "-f", "--retry", "3", "-o", zip_path, url],
        check=True,
    )

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # Delete zip
    os.remove(zip_path)

    clip_paths = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            clip_paths.append(os.path.join(root, file))

    return class_name, clip_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(URL_LIST_PATH, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    random.seed(args.seed)
    random.shuffle(urls)

    selected_urls = urls[: args.num_classes]

    all_clip_paths = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = {
            executor.submit(download_and_extract, url): url
            for url in selected_urls
        }

        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                class_name, clip_paths = future.result()
                all_clip_paths.extend(clip_paths)
                print(f"✓ {class_name}: {len(clip_paths)} clips")
            except Exception as e:
                print(f"✗ Failed {url}")
                print(e)

    all_clip_paths.sort()

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_path"])
        for path in all_clip_paths:
            writer.writerow([path])

    print(f"\nSaved {len(all_clip_paths)} clip paths to")
    print(CSV_PATH)


if __name__ == "__main__":
    main()