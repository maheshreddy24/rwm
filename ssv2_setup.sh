#!/usr/bin/env bash

set -euo pipefail

BASE_URL="https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2"
LABEL_URL="https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip"

mkdir -p evals/datasets/ssv2
cd evals/datasets/ssv2

echo "Downloading video archive parts..."

for part in 00 01; do
    wget -c --content-disposition \
        "${BASE_URL}/20bn-something-something-v2-${part}"
done

echo "Extracting videos..."

cat 20bn-something-something-v2-00 \
    20bn-something-something-v2-01 | tar -xvf -

rm -f \
    20bn-something-something-v2-00 \
    20bn-something-something-v2-01

echo "Downloading labels..."

wget -c --content-disposition \
    -O labels.zip \
    "${LABEL_URL}"

echo "Extracting labels..."

unzip -o labels.zip

rm -f labels.zip

echo "Dataset downloaded successfully!"