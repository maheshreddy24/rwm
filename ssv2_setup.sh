#!/bin/bash
set -e

BASE=https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2

mkdir -p evals/datasets/ssv2
cd evals/datasets/ssv2

echo "Downloading video parts..."
wget -c $BASE/20bn-something-something-v2-00
wget -c $BASE/20bn-something-something-v2-01

echo "Extracting videos..."
cat 20bn-something-something-v2-00 20bn-something-something-v2-01 | tar -xf -
rm -f 20bn-something-something-v2-00 20bn-something-something-v2-01

echo "Downloading labels..."
wget -c https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip

echo "Extracting labels..."
unzip -o 20bn-something-something-download-package-labels.zip
rm -f 20bn-something-something-download-package-labels.zip

echo "Done."