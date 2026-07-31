#!/bin/bash
set -e

mkdir -p datasets/ssv2
cd datasets/ssv2

echo "Downloading dataset..."

wget -c https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-00
wget -c https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-v2-01
wget -c https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Something-Something-V2/20bn-something-something-download-package-labels.zip

echo "Combining split archive..."
cat 20bn-something-something-v2-00 \
    20bn-something-something-v2-01 \
    > 20bn-something-something-v2.tar

echo "Extracting dataset..."
tar -xf 20bn-something-something-v2.tar

echo "Extracting labels..."
unzip -o 20bn-something-something-download-package-labels.zip

echo "Done."