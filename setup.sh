lea#!/bin/bash

set -e

ENV_NAME="gssl"
PYTHON_VERSION="3.10"

echo "Creating conda environment: ${ENV_NAME}..."
conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
source /root/miniconda3/etc/profile.d/conda.sh 
conda activate "${ENV_NAME}"

echo "Installing PyTorch..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# echo "Installing Python requirements recursively..."

# find . -type f -name "requirements.txt" | while read -r req_file; do
#     echo "Installing from: ${req_file}"
#     pip install -r "${req_file}"
# done

echo "Done!"

conda install ipykernel
conda install -c conda-forge ffmpeg
echo "Environment '${ENV_NAME}' is ready."

sudo apt install nvtop
sudo apt install tmux 
sudo apt install htop



# mkdir -p rvm_ckpts
# cd rvm_ckpts

# wget https://storage.googleapis.com/representations4d/checkpoints/pretrain_rvm_small16_256_204031069.npz