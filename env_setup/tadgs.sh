#!/bin/bash
#
# Sets up the "tadgs" conda environment used to train/evaluate TAD-GS.
#
# Rasterizer (submodules/gaussian_rasterization_ch3) and simple-knn
# (submodules/simple-knn) are borrowed from SpacetimeGaussians:
# https://github.com/oppo-us-research/SpacetimeGaussians/blob/main/script/setup.sh
#
# Submodules live under submodules/ at the repo root. If they're empty, run:
#   git submodule update --init --recursive

set -e

# Source conda's shell hook directly so `conda activate` works even when this
# script is run non-interactively (e.g. `bash tadgs.sh`), which skips ~/.bashrc.
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Conda environment ---
conda create -y -n tadgs python=3.7.13
conda activate tadgs

# Installed one by one instead of via environment.yml, which tends to hang.
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

# --- Gaussian Splatting CUDA extensions ---

# Gaussian Rasterization (Ch3) - Ours-Lite
pip install submodules/gaussian_rasterization_ch3

# simple-knn
pip install submodules/simple-knn

# opencv-python-headless, to work with colmap on server
pip install opencv-python

# MMCV for CUDA KNN, used for init point sampling, to reduce number of points
# when SfM points are too many. Takes ~30min to build.
pip install -e submodules/mmcv -v

# --- Misc packages ---
pip install natsort
pip install scipy
pip install kornia