#!/bin/bash
#
# Borrowed from SpacetimeGaussians:
# https://github.com/oppo-us-research/SpacetimeGaussians/blob/main/script/setup.sh
#
# Usage: sets up a standalone "colmap_env" conda environment used to run COLMAP
# on the input multi-view video frames and generate the initial sparse point
# cloud (camera poses + 3D points) used to initialize training of the TADGS model.

set -e

# Source conda's shell hook directly so `conda activate` works even when this
# script is run non-interactively (e.g. `bash colmap.sh`), which skips ~/.bashrc.
source "$(conda info --base)/etc/profile.d/conda.sh"

# install colmap for preprocess, work with python3.8
conda create -y -n colmap_env python=3.8
conda activate colmap_env
pip install opencv-python-headless
pip install tqdm
pip install natsort
pip install Pillow
conda install -y pytorch==1.12.1 -c pytorch -c conda-forge
conda config --set channel_priority false
conda install -y colmap -c conda-forge