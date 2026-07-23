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

# conda's cudatoolkit only ships the runtime shared libs (libcudart.so etc.),
# not the nvcc compiler or the CUDA headers (cuda_runtime.h, cusparse.h, ...)
# needed to build the CUDA extensions below. Install a matching nvcc + dev
# headers and point CUDA_HOME at this env so torch's cpp_extension finds
# them, unless a system CUDA toolkit is already on PATH.
if ! command -v nvcc &> /dev/null; then
    # Runtime libs are pinned explicitly: without the pins conda resolves them to
    # CUDA 12-era builds that require libnvJitLink.so.12 and break torch's
    # cusolver ops (e.g. tensor.inverse()) at runtime.
    conda install -y -c nvidia cuda-nvcc=11.6 cuda-cudart-dev=11.6 \
        cuda-nvrtc-dev=11.6 libcusparse-dev=11.7.2.124 libcublas-dev=11.9.2.110 \
        libcusolver-dev=11.3.4.124 libcurand-dev=10.2.9.124 libcufft-dev=10.7.2.124 \
        cuda-cudart=11.6.55 cuda-cccl=11.6.55 libcublas=11.9.2.110 \
        libcusolver=11.3.4.124 libcusparse=11.7.2.124 libcufft=10.7.2.124 \
        libcurand=10.2.9.124
fi
export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"

# ninja parallelizes the CUDA extension builds (mmcv alone has ~150 .cu files).
pip install ninja

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
pip install tqdm
pip install plyfile
pip install "torchmetrics<1.0"  # >=1.0 drops python 3.7
pip install lpips
pip install scikit-image