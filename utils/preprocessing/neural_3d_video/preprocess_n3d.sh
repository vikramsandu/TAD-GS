#!/bin/bash
#
# Preprocess all 6 Neural 3D Video scenes for TAD-GS: unzip -> extract frames ->
# COLMAP point clouds on every Nth frame. Edit the hyperparameters below.
#
# One explicit python invocation per scene (no loop) so each scene's
# pointcloud type and any per-scene overrides are visible at a glance.
#
# Adapted from SpacetimeGaussians (https://github.com/oppo-us-research/SpacetimeGaussians)
# and Swift4D (https://github.com/WuJH2001/Swift4d).
#
# Run from the repo root:  bash utils/preprocessing/neural_3d_video/preprocess_n3d.sh

set -e

# ----------------------------- Hyperparameters ------------------------------
DATA_ROOT="data/neural_3d_video"          # folder holding the <scene>.zip archives
SCENES_ROOT="$DATA_ROOT/scenes"           # extracted scenes go here: scenes/<scene>/
START_FRAME=0                          # first frame to extract (inclusive)
END_FRAME=300                          # last frame to extract (exclusive)
COLMAP_FRAME_INTERVAL=20               # run COLMAP on every Nth frame
DOWNSCALE=2                            # image downscale factor (2 -> 1352x1014)

# coffee_martini and flame_salmon_1 are particularly sensitive to initialization
# due to their unbounded nature, requiring accurate depth initialization for
# distant static background regions -> use dense COLMAP (MVS) for them, but
# only on offset 0 (pre_n3d.py runs dense MVS on frame 0 only; every other
# COLMAP offset stays sparse, even for these two scenes). All other scenes use
# sparse (SfM triangulation) point clouds on every offset.
# -----------------------------------------------------------------------------

# Source conda's shell hook directly so `conda activate` works even when this
# script is run non-interactively (e.g. `bash preprocess_n3d.sh`), which skips ~/.bashrc.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate colmap_env

echo "=============================================================="
echo " Preprocessing scene: coffee_martini (pointcloud: dense, frame 0 only)"
echo "=============================================================="
python utils/preprocessing/neural_3d_video/pre_n3d.py \
    --videopath "$SCENES_ROOT/coffee_martini" \
    --startframe "$START_FRAME" \
    --endframe "$END_FRAME" \
    --frame_interval "$COLMAP_FRAME_INTERVAL" \
    --downscale "$DOWNSCALE" \
    --pointcloud dense

echo "=============================================================="
echo " Preprocessing scene: cook_spinach (pointcloud: sparse)"
echo "=============================================================="
python utils/preprocessing/neural_3d_video/pre_n3d.py \
    --videopath "$SCENES_ROOT/cook_spinach" \
    --startframe "$START_FRAME" \
    --endframe "$END_FRAME" \
    --frame_interval "$COLMAP_FRAME_INTERVAL" \
    --downscale "$DOWNSCALE" \
    --pointcloud sparse

echo "=============================================================="
echo " Preprocessing scene: cut_roasted_beef (pointcloud: sparse)"
echo "=============================================================="
python utils/preprocessing/neural_3d_video/pre_n3d.py \
    --videopath "$SCENES_ROOT/cut_roasted_beef" \
    --startframe "$START_FRAME" \
    --endframe "$END_FRAME" \
    --frame_interval "$COLMAP_FRAME_INTERVAL" \
    --downscale "$DOWNSCALE" \
    --pointcloud sparse

echo "=============================================================="
echo " Preprocessing scene: flame_salmon_1 (pointcloud: dense, frame 0 only)"
echo "=============================================================="
python utils/preprocessing/neural_3d_video/pre_n3d.py \
    --videopath "$SCENES_ROOT/flame_salmon_1" \
    --startframe "$START_FRAME" \
    --endframe "$END_FRAME" \
    --frame_interval "$COLMAP_FRAME_INTERVAL" \
    --downscale "$DOWNSCALE" \
    --pointcloud dense

echo "=============================================================="
echo " Preprocessing scene: flame_steak (pointcloud: sparse)"
echo "=============================================================="
python utils/preprocessing/neural_3d_video/pre_n3d.py \
    --videopath "$SCENES_ROOT/flame_steak" \
    --startframe "$START_FRAME" \
    --endframe "$END_FRAME" \
    --frame_interval "$COLMAP_FRAME_INTERVAL" \
    --downscale "$DOWNSCALE" \
    --pointcloud sparse

echo "=============================================================="
echo " Preprocessing scene: sear_steak (pointcloud: sparse)"
echo "=============================================================="
python utils/preprocessing/neural_3d_video/pre_n3d.py \
    --videopath "$SCENES_ROOT/sear_steak" \
    --startframe "$START_FRAME" \
    --endframe "$END_FRAME" \
    --frame_interval "$COLMAP_FRAME_INTERVAL" \
    --downscale "$DOWNSCALE" \
    --pointcloud sparse

echo "All scenes preprocessed."
