#!/bin/bash
# Shree KRISHNAya Namaha
#
# Benchmark TAD-GS on all 5 Interdigital scenes with the full method
# (VAD + TAT + TOW): train -> render -> mp4 of the rendered frames ->
# per-scene and consolidated results.md.
#
# Outputs go to runs/<dataset_name>/<run_id>/<scene>/ where <run_id>
# auto-increments (0 for the first run, 1 for the second, ...).
#
# Run from the repo root:  bash utils/benchmarking/interdigital.sh

set -e

# ----------------------------- Settings ------------------------------
DATASET_NAME="interdigital"
DATA_ROOT="data/interdigital"
CONFIG_DIR="arguments/interdigital"        # one <scene>.json per scene
FLOW_ROOT="data/interdigital/flow_masks"   # per-scene dynamic-region masks (masked metrics only)
FPS=30                       # frame rate of the rendered mp4
# ----------------------------------------------------------------------

# Source conda's shell hook directly so `conda activate` works even when this
# script is run non-interactively (e.g. `bash interdigital.sh`), which skips ~/.bashrc.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate tadgs

# Auto-incrementing run id: runs/<dataset>/0, runs/<dataset>/1, ...
mkdir -p "runs/$DATASET_NAME"
RUN_ID=0
while [ -d "runs/$DATASET_NAME/$RUN_ID" ]; do RUN_ID=$((RUN_ID + 1)); done
RUN_DIR="runs/$DATASET_NAME/$RUN_ID"
mkdir -p "$RUN_DIR"
echo "=============================================================="
echo " Benchmark run: $RUN_DIR (VAD + TAT + TOW)"
echo "=============================================================="

run_scene () {
    local scene=$1
    local model_path="$RUN_DIR/$scene"
    local source_path="$DATA_ROOT/scenes/$scene/colmap_0"
    local flow_masks_path="$FLOW_ROOT/$scene/valid_masks"
    local renders_dir="$model_path/test/ours_best/renders"
    local config="$CONFIG_DIR/$scene.json"   # per-scene config (pcd type + init pcd downsample)

    echo "=============================================================="
    echo " Scene: $scene"
    echo "=============================================================="

    # Train (full method: visibility aware densification, temporally
    # adaptive thresholding, temporal offset warping). Flow masks are used
    # only to compute the masked eval metrics, not for training.
    python train.py --eval --loader interdigital --configpath "$config" --vad --tat --tow \
        --model_path "$model_path" --source_path "$source_path" \
        --flow_masks_path "$flow_masks_path"

    # Render the test view with the best checkpoint (TOW affects the
    # forward pass, so it must be passed at render time too)
    python render.py --eval --skip_train --valloader interdigital --configpath "$config" --tow \
        --model_path "$model_path" --source_path "$source_path" \
        --flow_masks_path "$flow_masks_path"

    # MP4 from the rendered frames
    ffmpeg -y -loglevel error -framerate "$FPS" -i "$renders_dir/%05d.png" \
        -c:v libx264 -pix_fmt yuv420p "$model_path/test/ours_best/renders.mp4"
    echo "Saved video: $model_path/test/ours_best/renders.mp4"

    # Refresh per-scene + consolidated results.md (idempotent, so partial
    # runs already have up-to-date results)
    python utils/benchmarking/write_results.py "$RUN_DIR"
}

run_scene Painter
run_scene Remy
run_scene Theater
run_scene Train
run_scene Birthday

echo "=============================================================="
echo " Benchmark complete: $RUN_DIR/consolidated_results.md"
echo "=============================================================="
