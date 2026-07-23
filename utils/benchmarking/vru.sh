#!/bin/bash
# Shree KRISHNAya Namaha

########################################### Baseline #######################################################
# 1. DG
#CONFIG_PATH=vs_utils/configs/vru/_c_sparse.json
#MODEL_PATH=experiments/CVPR26/Oct11/VRU-250/Sparse/baseline/dg
#
#SRC_PATH=VRU_STG/Basketball_dg/colmap_0
#FLOW_PATH=VRU_STG/flow_masks/FEL001_FV01/DG/valid_masks
#
#python train.py --eval --loader vru --config "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
#python render.py --quiet --eval --skip_train --valloader vru --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
#
## 2. GZ
#CONFIG_PATH=vs_utils/configs/vru/_c_sparse.json
#MODEL_PATH=experiments/CVPR26/Oct11/VRU-250/Sparse/baseline/gz
#
#SRC_PATH=VRU_STG/Basketball_gz/colmap_0
#FLOW_PATH=VRU_STG/flow_masks/FEL001_FV01/GZ/valid_masks
#
#python train.py --eval --loader vru --config "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
#python render.py --quiet --eval --skip_train --valloader vru --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
#
########################################## 2. Opacity Weighting #######################################################
#
## 1. DG
#CONFIG_PATH=vs_utils/configs/vru/_c_sparse.json
#MODEL_PATH=experiments/CVPR26/Oct11/VRU-250/Sparse/d_opa_weight/dg
#
#SRC_PATH=VRU_STG/Basketball_dg/colmap_0
#FLOW_PATH=VRU_STG/flow_masks/FEL001_FV01/DG/valid_masks
#
#python train.py --eval --loader vru --config "$CONFIG_PATH" --opacity_weighting --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
#python render.py --quiet --eval --skip_train --valloader vru --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
#
## 2. GZ
#CONFIG_PATH=vs_utils/configs/vru/_c_sparse.json
#MODEL_PATH=experiments/CVPR26/Oct11/VRU-250/Sparse/d_opa_weight/gz
#
#SRC_PATH=VRU_STG/Basketball_gz/colmap_0
#FLOW_PATH=VRU_STG/flow_masks/FEL001_FV01/GZ/valid_masks
#
#python train.py --eval --loader vru --config "$CONFIG_PATH" --opacity_weighting --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
#python render.py --quiet --eval --skip_train --valloader vru --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

########################################## 3. Opacity Weighting + Adaptive thresholding #######################################################

# 1. DG
CONFIG_PATH=vs_utils/configs/vru/_c_sparse.json
MODEL_PATH=experiments/CVPR26/Oct11/VRU-250/Sparse/d_opa_weight_ada_thresh/dg

SRC_PATH=VRU_STG/Basketball_dg/colmap_0
FLOW_PATH=VRU_STG/flow_masks/FEL001_FV01/DG/valid_masks

python train.py --eval --loader vru --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader vru --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

# 2. GZ
CONFIG_PATH=vs_utils/configs/vru/_c_sparse.json
MODEL_PATH=experiments/CVPR26/Oct11/VRU-250/Sparse/d_opa_weight_ada_thresh/gz

SRC_PATH=VRU_STG/Basketball_gz/colmap_0
FLOW_PATH=VRU_STG/flow_masks/FEL001_FV01/GZ/valid_masks

python train.py --eval --loader vru --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader vru --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

########################################## 4. Opacity Weighting + Adaptive thresholding + Offset warping #######################################################

# 1. DG
CONFIG_PATH=vs_utils/configs/vru/_c_sparse.json
MODEL_PATH=experiments/CVPR26/Oct11/VRU-250/Sparse/d_opa_weight_ada_thresh_time_warp/dg

SRC_PATH=VRU_STG/Basketball_dg/colmap_0
FLOW_PATH=VRU_STG/flow_masks/FEL001_FV01/DG/valid_masks

python train.py --eval --loader vru --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader vru --configpath "$CONFIG_PATH" --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

# 2. GZ
CONFIG_PATH=vs_utils/configs/vru/_c_sparse.json
MODEL_PATH=experiments/CVPR26/Oct11/VRU-250/Sparse/d_opa_weight_ada_thresh_time_warp/gz

SRC_PATH=VRU_STG/Basketball_gz/colmap_0
FLOW_PATH=VRU_STG/flow_masks/FEL001_FV01/GZ/valid_masks

python train.py --eval --loader vru --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader vru --configpath "$CONFIG_PATH" --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
