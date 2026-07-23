#!/bin/bash
# Shree KRISHNAya Namaha

########################################### Baseline #######################################################
# 1. Painter
CONFIG_PATH=vs_utils/configs/ID/Painter.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/baseline/Painter

SRC_PATH=data/InterDigital/Dense/Painter/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Painter/valid_masks
python train.py --eval --loader interdigital --config "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

# 2. Remy
CONFIG_PATH=vs_utils/configs/ID/Remy.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/baseline/Remy

SRC_PATH=data/InterDigital/Dense/Remy/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Remy/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

# 3. Theater
CONFIG_PATH=vs_utils/configs/ID/Theater.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/baseline/Theater

SRC_PATH=data/InterDigital/Dense/Theater/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Theater/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


# 4. Train
CONFIG_PATH=vs_utils/configs/ID/Train.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/baseline/Train

SRC_PATH=data/InterDigital/Dense/Train/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Train/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


# 5. Birthday
CONFIG_PATH=vs_utils/configs/ID/Birthday.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/baseline/Birthday

SRC_PATH=data/InterDigital/Dense/Birthday/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Birthday/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


########################################## 2. Opacity Weighting #######################################################

# 1. Painter
CONFIG_PATH=vs_utils/configs/ID/Painter.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight/Painter

SRC_PATH=data/InterDigital/Dense/Painter/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Painter/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

# 2. Remy
CONFIG_PATH=vs_utils/configs/ID/Remy.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight/Remy

SRC_PATH=data/InterDigital/Dense/Remy/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Remy/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

# 3. Theater
CONFIG_PATH=vs_utils/configs/ID/Theater.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight/Theater

SRC_PATH=data/InterDigital/Dense/Theater/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Theater/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


# 4. Train
CONFIG_PATH=vs_utils/configs/ID/Train.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight/Train

SRC_PATH=data/InterDigital/Dense/Train/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Train/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


# 5. Birthday
CONFIG_PATH=vs_utils/configs/ID/Birthday.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight/Birthday

SRC_PATH=data/InterDigital/Dense/Birthday/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Birthday/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


########################################## 3. Opacity Weighting + Adaptive thresholding #######################################################

# 1. Painter
CONFIG_PATH=vs_utils/configs/ID/Painter.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight_ada_thresh/Painter

SRC_PATH=data/InterDigital/Dense/Painter/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Painter/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

# 2. Remy
CONFIG_PATH=vs_utils/configs/ID/Remy.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight_ada_thresh/Remy

SRC_PATH=data/InterDigital/Dense/Remy/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Remy/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

# 3. Theater
CONFIG_PATH=vs_utils/configs/ID/Theater.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight_ada_thresh/Theater

SRC_PATH=data/InterDigital/Dense/Theater/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Theater/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


# 4. Train
CONFIG_PATH=vs_utils/configs/ID/Train.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight_ada_thresh/Train

SRC_PATH=data/InterDigital/Dense/Train/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Train/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


# 5. Birthday
CONFIG_PATH=vs_utils/configs/ID/Birthday.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight_ada_thresh/Birthday

SRC_PATH=data/InterDigital/Dense/Birthday/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Birthday/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


########################################## 4. Opacity Weighting + Adaptive thresholding + Offset warping #######################################################

# 1. Painter
CONFIG_PATH=vs_utils/configs/ID/Painter.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight_ada_thresh_time_warp/Painter

SRC_PATH=data/InterDigital/Dense/Painter/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Painter/valid_masks
python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


# 2. Remy
CONFIG_PATH=vs_utils/configs/ID/Remy.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight_ada_thresh_time_warp/Remy

SRC_PATH=data/InterDigital/Dense/Remy/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Remy/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"

# 3. Theater
CONFIG_PATH=vs_utils/configs/ID/Theater.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight_ada_thresh_time_warp/Theater

SRC_PATH=data/InterDigital/Dense/Theater/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Theater/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


# 4. Train
CONFIG_PATH=vs_utils/configs/ID/Train.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight_ada_thresh_time_warp/Train

SRC_PATH=data/InterDigital/Dense/Train/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Train/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"


# 5. Birthday
CONFIG_PATH=vs_utils/configs/ID/Birthday.json
MODEL_PATH=experiments/CVPR26/Oct11/ID/Sparse/d_opa_weight_ada_thresh_time_warp/Birthday

SRC_PATH=data/InterDigital/Dense/Birthday/colmap_0
FLOW_PATH=data/InterDigital/flow_masks/FEL001_FV01/Birthday/valid_masks

python train.py --eval --loader interdigital --config "$CONFIG_PATH" --opacity_weighting --is_adaptive --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
python render.py --quiet --eval --skip_train --valloader interdigital --configpath "$CONFIG_PATH" --use_offset_warping --model_path "$MODEL_PATH" --source_path "$SRC_PATH" --flow_masks_path "$FLOW_PATH"
