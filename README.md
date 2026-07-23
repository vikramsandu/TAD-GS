<h2 align="center">TAD-GS: Temporally Aware Densification for Dynamic 3D Gaussian Splatting</h2>

<p align="center">
  <a href="https://vikramsandu.github.io/"><strong>Vikram Sandu</strong></a>
  ·
  <strong>Mayurdeep Pathak</strong>
  ·
  <strong>Rajiv Soundararajan</strong>
  <br>
  Indian Institute of Science, Bengaluru
  <br>
  ECCV 2026
</p>

<p align="center">
  <a href="https://vikramsandu.github.io/publications/TADGS/index.html"><strong><code>Project Page</code></strong></a>
  <a href="https://arxiv.org/abs/2606.23212"><strong><code>Arxiv Paper</code></strong></a>
  <a href="https://github.com/vikramsandu/TAD-GS"><strong><code>Source Code</code></strong></a>
</p>

<div align='center'>
  <br>
  <img src="assets/comparison.gif" width=90% alt="Side-by-side comparison of 3DGS Densification and TAD-GS on the cook_spinach scene">
  <br>Existing 3DGS densification fails to refine short-lived dynamic Gaussians, resulting in blurry reconstructions. TAD-GS recovers highly dynamic regions.
</div>

<br>

## Contents

1. [Setup](#setup)
2. [Preprocess Datasets](#preprocess-datasets)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Benchmarking](#benchmarking)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)

## Setup

> **Tested environment:** NVIDIA RTX A4000 (16GB) GPU, driver 545.29.06,
> CUDA 12.3, on Ubuntu 22.04.4 LTS (Linux).

### 1. Clone the source code of this repo

```shell
git clone --recursive https://github.com/vikramsandu/TAD-GS.git
cd TAD-GS
```

If you already cloned without `--recursive`, fetch the submodules with:

```shell
git submodule update --init --recursive
```

### 2. Setup the COLMAP environment (for preprocessing)

This creates a separate `colmap_env` conda environment used to run COLMAP on
the input multi-view video frames and generate the initial point cloud
for training.

```shell
bash env_setup/colmap.sh
```

### 3. Setup the TAD-GS environment (for training/evaluation)

This creates the `tadgs` conda environment with PyTorch, the Gaussian
Splatting CUDA rasterizer, and all other dependencies needed to train and
evaluate TAD-GS.

```shell
bash env_setup/tadgs.sh
```

## Preprocess Datasets

We evaluate TAD-GS on 3 datasets: [Neural 3D Video](#neural-3d-video), [Interdigital](#interdigital), and [VRU Basketball](#vru-basketball).

We provide the dynamic-region **flow masks** for test camera (used for the masked evaluation
metrics) for all datasets: download them from
[Google Drive](https://drive.google.com/file/d/1ei_iU0WfHGahP3BgrjNezF7dmhSx3VNl/view?usp=sharing)
and extract each dataset's masks into `data/<dataset>/flow_masks/`, so every
scene lives at `data/<dataset>/flow_masks/<scene>/valid_masks/`.

### Neural 3D Video

Download the dataset from the official [`facebookresearch/Neural_3D_Video`](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0) release
and place the scene archives (e.g. `cook_spinach.zip`) in `data/neural_3d_video/`. Then run the below preprocessing pipeline (uses the `colmap_env` environment from
[Setup](#setup))

```shell
bash utils/preprocessing/neural_3d_video/preprocess_n3d.sh
```
Refer to
[`utils/preprocessing/neural_3d_video/README.md`](utils/preprocessing/neural_3d_video/README.md)
for more details and the expected output layout.

### Interdigital

This dataset is not publicly downloadable — request access by emailing the
authors. See the [Interdigital Light Field Dataset](https://www.interdigital.com/data_sets/light-field-dataset)
page for details and contact information.

### VRU Basketball

Download the dataset from the [`BestWJH/VRU_Basketball`](https://huggingface.co/datasets/BestWJH/VRU_Basketball/tree/main) Hugging Face repo.

## Training

All training uses the `tadgs` conda environment from [Setup](#setup). TAD-GS
is trained with its three components enabled: **VAD** (visibility-aware
densification), **TAT** (temporally adaptive thresholding), and **TOW**
(temporal offset warping), passed as the `--vad --tat --tow` flags.

To train one scene:

```shell
python train.py --eval --vad --tat --tow \
    --configpath arguments/neural_3d_video/coffee_martini.json \
    --source_path data/neural_3d_video/scenes/coffee_martini/colmap_0 \
    --flow_masks_path data/neural_3d_video/flow_masks/coffee_martini/valid_masks \
    --model_path runs/neural_3d_video/0/coffee_martini
```

- `--configpath` — per-scene config json path in `arguments/<dataset>/<scene>.json`
- `--source_path` — the preprocessed `colmap_0` workspace of the scene.
- `--flow_masks_path` — per-scene dynamic-region masks; used **only** for the
  masked eval metrics, not for training.
- `--model_path` — output directory for checkpoints and logs.

## Evaluation

Rendering and metric computation use `render.py`. Since **TOW** affects the
forward pass, it must be passed at render time too. This renders the held-out
test view with the best checkpoint and writes the metrics:

```shell
python render.py --eval --skip_train --valloader colmap --tow \
    --configpath arguments/neural_3d_video/coffee_martini.json \
    --source_path data/neural_3d_video/scenes/coffee_martini/colmap_0 \
    --flow_masks_path data/neural_3d_video/flow_masks/coffee_martini/valid_masks \
    --model_path runs/neural_3d_video/0/coffee_martini
```

## Benchmarking

To reproduce the paper results end-to-end, use the benchmark script. It
trains → renders → encodes an mp4 → writes a consolidated
`consolidated_results.md` for **all 6 Neural 3D Video scenes**, so you don't
have to run the [Training](#training) and [Evaluation](#evaluation) steps by
hand.

```shell
bash utils/benchmarking/neural_3d_video.sh   # Neural 3D Video (all 6 scenes)
```

Outputs are written to `runs/neural_3d_video/<run_id>/<scene>/`, where
`<run_id>` auto-increments (`0` for the first run, `1` for the next, ...). Each
scene is trained with the full method (`--vad --tat --tow`) using its per-scene
config in `arguments/neural_3d_video/`. The consolidated table is written to
`runs/neural_3d_video/<run_id>/consolidated_results.md` (one row per scene plus
an average row):

| Scene | PSNR | SSIM | MS-SSIM | LPIPS-Alex | Masked-PSNR | Masked-SSIM | Train time |
|---|---|---|---|---|---|---|---|
| coffee_martini | 29.2530 | 0.9215 | 0.9583 | 0.1092 | 23.7699 | 0.8767 | 01:16:24 |
| cook_spinach | 33.4779 | 0.9567 | 0.9763 | 0.0857 | 24.4013 | 0.8322 | 01:11:02 |
| cut_roasted_beef | 33.7227 | 0.9584 | 0.9782 | 0.0855 | 27.5878 | 0.9017 | 01:13:10 |
| flame_salmon_1 | 29.9244 | 0.9259 | 0.9618 | 0.1019 | 21.1544 | 0.8195 | 01:27:41 |
| flame_steak | 34.1135 | 0.9637 | 0.9799 | 0.0770 | 23.3979 | 0.8502 | 01:09:37 |
| sear_steak | 34.1498 | 0.9649 | 0.9812 | 0.0768 | 27.2333 | 0.8946 | 01:13:00 |
| **Average** | **32.4402** | **0.9485** | **0.9726** | **0.0893** | **24.5908** | **0.8625** | **01:15:09** |

> **Note:** benchmarking for [Interdigital](#interdigital) and
> [VRU Basketball](#vru-basketball) is releasing soon.

## Citation

If you find this work useful, please cite:

```bibtex
@article{sandu2026tadgs,
  author    = {Sandu, Vikram and Pathak, Mayurdeep and Soundararajan, Rajiv},
  title     = {Temporally Aware Densification for Dynamic 3D Gaussian Splatting},
  journal   = {ECCV},
  year      = {2026},
}
```

And the 3D Gaussian Splatting paper this work builds on:

```bibtex
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## Acknowledgements

This codebase builds on the environment setup and Gaussian Splatting
submodules (`submodules/gaussian_rasterization_ch3`, `submodules/simple-knn`)
from [SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians),
which are themselves derived from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(Inria / MPII). We also use [MMCV](https://github.com/open-mmlab/mmcv) for
CUDA-accelerated KNN during point cloud initialization. We thank the authors
for open-sourcing their work.

We also thank [Claude Code](https://claude.com/claude-code) (Anthropic) for
help with documenting and refactoring the codebase for release.
