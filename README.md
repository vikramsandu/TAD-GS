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

> **Code coming soon.** We are cleaning up the codebase and will release it here shortly.

## Contents

1. [Setup](#setup)
2. [Preprocess Datasets](#preprocess-datasets)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Pretrained Models](#pretrained-models)
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

### Neural 3D Video

Download the dataset from the official [`facebookresearch/Neural_3D_Video`](https://github.com/facebookresearch/Neural_3D_Video) repo.

### Interdigital

This dataset is not publicly downloadable — request access by emailing the
authors. See the [Interdigital Light Field Dataset](https://www.interdigital.com/data_sets/light-field-dataset)
page for details and contact information.

### VRU Basketball

Download the dataset from the [`BestWJH/VRU_Basketball`](https://huggingface.co/datasets/BestWJH/VRU_Basketball/tree/main) Hugging Face repo.

## Training

## Evaluation

## Pretrained Models

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

## Acknowledgements

This codebase builds on the environment setup and Gaussian Splatting
submodules (`submodules/gaussian_rasterization_ch3`, `submodules/simple-knn`)
from [SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians),
which are themselves derived from [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
(Inria / MPII). We also use [MMCV](https://github.com/open-mmlab/mmcv) for
CUDA-accelerated KNN during point cloud initialization. We thank the authors
for open-sourcing their work.
