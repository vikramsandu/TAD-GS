# Preprocessing: Neural 3D Video

Turns the raw Neural 3D Video (DyNeRF) scene archives into training-ready data
in the **same format as [SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians)**:
a `colmap_<t>/` folder per timestamp (each with `images/` + `manual/`), with
COLMAP point clouds reconstructed on every Nth frame.

Requires the `colmap_env` conda environment (see `env_setup/colmap.sh`).

## Usage

Place the scene archives of the 6 Neural 3D Video scenes — `coffee_martini`,
`cook_spinach`, `cut_roasted_beef`, `flame_salmon_1`, `flame_steak`,
`sear_steak` — in `data/neural_3d_video/`. The pipeline extracts each archive
into `data/neural_3d_video/scenes/<scene>/`, keeping the `.zip` archives
themselves at the dataset root:

```
data/neural_3d_video/
├── coffee_martini.zip …            # the 6 scene archives
└── scenes/
    ├── coffee_martini/ … sear_steak/   # extracted + preprocessed scenes
```

`flame_salmon_1` ships as a 4-part split zip (`flame_salmon_1_split.z01`,
`.z02`, `.z03`, `.zip`) because of its size. Join it into a regular zip once
before running the pipeline (requires Info-Zip's `zip`, `sudo apt install
zip` if missing):

```shell
cd data/neural_3d_video
zip -s 0 flame_salmon_1_split.zip --out flame_salmon_1.zip
cd -
```

Then, from the repo root:

```shell
bash utils/preprocessing/neural_3d_video/preprocess_n3d.sh
```

`preprocess_n3d.sh` runs one explicit `pre_n3d.py` invocation per scene (no
loop), so each scene's pointcloud type is visible at a glance. Edit the
hyperparameters at the top of the script, or the `--pointcloud
sparse`/`dense` flag on individual scene invocations, as needed.

| Hyperparameter | Default | Meaning |
|---|---|---|
| `START_FRAME` / `END_FRAME` | `0` / `300` | frame range to extract from the videos |
| `COLMAP_FRAME_INTERVAL` | `20` | run COLMAP on every Nth frame plus the final frame (offsets 0, 20, …, 280, 299) |
| `DOWNSCALE` | `2` | image downscale factor; 2 → **1352×1014** for N3D |
| `--pointcloud` | `dense` for `coffee_martini`/`flame_salmon_1`, `sparse` for the rest | point cloud type per scene invocation |

> **Sparse vs dense initialization.** We observe that two Neural 3D Video
> scenes, *coffee martini* and *flame salmon*, are particularly sensitive to
> initialization due to their unbounded nature, requiring accurate depth
> initialization for distant static background regions. To address this, we
> use a dense COLMAP initialization (MVS + downsampling) for these two scenes
> on **every COLMAP offset** (0, 20, …, 280, 299), while the remaining four
> scenes use sparse (SfM triangulation) initializations on every offset.
>
> Dense MVS is heavy, so it runs once per offset as an isolated COLMAP
> subprocess — GPU memory is released between offsets, and the bulky MVS
> intermediates (per-view depth/normal maps, undistorted image copies, the
> full `fused.ply`) are deleted right after downsampling, keeping only
> `fused_downsample.ply` so the disk footprint stays bounded across the
> sequence. The step is resumable: an offset whose `fused_downsample.ply`
> already exists is skipped without re-running MVS. At training time the
> loader auto-detects, per anchor frame, whether a dense cloud exists and
> initializes from it, falling back to the sparse cloud otherwise.

> **Missing cameras.** Some scenes ship fewer than 21 cameras — `coffee_martini`
> has 18 (missing cam03, cam15, cam17) and `cut_roasted_beef` has 20 (missing
> cam04). `poses_bounds.npy` only has rows for the cameras actually present,
> in the same sorted order as the `cam*.mp4` files, so pose lookup by sorted
> position is always correct. The pipeline renumbers the present cameras
> sequentially (`cam00, cam01, …`) by that sorted position for every output
> path (`images/`, `manual/`, COLMAP database), so a scene missing e.g. cam04
> still produces a gap-free `cam00 … cam19` range rather than skipping
> straight from `cam03` to `cam05`.

## Pipeline steps and outputs

Each scene goes through five steps (`pre_n3d.py`). Everything is resumable —
already-extracted frames and already-reconstructed offsets are skipped.

### 1. Unzip

`data/neural_3d_video/cook_spinach.zip` → the raw scene under `scenes/`:

```
data/neural_3d_video/scenes/cook_spinach/
├── cam00.mp4 … cam20.mp4     # 21 synchronized multi-view videos
└── poses_bounds.npy          # LLFF-style camera poses + bounds
```

### 2. Extract frames (temporary)

Every video is decoded to PNG frames (downscaled to 1352×1014 by default) into
per-camera folders `cam00/ … cam20/`. These folders are **temporary** and are
removed in step 5 once every frame has been moved to its `colmap_<t>/` folder.

### 3. Run COLMAP on every Nth frame

For each offset `t` in {0, 20, …, 280} — plus the final frame 299, which is
always included even when not divisible by the interval — the known camera poses from
`poses_bounds.npy` are converted into a COLMAP database (`input.db`) + text
model (`manual/`), frame `t` of every camera is provided as input, and COLMAP
extracts/matches SIFT features and triangulates a sparse point cloud with the
poses fixed (`point_triangulator`), then undistorts the images:

```
scenes/cook_spinach/colmap_0/
├── images/                   # undistorted images, one per camera
├── manual/                   # known-pose model: cameras.txt, images.txt, points3D.txt
├── sparse/0/                 # cameras.bin, images.bin, points3D.bin  <- initial point cloud
├── input.db                  # COLMAP database
└── distorted/, stereo/, …    # COLMAP byproducts
```

With `--pointcloud dense`, COLMAP MVS additionally runs on **every offset**
(`patch_match_stereo` + `stereo_fusion`), and each fused cloud is
voxel-downsampled to ≤ 40k points as in
[Swift4D](https://github.com/WuJH2001/Swift4d/blob/main/scripts/downsample_point.py).
The bulky MVS intermediates are removed right after downsampling, leaving only
the downsampled cloud per offset:

```
scenes/coffee_martini/colmap_0/dense/workspace/
└── fused_downsample.ply      # voxel-downsampled to <= 40k points (kept)
scenes/coffee_martini/colmap_20/dense/workspace/
└── fused_downsample.ply
…                             # one per COLMAP offset (0, 20, …, 280, 299)
```

### 4. Distribute all remaining timestamps

Every other timestamp gets the **same structure** (`images/` + `manual/`),
with the extracted frames moved in as `images/camXX.png` under their
sequentially-renumbered name (see "Missing cameras" above):

```
scenes/cook_spinach/colmap_1/
├── images/                   # cam00.png … cam20.png (raw frames at this timestamp)
└── manual/                   # same known-pose model
```

### 5. Cleanup

After verifying that every `colmap_<t>/images/` holds every camera present in
the scene, the temporary per-camera frame folders are deleted. Final scene
layout:

```
scenes/cook_spinach/
├── cam00.mp4 … cam20.mp4, poses_bounds.npy    # originals
├── colmap_0/                                  # COLMAP offsets: images/ manual/ sparse/0/ [dense/]
├── colmap_1/ … colmap_19/                     # other timestamps: images/ manual/
├── colmap_20/                                 # COLMAP offset
│   …
└── colmap_299/
```

## Files

| File | Role | Adapted from |
|---|---|---|
| `preprocess_n3d.sh` | batch driver over scenes | — |
| `pre_n3d.py` | unzip, frame extraction, COLMAP input prep, distribution, cleanup | STG `script/pre_n3d.py`, `script/utils_pre.py` |
| `colmap_runner.py` | sparse + dense COLMAP pipelines | STG `helper3dg.py`, Swift4D `colmap.sh` |
| `colmap_database.py` | COLMAP sqlite database writer | STG `thirdparty/colmap/pre_colmap.py` ← COLMAP scripts |
| `pose_utils.py` | LLFF pose → COLMAP w2c/quaternion conversion | STG `utils/my_utils.py` |
| `downsample_point.py` | voxel-downsample dense point clouds | Swift4D `scripts/downsample_point.py` |

## Acknowledgements

This pipeline is adapted from
[SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians) (MIT, OPPO)
and [Swift4D](https://github.com/WuJH2001/Swift4d), which in turn build on the official
[COLMAP](https://colmap.github.io/) scripts. We thank the authors for open-sourcing their work.
