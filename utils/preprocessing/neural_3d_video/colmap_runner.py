# COLMAP pipeline for a single frame-offset workspace (colmap_<offset>/).
#
# Sparse pipeline adapted from SpacetimeGaussians (MIT License, Copyright (c) 2023 OPPO):
# https://github.com/oppo-us-research/SpacetimeGaussians/blob/main/thirdparty/gaussian_splatting/helper3dg.py
# (getcolmapsinglen3d), rewritten with subprocess + error checking.
#
# Dense (MVS) pipeline adapted from Swift4D:
# https://github.com/WuJH2001/Swift4d/blob/main/colmap.sh

import os
import shutil
import subprocess


def run_cmd(cmd):
    """Run a colmap command, aborting on failure."""
    print("[colmap] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_colmap_sparse(folder):
    """Triangulate a sparse point cloud with known camera poses.

    Expects (created by pre_n3d.py):
        folder/input/     - one frame per camera (camXX.png)
        folder/input.db   - COLMAP database pre-filled with cameras + images
        folder/manual/    - known-pose model (cameras.txt, images.txt, empty points3D.txt)

    Produces (SpacetimeGaussians layout):
        folder/images/    - undistorted images
        folder/sparse/0/  - cameras.bin, images.bin, points3D.bin
    """
    dbfile = os.path.join(folder, "input.db")
    inputimagefolder = os.path.join(folder, "input")
    distortedmodel = os.path.join(folder, "distorted", "sparse")
    manualinputfolder = os.path.join(folder, "manual")
    os.makedirs(distortedmodel, exist_ok=True)

    run_cmd(["colmap", "feature_extractor",
             "--database_path", dbfile,
             "--image_path", inputimagefolder])

    run_cmd(["colmap", "exhaustive_matcher",
             "--database_path", dbfile])

    # Tolerance from https://github.com/google-research/multinerf scripts.
    run_cmd(["colmap", "point_triangulator",
             "--database_path", dbfile,
             "--image_path", inputimagefolder,
             "--input_path", manualinputfolder,
             "--output_path", distortedmodel,
             "--Mapper.ba_global_function_tolerance=0.000001"])

    run_cmd(["colmap", "image_undistorter",
             "--image_path", inputimagefolder,
             "--input_path", distortedmodel,
             "--output_path", folder,
             "--output_type", "COLMAP"])

    # Input symlinks are no longer needed once undistorted copies exist.
    shutil.rmtree(inputimagefolder)

    # Move sparse/{cameras,images,points3D}.bin -> sparse/0/ (3DGS convention).
    sparsefolder = os.path.join(folder, "sparse")
    os.makedirs(os.path.join(sparsefolder, "0"), exist_ok=True)
    for file in os.listdir(sparsefolder):
        if file == "0":
            continue
        shutil.move(os.path.join(sparsefolder, file),
                    os.path.join(sparsefolder, "0", file))


def run_colmap_dense(folder):
    """Multi-view stereo on top of the sparse reconstruction (requires CUDA COLMAP).

    Expects run_colmap_sparse() to have completed for this folder.

    Produces:
        folder/dense/workspace/fused.ply  - dense fused point cloud

    Each COLMAP stage is its own subprocess, so all GPU memory used by
    patch_match_stereo is released back to the driver when the stage exits.
    That makes it safe to call this repeatedly (once per frame offset) in a
    sequence without GPU memory accumulating across offsets.
    """
    imagefolder = os.path.join(folder, "images")
    sparsemodel = os.path.join(folder, "sparse", "0")
    workspace = os.path.join(folder, "dense", "workspace")
    os.makedirs(workspace, exist_ok=True)

    run_cmd(["colmap", "image_undistorter",
             "--image_path", imagefolder,
             "--input_path", sparsemodel,
             "--output_path", workspace])

    # cache_size caps the host-RAM image cache; patch match still runs on GPU
    # but processes one reference view at a time, bounding GPU memory per run.
    run_cmd(["colmap", "patch_match_stereo",
             "--workspace_path", workspace,
             "--PatchMatchStereo.cache_size", "32"])

    run_cmd(["colmap", "stereo_fusion",
             "--workspace_path", workspace,
             "--output_path", os.path.join(workspace, "fused.ply")])


def cleanup_dense_workspace(folder):
    """Delete the bulky MVS intermediates, keeping only fused_downsample.ply.

    patch_match_stereo writes per-view depth/normal maps (stereo/) and a full
    set of undistorted images (images/) plus the full fused cloud (fused.ply) --
    together hundreds of MB per offset. Only fused_downsample.ply is consumed
    downstream, so everything else is removed after downsampling to keep the
    disk footprint bounded across the full offset sequence.
    """
    workspace = os.path.join(folder, "dense", "workspace")
    for sub in ("stereo", "images", "sparse"):
        shutil.rmtree(os.path.join(workspace, sub), ignore_errors=True)
    for f in ("fused.ply", "fused.ply.vis"):
        try:
            os.remove(os.path.join(workspace, f))
        except FileNotFoundError:
            pass
