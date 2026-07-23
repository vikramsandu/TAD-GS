#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary
from utils.common.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.common.sh_utils import SH2RGB
from utils.common.graphics_utils import BasicPointCloud
import natsort


################### Basic Utilities  ###################
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    near: float
    far: float
    timestamp: float
    pose: np.array
    hpdirecitons: np.array
    cxr: float
    cyr: float

    # Flow priors
    flow_file_path: str
    flow_mask_file_path: str


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def fetchPly(path, return_time=True):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    times = np.vstack([vertices['t']]).T if return_time else normals
    return BasicPointCloud(points=positions, colors=colors, normals=normals, times=times)


def storePly(path, xyzt, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('t', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    xyz = xyzt[:, :3]
    normals = np.zeros_like(xyz)

    elements = np.empty(xyzt.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzt, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


############################### N3DV Dataset  ###############################
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50):
    cam_infos = []

    # pose in llff. pipeline by hypereel
    originnumpy = os.path.join(os.path.dirname(os.path.dirname(images_folder)), "poses_bounds.npy")
    with open(originnumpy, 'rb') as numpy_file:
        poses_bounds = np.load(numpy_file)

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        bounds = poses_bounds[:, -2:]

        near = bounds.min() * 0.95
        far = bounds.max() * 1.05

        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # 19, 3, 5

        H, W, focal = poses[0, :, -1]
        cx, cy = W / 2.0, H / 2.0

        K = np.eye(3)
        K[0, 0] = focal * W / W / 2.0
        K[0, 2] = cx * W / W / 2.0
        K[1, 1] = focal * H / H / 2.0
        K[1, 2] = cy * H / H / 2.0

        imageH = int(H // 2)  # note hard coded to half of the original image size
        imageW = int(W // 2)

    totalcamname = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)

    sortedtotalcamelist = natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]

        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, ("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE "
                           "cameras) supported!")

        for j in range(startime, startime + int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_" + str(startime), "colmap_{}".format(j), 1)
            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)
            image = None
            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,
                                      timestamp=(j - startime) / duration, pose=1, hpdirecitons=1, cxr=0.0, cyr=0.0,
                                      flow_file_path=None, flow_mask_file_path=None)

            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,
                                      timestamp=(j - startime) / duration, pose=None, hpdirecitons=None, cxr=0.0,
                                      cyr=0.0, flow_file_path=None, flow_mask_file_path=None)
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfo(path, images, eval, duration=50,
                        init_pcd_every=None, pcd="sparse"
                        ):
    # Read Camera Intrinsics and Extrinsics
    try:
        # Comment: un-comment path_
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images
    start_time = os.path.basename(path).split("_")[1]  # colmap_0,
    assert start_time.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    start_time = int(start_time)

    # Read Colmap cameras
    near = 0.01
    far = 100
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir), near=near, far=far,
                                           startime=start_time, duration=duration)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        # 0-th camera is the test cam
        train_cam_infos = [_ for _ in cam_infos if "cam00" not in _.image_name]
        test_cam_infos = [_ for _ in cam_infos if "cam00" in _.image_name]
        unique_check = []
        for cam_info in test_cam_infos:
            if cam_info.image_name not in unique_check:
                unique_check.append(cam_info.image_name)
        assert len(unique_check) == 1

        sanitycheck = []
        for cam_info in train_cam_infos:
            if cam_info.image_name not in sanitycheck:
                sanitycheck.append(cam_info.image_name)
        for test_name in unique_check:
            assert test_name not in sanitycheck
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:2]  # dummy

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Initialize Point cloud (STGS-type Initialization)
    # Dense (MVS) initialization is auto-detected per anchor frame, independent
    # of the --pcd flag: any COLMAP offset that has a dense cloud at
    # dense/workspace/fused_downsample.ply is initialized from it, the rest use
    # the sparse SfM triangulation. coffee_martini and flame_salmon_1 are
    # preprocessed with dense clouds on every anchor; other scenes are fully sparse.
    def dense_ply_for(offset):
        return os.path.join(path, "dense/workspace/fused_downsample.ply").replace(
            "colmap_" + str(start_time), "colmap_" + str(offset), 1)

    anchors = [i for i in init_pcd_every if start_time <= i < start_time + duration]
    n_dense = sum(os.path.exists(dense_ply_for(i)) for i in anchors)
    pcd_type = "dense" if n_dense else "sparse"
    print(f"Point cloud initialization: {pcd_type}"
          + (f" ({n_dense}/{len(anchors)} anchors dense)" if n_dense else ""))

    total_ply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + "_" + pcd_type + ".ply")
    if not os.path.exists(total_ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        total_xyz = []
        total_rgb = []
        total_time = []
        for i in range(start_time, start_time + duration):
            if i in init_pcd_every:
                dense_ply = dense_ply_for(i)
                if os.path.exists(dense_ply):
                    # MVS fused + voxel-downsampled cloud for this anchor.
                    pcd_ = fetchPly(dense_ply, return_time=False)
                    xyz = np.asarray(pcd_.points)
                    rgb = np.asarray(pcd_.colors) * 255.
                else:
                    # Read Sparse point cloud (SfM triangulation)
                    bin_path = os.path.join(path, f"sparse/0/points3D.bin").replace(
                        "colmap_" + str(start_time),
                        "colmap_" + str(i), 1)
                    xyz, rgb, _ = read_points3D_binary(bin_path)

                # Accumulate
                total_xyz.append(xyz)
                total_rgb.append(rgb)
                total_time.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / duration)

        # Concat
        xyz = np.concatenate(total_xyz, axis=0)
        rgb = np.concatenate(total_rgb, axis=0)
        total_time = np.concatenate(total_time, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, total_time), axis=1)
        storePly(total_ply_path, xyzt, rgb)

    try:
        pcd = fetchPly(total_ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=total_ply_path)
    return scene_info


############################### Interdigital Dataset  ###############################
def readColmapCamerasInterdigital(cam_extrinsics, cam_intrinsics, images_folder, near, far, startime=0, duration=50,
                                 flow_dirpath=None, flow_mask_dirpath=None  # Optical Flow
                                 ):
    cam_infos = []
    totalcamname = []
    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        totalcamname.append(extr.name)

    sortedtotalcamelist = natsort.natsorted(totalcamname)
    sortednamedict = {}
    for i in range(len(sortedtotalcamelist)):
        sortednamedict[sortedtotalcamelist[i]] = i  # map each cam with a number

    for idx, key in enumerate(cam_extrinsics):  # first is cam20_ so we strictly sort by camera name
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        for j in range(startime, startime + int(duration)):
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image_path = image_path.replace("colmap_" + str(startime), "colmap_{}".format(j), 1)

            cxr = ((intr.params[2]) / width - 0.5)
            cyr = ((intr.params[3]) / height - 0.5)

            K = np.eye(3)
            K[0, 0] = focal_length_x  # * 0.5
            K[0, 2] = intr.params[2]  # * 0.5
            K[1, 1] = focal_length_y  # * 0.5
            K[1, 2] = intr.params[3]  # * 0.5

            halfH = round(height / 2.0)
            halfW = round(width / 2.0)

            assert os.path.exists(image_path), "Image {} does not exist!".format(image_path)

            # image = Image.open(image_path)
            image = None

            # Add Optical Flow and Mask Paths (Assuming 5 frames apart)
            cam_id = image_name[3:].zfill(4)

            flow_file_path = None
            flow_mask_file_path = None
            if flow_dirpath is not None:
                flow_file_path = os.path.join(flow_dirpath, f"{cam_id}_{j:04}__{cam_id}_{j + 5:04}.npz")
                flow_mask_file_path = os.path.join(flow_mask_dirpath, f"{cam_id}_{j:04}__{cam_id}_{j + 5:04}.png")

            if j == startime:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,
                                      timestamp=(j - startime) / duration, pose=1, hpdirecitons=1, cxr=cxr, cyr=cyr,
                                      flow_file_path=flow_file_path,
                                      flow_mask_file_path=flow_mask_file_path)  # Flow priors
            else:
                cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path,
                                      image_name=image_name, width=width, height=height, near=near, far=far,
                                      timestamp=(j - startime) / duration, pose=None, hpdirecitons=None, cxr=cxr,
                                      cyr=cyr,
                                      flow_file_path=flow_file_path,
                                      flow_mask_file_path=flow_mask_file_path)  # Flow priors
            cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapSceneInfoInterdigital(path, images, eval, duration=50,
                                   init_pcd_every=None, pcd="sparse"
                                   ):
    # Read Camera Intrinsics and Extrinsics
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images

    start_time = os.path.basename(path).split("_")[1]  # colmap_0,
    assert start_time.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    start_time = int(start_time)

    near = 0.01
    far = 100
    # Comment: Replace path_ with path here
    cam_infos_unsorted = readColmapCamerasInterdigital(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                                      images_folder=os.path.join(path, reading_dir), near=near, far=far,
                                                      startime=start_time, duration=duration,
                                                      flow_dirpath=None, flow_mask_dirpath=None
                                                      )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        # 2nd row-2nd column camera for testing.
        train_cam_infos = [_ for _ in cam_infos if "cam05" not in _.image_name]
        test_cam_infos = [_ for _ in cam_infos if "cam05" in _.image_name]
        if len(test_cam_infos) > 0:
            unique_check = []
            for cam_info in test_cam_infos:
                if cam_info.image_name not in unique_check:
                    unique_check.append(cam_info.image_name)
            assert len(unique_check) == 1

            sanitycheck = []
            for cam_info in train_cam_infos:
                if cam_info.image_name not in sanitycheck:
                    sanitycheck.append(cam_info.image_name)
            for test_name in unique_check:
                assert test_name not in sanitycheck
        else:
            first_cam = cam_infos[0].image_name
            print("do custom loader training, select first cam as test frame: ", first_cam)
            cam_infos = natsort.natsorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
            train_cam_infos = [_ for _ in cam_infos if first_cam not in _.image_name]
            test_cam_infos = [_ for _ in cam_infos if first_cam in _.image_name]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:4]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Initialize Point cloud (STGS-type Initialization)
    total_ply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")
    if not os.path.exists(total_ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        total_xyz = []
        total_rgb = []
        total_time = []
        for i in range(start_time, start_time + duration):
            if i in init_pcd_every:
                if pcd == "sparse":
                    # Read Sparse point cloud
                    bin_path = os.path.join(path, f"sparse/0/points3D.bin").replace(
                        "colmap_" + str(start_time),
                        "colmap_" + str(i), 1)
                    xyz, rgb, _ = read_points3D_binary(bin_path)
                else:
                    # Read Dense point cloud
                    ply_path = os.path.join(path, f"sparse/0/points3D.ply").replace(
                        "colmap_" + str(start_time),
                        "colmap_" + str(i), 1)
                    pcd_ = fetchPly(ply_path, return_time=False)
                    xyz = np.asarray(pcd_.points)
                    rgb = np.asarray(pcd_.colors) * 255.

                # Accumulate
                total_xyz.append(xyz)
                total_rgb.append(rgb)
                total_time.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / duration)

        # Concat
        xyz = np.concatenate(total_xyz, axis=0)
        rgb = np.concatenate(total_rgb, axis=0)
        total_time = np.concatenate(total_time, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, total_time), axis=1)
        storePly(total_ply_path, xyzt, rgb)

    try:
        pcd = fetchPly(total_ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=total_ply_path)
    return scene_info


def readColmapSceneInfoVru(path, images, eval, duration=50,
                           init_pcd_every=None, pcd="sparse"
                           ):
    # Read Camera Intrinsics and Extrinsics
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images

    start_time = os.path.basename(path).split("_")[1]  # colmap_0,
    assert start_time.isdigit(), "Colmap folder name must be colmap_<startime>_<duration>!"
    start_time = int(start_time)

    near = 0.01
    far = 100
    cam_infos_unsorted = readColmapCamerasInterdigital(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                                      images_folder=os.path.join(path, reading_dir), near=near, far=far,
                                                      startime=start_time, duration=duration,
                                                      flow_dirpath=None, flow_mask_dirpath=None
                                                      )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        # 2nd row-2nd column camera for testing.
        train_cam_infos = [_ for _ in cam_infos if "cam17" not in _.image_name]
        test_cam_infos = [_ for _ in cam_infos if "cam17" in _.image_name]
        if len(test_cam_infos) > 0:
            unique_check = []
            for cam_info in test_cam_infos:
                if cam_info.image_name not in unique_check:
                    unique_check.append(cam_info.image_name)
            assert len(unique_check) == 1

            sanitycheck = []
            for cam_info in train_cam_infos:
                if cam_info.image_name not in sanitycheck:
                    sanitycheck.append(cam_info.image_name)
            for test_name in unique_check:
                assert test_name not in sanitycheck
        else:
            first_cam = cam_infos[0].image_name
            print("do custom loader training, select first cam as test frame: ", first_cam)
            cam_infos = natsort.natsorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
            train_cam_infos = [_ for _ in cam_infos if first_cam not in _.image_name]
            test_cam_infos = [_ for _ in cam_infos if first_cam in _.image_name]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos[:4]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Initialize Point cloud (STGS-type Initialization)
    total_ply_path = os.path.join(path, "sparse/0/points3D_total" + str(duration) + ".ply")
    if not os.path.exists(total_ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        total_xyz = []
        total_rgb = []
        total_time = []
        for i in range(start_time, start_time + duration):
            if i in init_pcd_every:
                if pcd == "sparse":
                    # Read Sparse point cloud
                    bin_path = os.path.join(path, f"sparse/0/points3D.bin").replace(
                        "colmap_" + str(start_time),
                        "colmap_" + str(i), 1)
                    xyz, rgb, _ = read_points3D_binary(bin_path)
                else:
                    # Read Dense point cloud
                    ply_path = os.path.join(path, f"sparse/0/points3D.ply").replace(
                        "colmap_" + str(start_time),
                        "colmap_" + str(i), 1)
                    pcd_ = fetchPly(ply_path, return_time=False)
                    xyz = np.asarray(pcd_.points)
                    rgb = np.asarray(pcd_.colors) * 255.

                # Accumulate
                total_xyz.append(xyz)
                total_rgb.append(rgb)
                total_time.append(np.ones((xyz.shape[0], 1)) * (i - start_time) / duration)

        # Concat
        xyz = np.concatenate(total_xyz, axis=0)
        rgb = np.concatenate(total_rgb, axis=0)
        total_time = np.concatenate(total_time, axis=0)
        assert xyz.shape[0] == rgb.shape[0]
        xyzt = np.concatenate((xyz, total_time), axis=1)
        storePly(total_ply_path, xyzt, rgb)

    try:
        pcd = fetchPly(total_ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=total_ply_path)
    return scene_info


# Callbacks
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Interdigital": readColmapSceneInfoInterdigital,
    "VRU": readColmapSceneInfoVru
}
