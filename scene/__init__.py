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

import json
import os
import random

from arguments.default import ModelParams
from helper_train import recordpointshelper
from scene.dataset import STGSdataset
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.common.camera_utils import camera_to_JSON


class Scene:

    # gaussians : GaussianModel

    def __init__(self, args: ModelParams, gaussians, load_iteration=None, shuffle=False, resolution_scales=[1.0],
                 multiview=False, duration=50.0, loader="colmap",
                 flow_dirpath=None, flow_mask_dirpath=None, init_pcd_every=[1], pcd="sparse"
                 ):  # Optical FLow
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.refmodelpath = None

        if load_iteration:
            # Training keeps a single overwritten best checkpoint (point_cloud/best/).
            self.loaded_iter = "best"
            print("Loading best trained model")

        self.train_cameras = {}
        self.test_cameras = {}

        if loader == "colmap":
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval,
                                                          duration=duration, init_pcd_every=init_pcd_every, pcd=pcd)

        elif loader == "interdigital":
            scene_info = sceneLoadTypeCallbacks["Interdigital"](args.source_path, args.images, args.eval,
                                                               duration=duration, init_pcd_every=init_pcd_every, pcd=pcd)

        elif loader == "vru":
            scene_info = sceneLoadTypeCallbacks["VRU"](args.source_path, args.images, args.eval,
                                                               duration=duration, init_pcd_every=init_pcd_every, pcd=pcd)

        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=2)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        self.train_cameras = STGSdataset(scene_info.train_cameras, args, loader_type=loader, split="train",
                                         flow_dirpath=flow_dirpath, resolution=args.resolution  # Optical Flow
                                         )
        print("Loading Test Cameras")
        self.test_cameras = STGSdataset(scene_info.test_cameras, args, loader_type=loader, split="test",
                                        flow_dirpath=None, resolution=args.resolution)

        # Debug
        print(f"# of Training Images: {len(self.train_cameras)}")
        print(f"# of Test Images: {len(self.test_cameras)}")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                 "point_cloud", "best",
                                                 "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # Delete plyfile
        self.ply_path = scene_info.ply_path
        os.remove(scene_info.ply_path)

    def save(self, iteration):
        """Save the current model as the single best checkpoint (overwritten on every new best)."""
        point_cloud_path = os.path.join(self.model_path, "point_cloud", "best")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        with open(os.path.join(self.model_path, "best_iteration.txt"), "w") as f:
            f.write(f"best iteration: {iteration}\n")

    def recordpoints(self, iteration, string):
        numpoints = self.gaussians._xyz.shape[0]
        recordpointshelper(self.model_path, numpoints, iteration, string)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras

    def getTestCameras(self, scale=1.0):
        return self.test_cameras
