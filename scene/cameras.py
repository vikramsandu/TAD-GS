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

import numpy as np
import torch
from torch import nn


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 near=0.01, far=100.0, timestamp=0.0, image_height=None, image_width=None,
                 rayo=None, rayd=None, rays=None, projection_matrix=None, world_view_transform=None,
                 full_proj_transform=None, camera_center=None, gt_alpha_mask=None,
                 flow=None, flow_mask=None # Optical flow Priors
                 ):
        super(Camera, self).__init__()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestamp = timestamp
        self.gt_alpha_mask = gt_alpha_mask

        self.zfar = far
        self.znear = near
        self.trans = trans
        self.scale = scale
        self.image_height = image_height
        self.image_width = image_width

        self.original_image = image.to(self.data_device)
        self.projection_matrix = projection_matrix.to(self.data_device)
        self.world_view_transform = world_view_transform.to(self.data_device)
        self.full_proj_transform = full_proj_transform.to(self.data_device)
        self.camera_center = camera_center.to(self.data_device)

        self.rayo = rayo.to(self.data_device) if rayo is not None else None
        self.rayd = rayd.to(self.data_device) if rayd is not None else None
        self.rays = rays.to(self.data_device) if rays is not None else None

        # Add Optical flows
        self.flow = flow.to(self.data_device) if flow is not None else None
        self.flow_mask = flow_mask.to(self.data_device) if flow_mask is not None else None

