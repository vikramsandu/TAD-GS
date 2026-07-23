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

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE ####################################
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################


import math
import time

import torch

from scene.tad_gaussian import GaussianModel


def render(viewpoint_camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0,
                    GRsetting=None, GRzer=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    start_time = time.time()
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    # Get Gaussian params
    means2D = screenspace_points
    trbf_center = pc.get_trbf_center
    scales = pc.get_scaling

    # Get Gaussian params at time=t
    means3D = pc.get_xyz_at_t(t=viewpoint_camera.timestamp, center=trbf_center)
    opacity = pc.get_opacity_at_t(t=viewpoint_camera.timestamp, center=trbf_center)
    rotations = pc.get_rotation_at_t(t=viewpoint_camera.timestamp, center=trbf_center)
    shs = pc.get_features
    colors_precomp = None
    cov3D_precomp = None

    end_pre_time = time.time()

    # Render t with opacity = t # 3, 12, 210
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    end_rast_time = time.time()

    duration_pre = end_rast_time - end_pre_time
    duration_rast = end_pre_time - start_time
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration_pre": duration_pre,
            "duration_rast": duration_rast
            }
