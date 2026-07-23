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

import torch
import numpy as np
from utils.common.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.common.system_utils import mkdir_p
from utils.common.sh_utils import RGB2SH
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.common.graphics_utils import BasicPointCloud
from utils.common.general_utils import strip_symmetric, build_scaling_rotation
from helper_model import interpolate_point
import math


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.trbf_scale_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.denom_op = torch.empty(0)
        self._motion = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._omega = torch.empty(0)
        self.duration = 0
        self.trbfslinit = None
        self.curr_opacity = None
        self.pcd_downsample = False
        self.motion_scheduler_args = None
        self.xyz_scheduler_args = None
        self._omega_grd = None
        self._motion_grd = None
        self._trbf_scale_grd = None
        self._opacity_grd = None
        self._rotation_grd = None
        self._scaling_grd = None
        self._features_dc_grd = None
        self._xyz_grd = None
        self._features_rest = None
        self.rotation_activation = None
        self.inverse_opacity_activation = None
        self.opacity_activation = None
        self.covariance_activation = None
        self.scaling_inverse_activation = None
        self.scaling_activation = None
        self._amplitude_grd = None
        self._trbf_center = None
        self._amplitude = None
        self._features_rest_grd = None
        self.opacity_mode = None
        self.motion_mode = None
        self.motion_degree = 0
        self._trbf_scale = None
        self.min_visible_timestamps = None
        self.visibility_threshold = None
        self.temporal_span = None
        self.span_budget = None
        self.current_trbf_scale = None
        self.beta = None
        self.alpha = None
        self.tat = False
        self.tow = False
        self.per_camera_vis_filter = False
        self.num_cams = None

        self.setup_functions()

    def init_per_cam_vis_filter(self):
        self.per_camera_vis_filter = torch.zeros((self.get_xyz.shape[0], self.num_cams), device="cuda",
                                                 dtype=torch.bool)

    #####################################################################################################
    ##################################### Offset ########################################################
    # TOD0: call get_offset just once.
    def get_offset(self, t, center):
        point_times = torch.ones((self._xyz.shape[0], 1), dtype=self._xyz.dtype, requires_grad=False,
                                 device="cuda") + 0
        if self.tow:
            # TOW: Temporal Offset Warping
            offset = self.offset_warping_vectorized(timestamp=t * point_times,
                                                    temporal_center=center,
                                                    span_budget=self.span_budget,
                                                    temporal_span=self.temporal_span,
                                                    resolution=self.duration
                                                    )
        else:
            offset = (t * point_times - center).detach()

        return offset

    ###################################### Gaussian Properties ############################################

    ################# Scale #################
    # Scaling --> Constant with time
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    ################# Rotation #################
    def get_rotation_at_t(self, t, center=0.5):
        offset = self.get_offset(t, center)
        rotation = self._rotation + offset * self._omega
        return self.rotation_activation(rotation)

    ################# Covariance #################
    @property
    def get_xyz(self):
        return self._xyz

    # Motion: asymmetric Fourier deformation
    def deform_fourier_asym(self, t, center=0.5, num_freq=4):
        offset = self.get_offset(t, center)
        frequencies = torch.tensor(
            [2 * math.pi * (i + 1) for i in range(num_freq)],
            dtype=torch.float32,
            device="cuda"
        ).unsqueeze(0)

        arg = offset * frequencies + self._amplitude
        cos_terms = torch.cos(arg).unsqueeze(-1)  # (N, F, 1)
        sin_terms = torch.sin(arg).unsqueeze(-1)  # (N, F, 1)
        motion = self._motion.view(self._xyz.shape[0], num_freq, 2, 3)
        deform_ = (motion[:, :, 0, :] * cos_terms + motion[:, :, 1, :] * sin_terms).sum(dim=1)
        return deform_

    def deform_motion(self, t, center, degree):
        if self.motion_mode == "fourier_asym":
            return self.deform_fourier_asym(t, center, degree)
        else:
            raise ValueError(
                f"Unknown mode '{self.motion_mode}'. Supported mode is: 'fourier_asym'."
            )

    # Get XYZ at time=t
    def get_xyz_at_t(self, t, center):
        deform_ = self.deform_motion(t, center, self.motion_degree)
        return self._xyz + deform_

    ################# Opacity #################
    # Opacity deformation: asymmetric Gaussian kernel
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_trbf_scale(self, re_param=True):
        if re_param:
            sigma = self.sigma_min_for_M_frames * torch.exp(self._trbf_scale)
            return torch.clamp(sigma, min=0.0, max=1.1)
        return self._trbf_scale

    def deform_gauss_op_asym(self, t, center):
        """
        Asymmetric Gaussian parameterization:
        - Different decay rates (scale_left, scale_right) on each side of the center.
        """

        def trbf_function(x, scale_left, scale_right):
            # If x < 0 → left side → use scale_left
            # If x >= 0 → right side → use scale_right
            scale = torch.where(x < 0, scale_left, scale_right)
            return torch.exp(-1 * (x / scale).pow(2)), scale

        offset = self.get_offset(t, center)
        scales = self.get_trbf_scale  # should return (left_scale, right_scale)
        deform_, _ = trbf_function(offset, scales[:, 0:1], scales[:, 1:2])
        self.current_trbf_scale = _ + 0.
        return deform_

    def deform_opacity(self, t, center):
        if self.opacity_mode == "deform_gauss_op_asym":
            return self.deform_gauss_op_asym(t, center)
        else:
            raise ValueError(
                f"Unknown mode '{self.opacity_mode}'. Supported mode is: 'deform_gauss_op_asym'."
            )

    def get_opacity_at_t(self, t, center):
        deform_ = self.deform_opacity(t, center)
        opacity_t = self.get_opacity * deform_
        self.curr_opacity = opacity_t
        return opacity_t

    ################# Colour #################
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    ################ Others ###################
    @property
    def get_trbf_center(self):
        return self._trbf_center

    #####################################################################################################

    ################ Point Cloud Helpers ###################
    # 1. Initialization from point cloud.
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):

        # Downsample if required
        print(f"Down-sampling Initial point cloud by a factor of {self.pcd_downsample}")
        pcd = interpolate_point(pcd, self.pcd_downsample)

        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales = torch.clamp(scales, -10, 1.0)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # Deformations
        omega = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        num_motion_feats = self.get_num_motion_features

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        # Deformations
        motion = torch.zeros((fused_point_cloud.shape[0], num_motion_feats), device="cuda")
        self._motion = nn.Parameter(motion.requires_grad_(True))
        self._omega = nn.Parameter(omega.requires_grad_(True))
        self._trbf_scale = nn.Parameter(torch.ones((self.get_xyz.shape[0], 2), device="cuda").requires_grad_(True))
        self._trbf_center = torch.tensor(np.asarray(pcd.times)).float().cuda()

        amplitudes = torch.zeros((fused_point_cloud.shape[0], self.motion_degree), device="cuda")
        self._amplitude = nn.Parameter(amplitudes.requires_grad_(True))

        # Helpers
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        if self.trbfslinit is not None:
            nn.init.constant_(self._trbf_scale, self.trbfslinit)  # too large ?
        else:
            nn.init.constant_(self._trbf_scale, 0)  # too large ?
        # Initialize omega
        nn.init.constant_(self._omega, 0)

    # 2. Save PLY File.
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        opacities = self._opacity.detach().cpu().numpy()
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        trbf_center = self._trbf_center.detach().cpu().numpy()
        trbf_scale = self._trbf_scale.detach().cpu().numpy()
        motion = self._motion.detach().cpu().numpy()
        omega = self._omega.detach().cpu().numpy()

        gauss_feat_cat = (
            xyz, trbf_center, trbf_scale, motion, f_dc, f_rest, opacities, scale, rotation, omega)

        amplitude = self._amplitude.detach().cpu().numpy()
        gauss_feat_cat = gauss_feat_cat + (amplitude,)

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate(
            gauss_feat_cat, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 3. Load PLY File.
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        trbf_center = np.asarray(plydata.elements[0]["trbf_center"])[..., np.newaxis]
        trbf_scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("trbf_scale_")]
        trbf_scale = np.zeros((xyz.shape[0], len(trbf_scale_names)))
        for idx, attr_name in enumerate(trbf_scale_names):
            trbf_scale[:, idx] = np.asarray(plydata.elements[0][attr_name])

        num_motion = self.get_num_motion_features
        motion = np.zeros((xyz.shape[0], num_motion))
        for i in range(num_motion):
            motion[:, i] = np.asarray(plydata.elements[0]["motion_" + str(i)])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._trbf_center = torch.tensor(trbf_center, dtype=torch.float, device="cuda")
        self._trbf_scale = nn.Parameter(torch.tensor(trbf_scale, dtype=torch.float, device="cuda").requires_grad_(True))
        self._motion = nn.Parameter(torch.tensor(motion, dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))

        amp_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("amplitude")]
        amplitudes = np.zeros((xyz.shape[0], len(amp_names)))
        for idx, attr_name in enumerate(amp_names):
            amplitudes[:, idx] = np.asarray(plydata.elements[0][attr_name])
        self._amplitude = nn.Parameter(
            torch.tensor(amplitudes, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    #####################################################################################################

    ################ Gradient Helpers ###################

    def cache_gradient(self):
        self._xyz_grd += self._xyz.grad.clone()
        self._features_dc_grd += self._features_dc.grad.clone()
        self._features_rest_grd += self._features_rest.grad.clone()
        self._scaling_grd += self._scaling.grad.clone()
        self._rotation_grd += self._rotation.grad.clone()
        self._opacity_grd += self._opacity.grad.clone()
        self._trbf_scale_grd += self._trbf_scale.grad.clone()
        self._motion_grd += self._motion.grad.clone()
        self._omega_grd += self._omega.grad.clone()

        self._amplitude_grd += self._amplitude.grad.clone()

    def zero_gradient_cache(self):
        self._xyz_grd = torch.zeros_like(self._xyz, requires_grad=False)
        self._features_dc_grd = torch.zeros_like(self._features_dc, requires_grad=False)
        self._features_rest_grd = torch.zeros_like(self._features_rest, requires_grad=False)
        self._scaling_grd = torch.zeros_like(self._scaling, requires_grad=False)
        self._rotation_grd = torch.zeros_like(self._rotation, requires_grad=False)
        self._opacity_grd = torch.zeros_like(self._opacity, requires_grad=False)
        self._trbf_scale_grd = torch.zeros_like(self._trbf_scale, requires_grad=False)
        self._motion_grd = torch.zeros_like(self._motion, requires_grad=False)
        self._omega_grd = torch.zeros_like(self._omega, requires_grad=False)

        self._amplitude_grd = torch.zeros_like(self._amplitude, requires_grad=False)

    def set_batch_gradient(self, cnt):
        ratio = 1 / cnt
        self._features_dc.grad = self._features_dc_grd * ratio
        self._features_rest.grad = self._features_rest_grd * ratio
        self._xyz.grad = self._xyz_grd * ratio
        self._scaling.grad = self._scaling_grd * ratio
        self._rotation.grad = self._rotation_grd * ratio
        self._opacity.grad = self._opacity_grd * ratio
        self._trbf_scale.grad = self._trbf_scale_grd * ratio
        self._motion.grad = self._motion_grd * ratio
        self._omega.grad = self._omega_grd * ratio

        self._amplitude.grad = self._amplitude_grd * ratio

    #####################################################################################################

    ################ Optimizer Helpers ###################

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.trbf_scale_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_op = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20., "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._omega], 'lr': training_args.omega_lr, "name": "omega"},
            {'params': [self._trbf_scale], 'lr': training_args.trbfs_lr, "name": "trbf_scale"},
        ]

        l += [
            {'params': [self._motion],  # TODO: reset 0.05 to 0.5
             'lr': training_args.position_lr_init * self.spatial_lr_scale * 0.1 * training_args.movelr,
             "name": "motion"},
        ]

        l += [
            {'params': [self._amplitude],
             'lr': training_args.position_lr_init * self.spatial_lr_scale * 0.02 * training_args.movelr,
             "name": "amplitude"}
        ]

        # Optimizer
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # Schedular
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

        self.motion_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale * 0.1 * training_args.movelr,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale * 0.1 * training_args.movelr,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        """
        Learning rate scheduling per step
        """

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "motion":
                lr = self.motion_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'trbf_center']

        for i in range(self._trbf_scale.shape[1]):
            l.append('trbf_scale_{}'.format(i))

        for i in range(self._motion.shape[1]):
            l.append('motion_{}'.format(i))

        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._omega.shape[1]):
            l.append('omega_{}'.format(i))

        for i in range(self._amplitude.shape[1]):
            l.append('amplitude_{}'.format(i))

        return l

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] != 'decoder':
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if len(group["params"]) == 1 and group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                        dim=0)
                    stored_state["exp_avg_sq"] = torch.cat(
                        (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_trbf_center, new_trbfscale, new_omega, new_amplitude=None,
                              new_motion=None):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "trbf_scale": new_trbfscale,
             "omega": new_omega,
             "amplitude": new_amplitude,
             "motion": new_motion,
             }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._omega = optimizable_tensors["omega"]
        self._motion = optimizable_tensors["motion"]
        self._amplitude = optimizable_tensors["amplitude"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.trbf_scale_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom_op = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._trbf_center = torch.cat((self._trbf_center, new_trbf_center), dim=0).cuda()

        # Reset visibility filter
        self.init_per_cam_vis_filter()

    #####################################################################################################

    ################ Densification helpers ###################

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros(n_init_points, device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()

        if self.tat:
            padded_grad_thresh = torch.ones((n_init_points), device="cuda")
            padded_grad_thresh[:grad_threshold.shape[0]] = grad_threshold.squeeze()
            selected_pts_mask = padded_grad >= padded_grad_thresh
        else:
            selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_trbf_center = self._trbf_center[selected_pts_mask].repeat(N, 1)
        new_trbf_scale = self._trbf_scale[selected_pts_mask].repeat(N, 1)
        new_omega = self._omega[selected_pts_mask].repeat(N, 1)
        new_motion = self._motion[selected_pts_mask].repeat(N, 1)

        new_amplitude = self._amplitude[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling,
                                   new_rotation, new_trbf_center, new_trbf_scale, new_omega, new_amplitude,
                                   new_motion)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=torch.bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        if self.tat:
            selected_pts_mask = (grads >= grad_threshold).squeeze(-1)
        else:
            selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_trbf_center = self._trbf_center[selected_pts_mask]
        new_trbf_scale = self._trbf_scale[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        new_motion = self._motion[selected_pts_mask]

        new_amplitude = self._amplitude[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_trbf_center, new_trbf_scale, new_omega, new_amplitude,
                                   new_motion)

    def densify_gaussians(self, max_grad, min_opacity, extent, max_screen_size, splitN=1):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # TAT: Temporally Adaptive Thresholding
        if self.tat:
            avg_trbf_scale = self.trbf_scale_accum / self.denom_op
            avg_trbf_scale[avg_trbf_scale.isnan()] = 1.0
            grad_mult = self.grad_normalizer(avg_trbf_scale)
            max_grad = max_grad * grad_mult

        n_before_clone = self._xyz.shape[0]
        self.densify_and_clone(grads, max_grad, extent)
        n_after_clone = self._xyz.shape[0]

        self.densify_and_split(grads, max_grad, extent, 2)

        print("\nClone:", n_after_clone - n_before_clone)
        print("Split", self._xyz.shape[0] - n_after_clone)

        torch.cuda.empty_cache()

    def d_add_densification_stats(self, viewspace_point_tensor, update_filter, vad=False):

        if not vad:
            # Original 3DGS Densification
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                                 keepdim=True)
            self.denom[update_filter] += 1

        else:
            # VAD: Visibility Aware Densification
            screenspace_grad = torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True)

            # Filter out nearly invisible Gaussians for stability.
            update_filter &= (self.curr_opacity > 0.05).squeeze(1)

            self.xyz_gradient_accum[update_filter] += (
                    screenspace_grad[update_filter] * self.curr_opacity[update_filter])
            self.denom[update_filter] += self.curr_opacity[update_filter]

            # trbf-scale accum. (Used for adaptive thresholding)
            self.trbf_scale_accum[update_filter] += self.current_trbf_scale[update_filter]
            self.denom_op[update_filter] += 1

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._trbf_scale = optimizable_tensors["trbf_scale"]
        self._omega = optimizable_tensors["omega"]
        self._motion = optimizable_tensors["motion"]
        self._amplitude = optimizable_tensors["amplitude"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]

        self.trbf_scale_accum = self.trbf_scale_accum[valid_points_mask]
        self.denom_op = self.denom_op[valid_points_mask]

        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self._trbf_center = self._trbf_center[valid_points_mask]
        self.per_camera_vis_filter = self.per_camera_vis_filter[valid_points_mask]

    ################ Reset helpers ###################

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    ################ TRBF-SCALE helpers ###################
    @property
    def sigma_min_for_M_frames(self) -> float:
        """
        Compute the minimum Gaussian sigma so that exp(-(h/sigma)^2) >= tau
        for at least M timestamps out of T uniformly spaced in [0,1].
        """
        assert 0 < self.visibility_threshold < 1
        if self.duration <= 1 or self.min_visible_timestamps <= 1:
            return 0.0  # degenerate
        delta_t = 1.0 / (self.duration - 1)  # step in normalized time
        h = ((self.min_visible_timestamps - 1) / 2.0) * delta_t  # half-span
        k = 1.0 / math.sqrt(-math.log(self.visibility_threshold))  # this is your "k"
        return k * h

    ###### Additional Helper Functions ######

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1



    @property
    def get_num_motion_features(self):
        if self.motion_mode == "fourier_asym":
            return 2 * self.motion_degree * 3
        else:
            raise ValueError(
                f"Unknown motion_mode '{self.motion_mode}'. Supported mode is: 'fourier_asym'."
            )

    @staticmethod
    def offset_warping_vectorized(
            timestamp: torch.Tensor,  # shape (N, 1), values in [0,1)
            temporal_center: torch.Tensor,  # shape (N, 1), values in [0,1)
            temporal_span=50,  # in number of frames
            span_budget=90,  # percentage
            resolution=100  # total timestamps
    ):
        """
        1. timestamp is in range 0 to 1.
        2. temporal center is also in range 0 to 1.
        3. offset should span the range [-temporal_center, 1-temporal_center) for the
           timestamp range of [0, 1].

        Compute timestamp offset wrt temporal_center in a way that
        temporal_span number of frames around the temporal center
        is responsible for span_budget % of the timestamps.

        i.e., larger steps within temporal span and smaller steps
        outside the temporal span. steps are span budget is uniformly distributed
        within temporal span and (100 - budget) is uniformly distributed outside
        the temporal span.

        Vectorized version of offset_warping for tensors of shape (N,1).
        """

        margin = temporal_span / (2 * resolution)

        max_left_span = temporal_center - margin
        max_right_span = temporal_center + margin

        # Handle edge cases
        left_fix = (max_left_span == -margin)
        right_fix = (max_right_span == 1 + margin)

        max_left_span = torch.where(left_fix, torch.zeros_like(max_left_span), max_left_span)
        max_right_span = torch.where(left_fix, max_right_span + margin, max_right_span)

        max_left_span = torch.where(right_fix, max_left_span - margin, max_left_span)
        max_right_span = torch.where(right_fix, torch.ones_like(max_right_span), max_right_span)

        # Step sizes
        step_large = (span_budget * resolution) / (100 * temporal_span)
        step_small = ((100 - span_budget) * resolution) / (100 * (resolution - temporal_span))

        # Condition: timestamp within temporal span?
        inside_span = (timestamp >= max_left_span) & (timestamp <= max_right_span)

        # Case 1: inside span
        offset_inside = (timestamp - temporal_center) * step_large

        # Case 2: outside span
        delta90 = torch.where(timestamp > temporal_center, max_right_span, max_left_span)
        delta10 = timestamp - delta90
        sign = torch.sign(delta90)  # ensures direction
        offset_outside = (delta90 - temporal_center) * step_large + sign * delta10 * step_small

        # Combine cases
        offset = torch.where(inside_span, offset_inside, offset_outside)

        return offset

    def grad_normalizer(self, s, decay=None):
        # Fix α=1.0 sweep β={1,2,4,8} to see baseline effect.
        # Then fix β at best value and sweep α={0.7,0.9,1.0,1.2,1.5} to find desired curvature.
        s = torch.clamp(s, 0.0, 1.0)
        if decay == "inverse":
            return torch.pow(1 / (1 + self.beta * (1 - s)), self.alpha)
        return self.beta + (1 - self.beta) * torch.pow(s, self.alpha)
