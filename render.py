# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ========================================================================================================
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the thirdparty/gaussian_splatting/LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import sys

sys.path.append("./utils/common")

import json
import os
import warnings
from os import makedirs
from time import time

import torch
import torchvision
from tqdm import tqdm

from arguments.default import ModelParams, PipelineParams
from helper3dg import gettestparse
from helper_train import getrenderpip, getmodel, trbfunction
from metrics import psnr_ours, ssim_ours, msssim, rgb_lpips, load_flow_mask
from scene import Scene

warnings.filterwarnings("ignore")


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, rbfbasefunction, rdpip,
               flow_masks_path):
    render, GRsetting, GRzer = getrenderpip(rdpip)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    masked_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_mask")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(masked_render_path, exist_ok=True)

    # Gaussian statistics
    scales = gaussians.get_scaling
    op = gaussians.get_opacity
    statsdict = {
        "scales_max": torch.amax(scales).item(),
        "scales_mean": torch.amin(scales).item(),
        "op_max": torch.amax(op).item(),
        "op_mean": torch.mean(op).item(),
    }
    statspath = os.path.join(model_path, "stat_" + str(iteration) + ".json")
    with open(statspath, 'w') as fp:
        json.dump(statsdict, fp, indent=True)

    # Declare metrics
    psnrs = []
    ssims = []
    ms_ssims = []
    lpipss = []

    mask_psnrs = []
    mask_ssims = []

    full_dict = {}
    per_view_dict = {}
    scene_dir = model_path
    image_names = []

    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}

    full_dict[scene_dir][iteration] = {}
    per_view_dict[scene_dir][iteration] = {}

    views_length = len(views)
    all_time = 0
    for idx in tqdm(range(views_length), desc="Rendering and metric progress"):
        view = views[idx]
        time1 = time()
        renderingpkg = render(view, gaussians, background,
                              GRsetting=GRsetting, GRzer=GRzer)  # C x H x W
        time2 = time()
        all_time += (time2 - time1)

        rendering = renderingpkg["render"]
        rendering = torch.clamp(rendering, 0, 1.0)
        gt = view.original_image[0:3, :, :].cuda().float()

        # Flow mask (used only to compute the masked metrics below)
        mask = load_flow_mask(flow_masks_path, view.image_name, idx)

        # Calculate Metrics
        psnr_value, masked_psnr_value = psnr_ours(rendering.permute(1, 2, 0), gt.permute(1, 2, 0), mask)
        ssim_value, masked_ssim_value = ssim_ours(rendering.permute(1, 2, 0), gt.permute(1, 2, 0),
                                                  mask.cpu().numpy() if mask is not None else None)
        msssim_value = msssim(rendering.permute(1, 2, 0).detach().cpu(), gt.permute(1, 2, 0).detach().cpu())
        lpips_value = rgb_lpips(rendering.permute(1, 2, 0), gt.permute(1, 2, 0), net_name='alex')

        # Append
        psnrs.append(psnr_value)
        mask_psnrs.append(masked_psnr_value)
        ssims.append(ssim_value)
        mask_ssims.append(masked_ssim_value)
        ms_ssims.append(msssim_value)
        lpipss.append(lpips_value)

        # Save Images
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        if mask is not None:
            torchvision.utils.save_image(rendering * mask.unsqueeze(0).repeat(3, 1, 1),
                                         os.path.join(masked_render_path, '{0:05d}'.format(idx) + ".png"))
        image_names.append('{0:05d}'.format(idx) + ".png")

    if len(views) > 0:
        full_dict[model_path][iteration].update({
            "PSNR": torch.tensor(psnrs).mean().item(),
            "SSIM": torch.tensor(ssims).mean().item(),
            "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
            "LPIPS-Alex": torch.tensor(lpipss).mean().item(),
            "Masked-PSNR": torch.tensor(mask_psnrs).mean().item(),
            "Masked-SSIM": torch.tensor(mask_ssims).mean().item(),
        })

        per_view_dict[model_path][iteration].update(
            {
                "PSNR": {name: val for val, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "SSIM": {name: val for val, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "MS-SSIM": {name: val for val, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                "LPIPS-Alex": {name: val for val, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                "Masked-PSNR": {name: val for val, name in zip(torch.tensor(mask_psnrs).tolist(), image_names)},
                "Masked-SSIM": {name: val for val, name in zip(torch.tensor(mask_ssims).tolist(), image_names)},
            })

        with open(model_path + "/" + str(iteration) + "_runtimeresults_v2.json", 'w') as fp:
            json.dump(full_dict, fp, indent=True)

        with open(model_path + "/" + str(iteration) + "_runtimeperview_v2.json", 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)


# render free view
def render_setnogt(model_path, name, iteration, views, gaussians, pipeline, background, rbfbasefunction, rdpip):
    render, GRsetting, GRzer = getrenderpip(rdpip)
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, background,
                           GRsetting=GRsetting, GRzer=GRzer)["render"]  # C x H x W

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))


def run_test(dataset: ModelParams, opt, iteration: int, pipeline: PipelineParams,
             skip_train: bool, skip_test: bool, multiview: bool, duration: int, rgbfunction="rgbv1",
             rdpip="v2", loader="colmap", flow_masks_path=None, tat=False, tow=False
             ):
    with torch.no_grad():
        print("use model {}".format(dataset.model))
        GaussianModel = getmodel(dataset.model)

        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.motion_mode = opt.motion_mode
        gaussians.motion_degree = opt.motion_degree
        gaussians.opacity_mode = opt.opacity_mode
        gaussians.duration = duration
        gaussians.pcd_downsample = opt.pcd_downsample
        gaussians.visibility_threshold = opt.visibility_threshold
        gaussians.min_visible_timestamps = opt.min_visible_timestamps
        gaussians.span_budget = opt.span_budget
        gaussians.temporal_span = opt.temporal_span
        gaussians.alpha = opt.alpha
        gaussians.beta = opt.beta
        gaussians.tat = tat
        gaussians.tow = tow

        anchor_points = list(range(0, duration, opt.init_pcd_every))
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, multiview=multiview,
                      duration=duration, loader=loader, init_pcd_every=anchor_points, pcd=opt.pcd)

        rbfbasefunction = trbfunction
        numchannels = 9
        bg_color = [0 for _ in range(numchannels)]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_test and not multiview:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                       background, rbfbasefunction, rdpip, flow_masks_path)
        if multiview:
            render_setnogt(dataset.model_path, "mv", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline,
                           background, rbfbasefunction, rdpip)


if __name__ == "__main__":
    args, model_extract, op_extract, pp_extract, multiview = gettestparse()
    run_test(model_extract, op_extract, args.test_iteration, pp_extract,
             args.skip_train, args.skip_test, multiview, args.duration,
             rgbfunction=args.rgbfunction, rdpip=args.rdpip, loader=args.valloader,
             flow_masks_path=args.flow_masks_path, tat=args.tat,
             tow=args.tow
             )
