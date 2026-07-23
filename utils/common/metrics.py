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

from pathlib import Path
import os
import re
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import json
from tqdm import tqdm
from argparse import ArgumentParser
from torchmetrics import MultiScaleStructuralSimilarityIndexMeasure
import lpips
import math
import numpy as np

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).cuda()


def load_flow_mask(flow_masks_path, image_name, timestamp):
    """Flow-consistency mask for one test frame, used ONLY for the masked metrics.

    Masks never affect training or rendering. The mask for timestamp t is the
    flow-consistency map between frames (t-5) and t of the same camera, stored
    as '<cam>_<t-5>__<cam>_<t>.png'. Returns a bool CUDA tensor where True marks
    the dynamic region (black pixels in the PNG), or None when the path is unset
    or the specific mask does not exist (e.g. the first few timestamps).
    """
    if not flow_masks_path or flow_masks_path == "None":
        return None
    cam_id = int(re.findall(r'\d+', image_name)[0])
    fname = f"{cam_id:04}_{timestamp - 5:04}__{cam_id:04}_{timestamp:04}.png"
    fpath = os.path.join(flow_masks_path, fname)
    if not os.path.exists(fpath):
        return None
    mask = np.array(Image.open(fpath)) / 255.
    return torch.from_numpy(1 - mask).bool().cuda()


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname).convert('RGB')
        gt = Image.open(gt_dir / fname).convert('RGB')

        render = tf.to_tensor(render)  # TF converts to C, H, W
        render = render.permute(1, 2, 0).contiguous()  # H, W, C

        gt = tf.to_tensor(gt)  # TF converts to C, H, W
        gt = gt.permute(1, 2, 0).contiguous()  # H, W, C

        renders.append(render.cuda())
        gts.append(gt.cuda())
        image_names.append(fname)

    return renders, gts, image_names


def psnr_ours(rgb, gts, mask=None):
    """Calculate the PSNR metric.

    Assumes the RGB image is in [0,1]

    Args:
        rgb (torch.Tensor): Image tensor of shape [H,W3]
        mask: Dynamic Mask for the Image

    Returns:
        (float): The PSNR score
    """
    assert (rgb.shape[-1] == 3)
    assert (gts.shape[-1] == 3)

    mse_pixelwise = (rgb[..., :3] - gts[..., :3]) ** 2

    # Get PSNR
    mse = torch.mean(mse_pixelwise).item()
    psnr = 10 * math.log10(1.0 / mse)

    # Get Masked PSNR
    masked_psnr = psnr + 0.  # If no mask then Masked-PSNR = PSNR
    if mask is not None:
        masked_mse = torch.mean(mse_pixelwise[mask]).item()
        masked_psnr = 10 * math.log10(1.0 / masked_mse)

    return psnr, masked_psnr


def ssim_ours(rgb, gts, mask=None):
    """
    Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    """
    filter_size = 11
    filter_sigma = 1.5
    k1 = 0.01
    k2 = 0.03
    max_val = 1.0
    rgb = rgb.cpu().numpy()
    gts = gts.cpu().numpy()
    assert len(rgb.shape) == 3
    assert rgb.shape[-1] == 3
    assert rgb.shape == gts.shape
    import scipy.signal

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(rgb)
    mu1 = filt_fn(gts)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(rgb ** 2) - mu00
    sigma11 = filt_fn(gts ** 2) - mu11
    sigma01 = filt_fn(rgb * gts) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom

    ssim_value = np.mean(ssim_map)

    # Get Masked PSNR
    masked_ssim_value = ssim_value + 0.
    if mask is not None:
        # Adjust mask to incorporate for filter size
        mask = mask[5:-5, 5:-5]
        masked_ssim_value = np.mean(ssim_map[mask]).item()

    return ssim_value, masked_ssim_value


def msssim(rgb, gts):
    #assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    #assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return ms_ssim(torch.permute(rgb[None, ...], (0, 3, 1, 2)),
                   torch.permute(gts[None, ...], (0, 3, 1, 2))).item()


__LPIPS__ = {}


def init_lpips(net_name, device):
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)


def rgb_lpips(rgb, gts, net_name='alex', device='cpu'):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gts = gts.permute([2, 0, 1]).contiguous().to(device)
    rgb = rgb.permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gts, rgb, normalize=True).item()


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                lpipsa = []
                ms_ssims = []
                Dssims = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim_ours(renders[idx], gts[idx]))
                    psnrs.append(psnr_ours(renders[idx], gts[idx]))
                    #lpipss.append(rgb_lpips(renders[idx], gts[idx], net_name='vgg'))
                    ms_ssims.append(msssim(renders[idx], gts[idx]))
                    lpipsa.append(rgb_lpips(renders[idx], gts[idx], net_name='alex'))
                    Dssims.append((1 - ms_ssims[-1]) / 2)

                print("Scene: ", scene_dir, "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("Scene: ", scene_dir, "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                #print("Scene: ", scene_dir, "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("Scene: ", scene_dir, "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
                print("Scene: ", scene_dir, "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
                print("Scene: ", scene_dir, "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                     "PSNR": torch.tensor(psnrs).mean().item(),
                                                     #"LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                                     "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                                     "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                                     "D-SSIM": torch.tensor(Dssims).mean().item()
                                                     },

                                                    )
                per_view_dict[scene_dir][method].update(
                    {
                        "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                        "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                        #"LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                        "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                        "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                        "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)},
                    }
                )

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:

            print("Unable to compute metrics for model", scene_dir)
            raise e


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)