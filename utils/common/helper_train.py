#
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

# =============================================

# This license is additionally subject to the following restrictions:

# Licensor grants non-exclusive rights to use the Software for research purposes
# to research users (both academic and industrial), free of charge, without right
# to sublicense. The Software may be used "non-commercially", i.e., for research
# and/or evaluation purposes only.

# Subject to the terms and conditions of this License, you are granted a
# non-exclusive, royalty-free, license to reproduce, prepare derivative works of,
# publicly display, publicly perform and distribute its Work and any resulting
# derivative works in any form.
#

import os

import torch


def getrenderpip(option="train_ours_full"):
    print("render option", option)
    if option == "tad_gaussian":
        from gaussian_renderer import render
        from diff_gaussian_rasterization_ch3 import GaussianRasterizationSettings
        from diff_gaussian_rasterization_ch3 import GaussianRasterizer
        return render, GaussianRasterizationSettings, GaussianRasterizer
    else:
        raise NotImplementedError("Render {} not implemented".format(option))


def getmodel(model="oursfull"):
    if model == "tad_gaussian":
        from scene.tad_gaussian import GaussianModel
    else:
        raise NotImplementedError("model {} not implemented".format(model))
    return GaussianModel


def getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii, l_tv=None):
    if opt.reg == 1:  # add optical flow loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(
            gaussians._motion) / gaussians._motion.shape[0]
    elif opt.reg == 0:
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    elif opt.reg == 9:  # regularizer on the rotation
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(
            gaussians._omega[radii > 0] ** 2)
    elif opt.reg == 10:  # regularizer on the motion
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(
            gaussians._motion[radii > 0] ** 2)
    elif opt.reg == 4:
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.regl * torch.sum(
            gaussians.get_scaling) / gaussians._motion.shape[0]
    elif opt.reg == 5:
        loss = Ll1
    elif opt.reg == 6:
        ratio = torch.mean(gt_image) - 0.5 + opt.lambda_dssim
        ratio = torch.clamp(ratio, 0.0, 1.0)
        loss = (1.0 - ratio) * Ll1 + ratio * (1.0 - ssim(image, gt_image))
    elif opt.reg == 7:
        Ll1 = Ll1 / (torch.mean(gt_image) * 2.0)  # normalize L1 loss
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
    elif opt.reg == 8:
        N = gaussians._xyz.shape[0]
        mean = torch.mean(gaussians._xyz, dim=0, keepdim=True)
        varaince = (mean - gaussians._xyz) ** 2
        loss = (1.0 - opt.lambda_dssim) * Ll1 + 0.0002 * opt.lambda_dssim * torch.sum(varaince) / N
    elif opt.reg == 100:
        loss = ((1.0 - opt.lambda_dssim) * Ll1 +
                opt.lambda_dssim * (1.0 - ssim(image, gt_image)) +
                opt.lambda_tv * l_tv
                )
    return loss


def control_gaussians_n3dv(opt, gaussians, iteration, scene,
                           visibility_filter, radii, viewspace_point_tensor,
                           flag, vad=False):
    """Densification + pruning logic for Neural 3D Video (N3DV). Returns: flag"""

    # Stage 1: Before densify_until_iter
    if iteration < opt.densify_until_iter:

        # Gradient accumulation of the visible Gaussians.
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                             radii[visibility_filter])
        gaussians.d_add_densification_stats(viewspace_point_tensor, visibility_filter,
                                            vad=vad)

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            scene.recordpoints(iteration, "before densify")
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None

            # Densification
            gaussians.densify_gaussians(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
            scene.recordpoints(iteration, "after densify")

            # Opacity based Pruning
            prune_mask = (gaussians.get_opacity < opt.opthr).squeeze()
            print(f"\nPruning (Opacity criterion): {prune_mask.sum().item()}")
            gaussians.prune_points(prune_mask)
            torch.cuda.empty_cache()
            scene.recordpoints(iteration, "additionally prune_mask")

        if iteration % opt.opacity_reset_interval == 0:
            print("reset opacity and trbf scale")
            gaussians.reset_opacity()

    else:
        if iteration % 500 == 1:
            z_mask = gaussians.get_xyz[:, 2] < 4.5  # for stability
            print(f"\nPruning (Z-mask criterion): {z_mask.sum().item()}")
            gaussians.prune_points(z_mask)
            torch.cuda.empty_cache()

    return flag


def control_gaussians_interdigital(opt, gaussians, iteration, scene,
                                   visibility_filter, radii, viewspace_point_tensor,
                                   flag, vad=False):
    """Densification + pruning logic for Interdigital (also used for VRU). Returns: flag"""

    # Stage 1: Before densify_until_iter
    if iteration < opt.densify_until_iter:

        # Gradient accumulation of the visible Gaussians.
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                             radii[visibility_filter])
        gaussians.d_add_densification_stats(viewspace_point_tensor, visibility_filter,
                                            vad=vad)

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            scene.recordpoints(iteration, "before densify")
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None

            # Densification
            gaussians.densify_gaussians(opt.densify_grad_threshold, opt.opthr, scene.cameras_extent, size_threshold)
            scene.recordpoints(iteration, "after densify")

            # Opacity based Pruning
            prune_mask = (gaussians.get_opacity < opt.opthr).squeeze()
            print(f"\nPruning (Opacity criterion): {prune_mask.sum().item()}")
            gaussians.prune_points(prune_mask)
            torch.cuda.empty_cache()
            scene.recordpoints(iteration, "additionally prune_mask")

        if iteration % opt.opacity_reset_interval == 0:
            print("reset opacity and trbf scale")
            gaussians.reset_opacity()

    return flag


def get_sampling_probability(loss_tracker_dict, timestamp_counter_dict,
                             num_unique_cams, temperature=0.005
                             ):
    """
    Sample the frames (or timestamps) with larger errors more
    for better supervision and/or (probably densification)
    Args:
        -- num_unique_cams: number of unique cameras in training.
        -- loss_tracker_dict: Cumulative loss for all timestamps
        -- timestamp_counter_dict:  the number of times loss has been
         added in loss_tracker_dict for each timestamp.
        -- temperature: Controls the variance between sampling probability.

    Returns: Sampling probability for each camera/frame
    """
    # Calculate average loss for each timestamps
    accum_errors = []
    for timestamp in sorted(loss_tracker_dict.keys()):
        avg_loss = loss_tracker_dict[timestamp] / timestamp_counter_dict[timestamp]
        accum_errors.append(avg_loss)

    # Get sampling prob based on accum errors.
    # larger the error => higher the sampling prob.
    probs = torch.softmax(torch.tensor(accum_errors) / temperature, dim=0)

    # Spread the prob equally among each camera for every timestamp.
    probs = probs.repeat(num_unique_cams) / num_unique_cams

    return probs


def recordpointshelper(model_path, numpoints, iteration, string):
    txtpath = os.path.join(model_path, "exp_log.txt")

    with open(txtpath, 'a') as file:
        file.write("iteration at " + str(iteration) + "\n")
        file.write(string + " pointsnumber " + str(numpoints) + "\n")


def trbfunction(x):
    return torch.exp(-1 * x.pow(2))


def setgtisint8(value):
    print("set current resized gt image as int8 for memory: ", value)
    os.environ['gtisint8'] = str(value)


def getgtisint8():
    try:
        return bool(int(os.getenv('gtisint8')))
    except:
        return False


# Call control Gaussians based on dataset.
controlGaussianCallbacks = {
    "colmap": control_gaussians_n3dv,
    "interdigital": control_gaussians_interdigital,
    "vru": control_gaussians_interdigital
}
