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
import os
import random
import sys
import time
from argparse import Namespace
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

sys.path.append("./utils/common")

from loss_utils import l1_loss, ssim
from helper_train import getrenderpip, getmodel, getloss, controlGaussianCallbacks, \
    trbfunction, setgtisint8, getgtisint8, get_sampling_probability
from scene import Scene
from helper3dg import getparser, getrenderparts
from metrics import psnr_ours, load_flow_mask


# Setup random seed for reproducibility.
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(dataset, opt, duration=50, rdpip="v2",
          flow_masks_path=None, model_path=None,
          tat=False, tow=False,
          vad=False
          ):
    # Write configs
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Get Renderer
    render, GRsetting, GRzer = getrenderpip(rdpip)

    # Get Gaussian Class
    print("use model {}".format(dataset.model))
    GaussianModel = getmodel(dataset.model)

    # Initialize and Assign Gaussian properties
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.trbfslinit = opt.trbfslinit  #
    gaussians.pcd_downsample = opt.pcd_downsample
    gaussians.motion_mode = opt.motion_mode
    gaussians.motion_degree = opt.motion_degree
    gaussians.opacity_mode = opt.opacity_mode
    gaussians.duration = duration
    gaussians.visibility_threshold = opt.visibility_threshold
    gaussians.min_visible_timestamps = opt.min_visible_timestamps
    gaussians.span_budget = opt.span_budget
    gaussians.temporal_span = opt.temporal_span
    gaussians.alpha = opt.alpha
    gaussians.beta = opt.beta
    gaussians.tat = tat
    gaussians.tow = tow

    # Initialize Scene
    anchor_points = list(range(0, duration, opt.init_pcd_every)) + [duration - 1]
    scene = Scene(dataset, gaussians, duration=duration,
                  loader=dataset.loader, init_pcd_every=anchor_points,
                  pcd=opt.pcd
                  )

    # Initialize the optimizer
    gaussians.training_setup(opt)

    # Get Dataloader for Training.
    train_cameras_list = scene.getTrainCameras()
    num_cameras = len(train_cameras_list)
    num_unique_cams = num_cameras // duration

    gaussians.num_cams = num_unique_cams
    gaussians.init_per_cam_vis_filter()  # Initialize per cam visibility filter
    print(f"Total # of Training Images: {num_cameras}")

    # Enable Timing
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # Variable for Renderer
    num_channel = 9
    bg_color = [1, 1, 1] if dataset.white_background else [0 for i in range(num_channel)]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Logs + Progress Bar
    first_iter = 1
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    # Few helper vars
    gt_isint8 = getgtisint8()
    ema_loss_for_log = 0.0  # Loss for logging
    ema_psnr_for_log = 0.0  # PSNR Log logging
    flag = 0  # Densify Count
    # Detect Neural 3D Video by dataset folder in the source path, robust to the
    # scene living under either .../neural_3d_video/<scene> or
    # .../neural_3d_video/scenes/<scene>.
    is_n3dv = "neural_3d_video" in dataset.source_path.split('/')
    best_psnr = 0.0
    visibility_filter = None
    viewspace_point_tensor = None

    # Testing Iteration
    testing_iterations = [_ for _ in range(opt.densify_until_iter, opt.iterations) if _ % opt.evaluate_every_iter == 0]

    # Function to control Gaussians (Densify + Prune).
    controlGaussians = controlGaussianCallbacks[dataset.loader]

    # Depth based pruning (STG - https://github.com/oppo-us-research/SpacetimeGaussians/blob
    # /427abfc58309a4a5213843dd673fb22c4529306c/train.py#L149)
    if is_n3dv:
        z_mask = gaussians.get_xyz[:, 2] < 4.5
        print(f"\nPruning (STG Depth criterion): {z_mask.sum().item()}")
        gaussians.prune_points(z_mask)
        torch.cuda.empty_cache()

    # Training Loop
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Clear Gradient Cache if any.
        gaussians.zero_gradient_cache()

        # Sample cameras for optimization
        cam_indices = random.sample(range(num_cameras), opt.batch)

        for cam_idx in cam_indices:

            # Get Camera
            viewpoint_cam = train_cameras_list[cam_idx]

            # Render
            render_pkg = render(viewpoint_cam, gaussians, background,
                                GRsetting=GRsetting, GRzer=GRzer)
            image, viewspace_point_tensor, visibility_filter, radii, _ = getrenderparts(render_pkg)

            # Get the Ground Truth Image
            if gt_isint8:
                gt_image = viewpoint_cam.original_image.cuda().float() / 255.0
            else:
                # cast float on cuda will introduce gradient, so cast first then to cuda. at the cost of i/o
                gt_image = viewpoint_cam.original_image.float().cuda()

            # Calculate Loss (lam * L1 + (1-lam) * DSSIM)
            Ll1 = l1_loss(image, gt_image)
            loss = getloss(opt, Ll1, ssim, image, gt_image, gaussians, radii)

            # Backprop
            loss.backward()
            gaussians.cache_gradient()
            gaussians.optimizer.zero_grad(set_to_none=True)

        iter_end.record()
        gaussians.set_batch_gradient(opt.batch)

        # Progress Bar + Densification
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            psnr, _ = psnr_ours(image.permute(1, 2, 0), gt_image.permute(1, 2, 0))
            ema_psnr_for_log = 0.4 * psnr + 0.6 * ema_psnr_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}",
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}",
                                          "Gaussians": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)

            # Evaluate
            if iteration in testing_iterations:
                test_cameras_list = scene.getTestCameras()
                num_test_cameras = len(test_cameras_list)
                loss_test, psnr_test, mpsnr_test = 0., 0., 0.
                for idx in tqdm(range(num_test_cameras)):
                    viewpoint_cam = test_cameras_list[idx]
                    # Render
                    render_pkg = render(viewpoint_cam, gaussians, background,
                                        GRsetting=GRsetting, GRzer=GRzer)

                    image, viewspace_point_tensor_, visibility_filter_, radii_, depth_ = getrenderparts(render_pkg)

                    # GT
                    gt_image = viewpoint_cam.original_image.float().cuda()

                    # Flow mask (used only for the masked metric below)
                    mask = load_flow_mask(flow_masks_path, viewpoint_cam.image_name, idx)

                    # Evaluate PSNR, Loss
                    loss_test += l1_loss(image, gt_image).item()
                    p_test, m_test = psnr_ours(image.permute(1, 2, 0), gt_image.permute(1, 2, 0), mask=mask)
                    psnr_test += p_test
                    mpsnr_test += m_test

                with open(os.path.join(model_path, "training_log.txt"), "a") as f:
                    log_str = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                               f"Iteration: {iteration}  "
                               f"Train Loss: {ema_loss_for_log:.4f} "
                               f"Test Loss: {loss_test / num_test_cameras:.4f} "
                               f"Test PSNR: {psnr_test / num_test_cameras:.2f} "
                               f"Test Mask-PSNR: {mpsnr_test / num_test_cameras:.2f}\n")
                    print(log_str, end="")
                    f.write(log_str)

                # Test PSNR
                psnr_test = psnr_test / num_test_cameras

                # Save
                if psnr_test >= best_psnr:
                    print("\n[ITER {}] New best PSNR {:.2f} - overwriting best checkpoint".format(iteration, psnr_test))
                    scene.save(iteration)
                    best_psnr = psnr_test + 0.

            # Densification and pruning
            flag = controlGaussians(opt, gaussians, iteration, scene, visibility_filter,
                                    radii, viewspace_point_tensor, flag,
                                    vad=vad
                                    )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    # Close the Progress Bar.
    progress_bar.close()


if __name__ == "__main__":
    # Setup seed.
    args, lp_extract, op_extract, pp_extract = getparser()
    setgtisint8(op_extract.gtisint8)

    # Setup seed for reproducibility.
    print(f"Seed: {args.seed}")
    setup_seed(args.seed)

    # Start time.
    start_time = time.time()

    # Train
    train(lp_extract, op_extract, duration=args.duration,
          rdpip=args.rdpip, flow_masks_path=args.flow_masks_path,
          model_path=args.model_path, tat=args.tat,
          tow=args.tow,
          vad=args.vad
          )

    # End time.
    end_time = time.time()

    # Save Training Time
    total_time = end_time - start_time
    # Save training time to file
    with open(os.path.join(args.model_path, "train_time"), 'w') as cfg_log_f:
        cfg_log_f.write(f"Training Time (seconds): {total_time:.2f}\n")
        cfg_log_f.write(f"Training Time (HH:MM:SS): {time.strftime('%H:%M:%S', time.gmtime(total_time))}\n")

    # All done
    print("\nTraining complete.")
