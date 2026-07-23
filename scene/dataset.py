import os.path

from torch.utils.data import Dataset
from scene.cameras import Camera
from PIL import Image
from utils.common.general_utils import PILtoTorch
import numpy as np
import torch
from utils.common.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCV
from kornia import create_meshgrid
from helper_model import pix2ndc

# This Class should produce Camera object given CameraInfo object.
# Should be able to load original image and calculate rays on the fly during training.


class STGSdataset(Dataset):
    def __init__(
            self,
            dataset,
            args,
            loader_type=None,
            split="train",
            flow_dirpath=None,
            resolution = 1
    ):
        self.dataset = dataset
        self.data_device = args.data_device
        self.loader_type = loader_type
        self.flow_dirpath = flow_dirpath
        self.split = split
        self.resolution = resolution

    def __getitem__(self, index):
        cam_info = self.dataset[index]

        # Read Image From Image Path in camera_info and Convert to Tensor
        image = Image.open(cam_info.image_path)

        # Rescale
        h, w = image.size
        new_h, new_w = h//self.resolution, w//self.resolution

        image_tensor = PILtoTorch(image, resolution=(new_h, new_w))[:3, ...]
        image_tensor = image_tensor.clamp(0.0, 1.0).cuda()

        # Read Optical Flows
        flow = None
        flow_mask = None
        if (self.flow_dirpath is not None) and (self.split == "train") and os.path.exists(cam_info.flow_file_path):
            flow = torch.tensor(np.load(cam_info.flow_file_path)['arr_0'])
            flow_mask = torch.from_numpy(np.array(Image.open(cam_info.flow_mask_file_path))) / 255.

        # Get Height and Width
        H, W = image_tensor.shape[1], image_tensor.shape[2]

        # Get Timestamp, near , far, etc
        timestamp = cam_info.timestamp
        near = cam_info.near
        far = cam_info.far

        FoVx = cam_info.FovX
        FoVy = cam_info.FovY

        R = cam_info.R
        T = cam_info.T

        cxr = cam_info.cxr
        cyr = cam_info.cyr

        # Get World2View Transform, Projection, and Full Transforms
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrixCV(znear=near, zfar=far, fovX=FoVx, fovY=FoVy, cx=cxr, cy=cyr).transpose(0, 1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

        # Get Camera Center
        camera_center = world_view_transform.inverse()[3, :3]

        # Get rays
        rayo, rayd = self.get_rays(image_height=H, image_width=W, proj_matrix=projection_matrix,
                                   w2c_matrix=world_view_transform, cam_center=camera_center
                                   )
        rays = torch.cat([rayo, rayd], dim=1)

        # Return Camera
        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                      image_name=cam_info.image_name, uid=index, data_device=self.data_device,
                      near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, image_height=H, image_width=W,
                      image=image_tensor, projection_matrix=projection_matrix, world_view_transform=world_view_transform,
                      full_proj_transform=full_proj_transform, camera_center=camera_center,
                      rayo=rayo, rayd=rayd, rays=rays, flow=flow, flow_mask=flow_mask
                      )

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def get_rays(image_height,
                 image_width,
                 proj_matrix,
                 w2c_matrix,
                 cam_center
                 ):
        projectinverse = proj_matrix.T.inverse()
        camera2wold = w2c_matrix.T.inverse()
        pixgrid = create_meshgrid(image_height, image_width, normalized_coordinates=False, device="cpu")[0]
        pixgrid = pixgrid.cuda()  # H,W,

        xindx = pixgrid[:, :, 0]  # x
        yindx = pixgrid[:, :, 1]  # y

        ndcy, ndcx = pix2ndc(yindx, image_height), pix2ndc(xindx, image_width)
        ndcx = ndcx.unsqueeze(-1)
        ndcy = ndcy.unsqueeze(-1)  # * (-1.0)

        ndccamera = torch.cat((ndcx, ndcy, torch.ones_like(ndcy) * (1.0), torch.ones_like(ndcy)), 2)  # N,4

        projected = ndccamera @ projectinverse.T
        diretioninlocal = projected / projected[:, :, 3:]  # v

        direction = diretioninlocal[:, :, :3] @ camera2wold[:3, :3].T
        rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)

        rayo = cam_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(
            0)  # rayo.permute(2, 0, 1).unsqueeze(0)
        rayd = rays_d.permute(2, 0, 1).unsqueeze(0)

        return rayo, rayd