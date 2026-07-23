# Voxel-downsample a dense fused point cloud until it has at most max_points.
#
# Adapted from Swift4D:
# https://github.com/WuJH2001/Swift4d/blob/main/scripts/downsample_point.py
#
# Usage:
#   python utils/preprocessing/neural_3d_video/downsample_point.py <input.ply> <output.ply>

import argparse

import open3d as o3d


def downsample_ply(input_file, output_file, max_points=40000,
                   voxel_size=0.02, voxel_step=0.01):
    pcd = o3d.io.read_point_cloud(str(input_file))
    print(f"Total points: {len(pcd.points)}")

    while len(pcd.points) > max_points:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled points: {len(pcd.points)} (voxel_size={voxel_size:.2f})")
        voxel_size += voxel_step

    o3d.io.write_point_cloud(str(output_file), pcd)
    return len(pcd.points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxel-downsample a .ply point cloud")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--max_points", default=40000, type=int)
    args = parser.parse_args()
    downsample_ply(args.input, args.output, args.max_points)
