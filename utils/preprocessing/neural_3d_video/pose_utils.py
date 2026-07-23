# Vendored from SpacetimeGaussians (MIT License, Copyright (c) 2023 OPPO):
# https://github.com/oppo-us-research/SpacetimeGaussians/blob/main/thirdparty/gaussian_splatting/utils/my_utils.py
#
# Converts Neural 3D Video (DyNeRF) LLFF-style poses from poses_bounds.npy
# into world-to-camera matrices / quaternions in COLMAP's convention.

import numpy as np


def posetow2c_matrcs(poses):
    """LLFF-style c2w poses (3x5xN, [down right back] convention) -> list of 4x4 w2c matrices."""
    tmp = inversestep4(inversestep3(inversestep2(inversestep1(poses))))
    N = tmp.shape[0]
    ret = []
    for i in range(N):
        ret.append(tmp[i])
    return ret


def inversestep4(c2w_mats):
    return np.linalg.inv(c2w_mats)


def inversestep3(newposes):
    tmp = newposes.transpose([2, 0, 1])  # N, 3, 4
    N, _, __ = tmp.shape
    zeros = np.zeros((N, 1, 4))
    zeros[:, 0, 3] = 1
    c2w_mats = np.concatenate([tmp, zeros], axis=1)
    return c2w_mats


def inversestep2(newposes):
    return newposes[:, 0:4, :]


def inversestep1(newposes):
    # LLFF [down right back] -> [right up back] axis swap
    poses = np.concatenate([newposes[:, 1:2, :], newposes[:, 0:1, :],
                            -newposes[:, 2:3, :], newposes[:, 3:4, :],
                            newposes[:, 4:5, :]], axis=1)
    return poses


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
