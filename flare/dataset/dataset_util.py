# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import imageio
import numpy as np
import cv2
import skimage

###############################################################################
# Helpers/utils
###############################################################################

def _load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

def _load_mask(fn):
    alpha = imageio.imread(fn, as_gray=True) 
    alpha = skimage.img_as_float32(alpha)
    mask = torch.tensor(alpha / 255., dtype=torch.float32).unsqueeze(-1)
    mask[mask < 0.5] = 0.0
    # alpha = imageio.imread(fn) 
    # mask = torch.Tensor(np.array(alpha) > 127.5)[:, :, 1:2].bool().int().float()
    return mask

def _load_img(fn):
    img = imageio.imread(fn)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        # look into this
        img[..., 0:3] = srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

def _load_semantic(fn):
    img = imageio.imread(fn, as_gray=True)
    h, w = img.shape
    semantics = np.zeros((h, w, 9))
    semantics[:, :, 0] = ((img == 1) + (img == 10) + (img == 8) + (img == 7) + (img == 14) + (img == 6) + (img == 12) + (img == 13)) >= 1 # skin, nose, ears, neck, lips
    semantics[:, :, 1] = ((img == 4) + (img == 5)) >= 1 # left eye, right eye
    semantics[:, :, 2] = ((img == 2) + (img == 3)) >= 1 # left eyebrow, right eyebrow
    semantics[:, :, 3] = (img == 11) # mouth interior
    # semantics[:, :, 4] = (img == 12)  # upper lip
    # semantics[:, :, 5] = (img == 13)  # lower lip
    semantics[:, :, 5] = ((img == 17) + (img == 9)) >= 1 # hair
    semantics[:, :, 4] = ((img == 15) + (img == 16)) >= 1 # cloth, necklace
    semantics[:, :, 8] = 1. - np.sum(semantics[:, :, :8], 2) # background

    semantics = torch.tensor(semantics, dtype=torch.float32)
    return semantics


#----------------------------------------------------------------------------
# sRGB color transforms:Code adapted from Nvdiffrec
#----------------------------------------------------------------------------

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))

def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out

