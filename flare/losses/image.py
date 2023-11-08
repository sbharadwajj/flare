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

#----------------------------------------------------------------------------
# HDR image losses
#----------------------------------------------------------------------------

def _tonemap_srgb(f):
    return torch.where(f > 0.0031308, torch.pow(torch.clamp(f, min=0.0031308), 1.0/2.4)*1.055 - 0.055, 12.92*f)

def image_loss_fn(img, target):

    img    = _tonemap_srgb(torch.log(torch.clamp(img, min=0, max=65535) + 1))
    target = _tonemap_srgb(torch.log(torch.clamp(target, min=0, max=65535) + 1))

    out = torch.nn.functional.l1_loss(img, target)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of image_loss contains inf or NaN"
    return out, img, target


def mask_loss(masks, gbuffers, loss_function = torch.nn.MSELoss()):
    """ Compute the mask term as the mean difference between the original masks and the rendered masks.
    
    Args:
        views (List[View]): Views with masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """

    loss = 0.0
    for gt_mask, gbuffer_mask in zip(masks, gbuffers):
        loss += loss_function(gt_mask, gbuffer_mask)
    return loss / len(masks)


def shading_loss_batch(pred_color_masked, views, batch_size):
    """ Compute the image loss term as the mean difference between the original images and the rendered images from a shader.
    """

    color_loss, tonemap_pred, tonemap_target = image_loss_fn(pred_color_masked[..., :3] * views["mask"], views["img"] * views["mask"])

    return color_loss / batch_size, pred_color_masked[..., :3], [tonemap_pred, tonemap_target]