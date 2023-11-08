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
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

# Code adapted / written by Shrish Bharadwaj.

import torch
import torch.nn as nn
import numpy as np
import pickle
from pytorch3d import ops
from .lbs import *
from pytorch3d.io import load_obj
import open3d as o3d

FLAME_MOUTH_MESH = 'assets/canonical_eye_smpl.obj'

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)
def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, flame_model_path, n_shape, n_exp, shape_params, factor=4):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.n_exp = n_exp
        self.dtype = torch.float32
        _, faces, _ = load_obj(FLAME_MOUTH_MESH, load_textures=False)
        self.register_buffer('faces_tensor', to_tensor(to_np(faces.verts_idx, dtype=np.int64), dtype=torch.long))
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template) * factor, dtype=self.dtype))   

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shape_blendshapes = shapedirs[:, :, :n_shape]
        expression_blendshapes = shapedirs[:, :, 300:300+50]
        
        self.register_buffer('shapedirs_shape', shape_blendshapes * factor)
        self.register_buffer('shapedirs_expression', expression_blendshapes * factor)

        self.v_template = self.v_template + torch.einsum('bl,mkl->bmk', [shape_params, self.shapedirs_shape]).squeeze(0)

        self.canonical_pose = torch.zeros(1, 15).float()
        self.canonical_pose[:, 6] = 0.4 # Mouth opens slightly
        self.canonical_exp = torch.zeros(1, n_exp).float()

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs) * factor, dtype=self.dtype))

        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))
        self.n_shape = n_shape

    # FLAME mesh morphing
    def forward(self, expression_params, full_pose):
        """
            Input:
                expression_params: N X number of expression parameters
                full_pose: N X number of pose parameters (15)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = expression_params.shape[0]
        betas = expression_params
        template_canonical = self.v_template.unsqueeze(0).expand(batch_size, -1, -1).to(betas.device)
        vertices, pose_feature, transformations = lbs(betas, full_pose, template_canonical,
                          self.shapedirs_expression, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        return vertices, pose_feature, transformations

    def forward_pts_batch(self, pnts_c, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=torch.float32, map2_flame_original=True):
        assert len(pnts_c.shape) == 3
        batch_size, num_points, dim = pnts_c.shape
        betas = betas.unsqueeze(1).repeat(1, num_points, 1)
        pose_feature = pose_feature.unsqueeze(1).repeat(1, num_points, 1)
        transformations = transformations.unsqueeze(1).repeat(1, num_points, 1, 1, 1)

        if map2_flame_original:
            pnts_c_original = inverse_pts_batch(pnts_c, self.canonical_exp.unsqueeze(0).repeat(batch_size, num_points, 1), 
                                          self.canonical_transformations.unsqueeze(0).repeat(batch_size, num_points, 1, 1, 1), 
                                          self.canonical_pose_feature.unsqueeze(0).repeat(batch_size, num_points, 1), shapedirs, posedirs, lbs_weights, dtype=dtype)
            pnts_p = forward_pts_batch(pnts_c_original, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=dtype)
        else:    
            pnts_p = forward_pts_batch(pnts_c, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights, dtype=dtype)

        return pnts_p


    def blendshapes_nearest(self, canonical_vertices, ghostbone, c_pts_masked=None):
        '''
        Computes the nearest neighbours for blendshapes and skinning weights
        '''
        # find nearest indx
        knn_v = self.canonical_verts.clone()
        flame_distances, idx, _ = ops.knn_points(canonical_vertices.unsqueeze(0), knn_v, K=1, return_nn=True)
        idx = idx.reshape(-1)
        num_points = self.v_template.shape[0]
        # reshape flame posedirs to fit the shape of deformer net
        nearest_posedirs = torch.transpose(self.posedirs.reshape(36, -1, 3), 0, 1)[idx, :, :]
        nearest_shapedirs = self.shapedirs_expression[idx, :, :]

        if ghostbone:
            nearest_lbs_weights = torch.zeros(len(idx), 6).to(canonical_vertices.device)
            nearest_lbs_weights[:, 1:] = self.lbs_weights[idx, :]
        else:
            nearest_lbs_weights = self.lbs_weights[idx, :]   

        if c_pts_masked is not None:
            # mouth interior does not deform with expression/pose
            _, idx_mouth_nearest, _ = ops.knn_points(c_pts_masked[0].unsqueeze(0), canonical_vertices.unsqueeze(0), K=1, return_nn=True)
            idx_mouth = torch.unique(idx_mouth_nearest)
            nearest_shapedirs[idx_mouth, ...] = 0.0
            nearest_posedirs[idx_mouth, ...] = 0.0

            if c_pts_masked[1] is not None:
                # cloth does not deform with expression
                _, idx_cloth_nearest, _ = ops.knn_points(c_pts_masked[1].unsqueeze(0), canonical_vertices.unsqueeze(0), K=1, return_nn=True)
                idx_cloth = torch.unique(idx_cloth_nearest)
                nearest_shapedirs[idx_cloth, ...] = 0.0

                if ghostbone:
                    # cloth is only controlled by ghostbone
                    nearest_lbs_weights[idx_cloth, ...] = 0.0
                    nearest_lbs_weights[idx_cloth, 0] = 1.0

        return nearest_shapedirs, nearest_posedirs, nearest_lbs_weights, flame_distances