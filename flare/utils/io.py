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

import numpy as np
from pathlib import Path
import trimesh
from flare.core import Mesh

def make_dirs(args, run_name, finetune_color):
    experiment_dir = args.working_dir / args.output_dir / run_name


    if not finetune_color:
        images_save_path = experiment_dir / "stage_1" / "images"
        meshes_save_path = experiment_dir / "stage_1" / "meshes"
        shaders_save_path = experiment_dir / "stage_1" / "network_weights"
    else:
        images_save_path = experiment_dir / "stage_2" / "images"
        meshes_save_path = experiment_dir / "stage_2" / "meshes"
        shaders_save_path = experiment_dir / "stage_2" / "network_weights"
    images_eval_save_path = experiment_dir / "images_evaluation"
    images_save_path.mkdir(parents=True, exist_ok=True)
    images_eval_save_path.mkdir(parents=True, exist_ok=True)
    meshes_save_path.mkdir(parents=True, exist_ok=True)
    shaders_save_path.mkdir(parents=True, exist_ok=True)
    
    return images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir

def set_defaults_finetune(args):
    args.finetune_color = True
    args.final_iter = args.iterations
    args.lr_vertices = 1e-5
    args.train_deformer = False
    args.iterations = 1000
    args.sample_idx_ratio = 1
    args.fourier_features = "hashgrid"
    args.material_mlp_dims = [64, 64]
    args.light_mlp_dims = [64, 64]

def read_mesh(path, device='cpu'):
    mesh_ = trimesh.load_mesh(str(path), process=False)

    vertices = np.array(mesh_.vertices, dtype=np.float32)
    indices = None
    if hasattr(mesh_, 'faces'):
        indices = np.array(mesh_.faces, dtype=np.int32)

    return Mesh(vertices, indices, device)

def write_mesh(path, mesh):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = mesh.vertices.numpy()
    indices = mesh.indices.numpy() if mesh.indices is not None else None
    mesh_ = trimesh.Trimesh(vertices=vertices, faces=indices, process=False)
    mesh_.export(path)