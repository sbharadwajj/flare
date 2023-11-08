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

from arguments import config_parser
import os
from pathlib import Path
import torch
from flame.FLAME import FLAME
from flare.core import (
    Mesh, Renderer
)
from flare.modules import (
    NeuralShader, get_deformer_network
)
from flare.utils import (
    AABB, read_mesh,
    save_individual_img, make_dirs, save_relit_intrinsic_materials
)
import nvdiffrec.render.light as light
from flare.dataset import DatasetLoader
from flare.dataset import dataset_util
from flare.metrics import metrics

# Select the device
device = torch.device('cpu')
devices = 0
if torch.cuda.is_available() and devices >= 0:
    device = torch.device(f'cuda:{devices}')


# ==============================================================================================
# evaluation
# ==============================================================================================    
def run(args, mesh, views, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, lgt):
    ## ============== deform ==============================     
    shapedirs, posedirs, lbs_weights = deformer_net.query_weights(mesh.vertices)
    eval_vertices = mesh.vertices
    batched_verts = eval_vertices.unsqueeze(0).repeat(views["img"].shape[0], 1, 1)

    _, pose_features, transformations = FLAMEServer(expression_params=views["flame_expression"], full_pose=views["flame_pose"])
    if args.ghostbone:
        transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(views["img"].shape[0], -1, -1, -1).float().to(device), transformations], 1)
    deformed_vertices = FLAMEServer.forward_pts_batch(pnts_c=batched_verts, betas=views["flame_expression"], transformations=transformations, pose_feature=pose_features, 
                                        shapedirs=shapedirs, posedirs=posedirs, lbs_weights=lbs_weights, dtype=torch.float32, map2_flame_original=True)

    d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
    ## ============== Rasterize ==============================
    gbuffers = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                        channels=channels_gbuffer, with_antialiasing=True, 
                        canonical_v=mesh.vertices, canonical_idx=mesh.indices)
    
    ## ============== predict color ==============================
    rgb_pred, cbuffers, gbuffer_mask = shader.shade(gbuffers, views, mesh, args.finetune_color, lgt)

    return rgb_pred, gbuffers, cbuffers

# ==============================================================================================
# relight: run
# ==============================================================================================  
def run_relight(args, mesh, views, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, lgt_list, images_save_path):
    ## ============== deform ==============================     
    shapedirs, posedirs, lbs_weights = deformer_net.query_weights(mesh.vertices)
    eval_vertices = mesh.vertices
    batched_verts = eval_vertices.unsqueeze(0).repeat(views["img"].shape[0], 1, 1)

    _, pose_features, transformations = FLAMEServer(expression_params=views["flame_expression"], full_pose=views["flame_pose"])
    if args.ghostbone:
        transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(views["img"].shape[0], -1, -1, -1).float().to(device), transformations], 1)
    deformed_vertices = FLAMEServer.forward_pts_batch(pnts_c=batched_verts, betas=views["flame_expression"], transformations=transformations, pose_feature=pose_features, 
                                        shapedirs=shapedirs, posedirs=posedirs, lbs_weights=lbs_weights, dtype=torch.float32, map2_flame_original=True)

    d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)
    ## ============== Rasterize ==============================
    gbuffers = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                        channels=channels_gbuffer, with_antialiasing=True, 
                        canonical_v=mesh.vertices, canonical_idx=mesh.indices)
    
    ## ============== predict color ==============================
    relit_imgs, cbuffers, gbuffer_mask = shader.relight(gbuffers, views, mesh, args.finetune_color, lgt_list)
    save_relit_intrinsic_materials(relit_imgs, views_subset, gbuffer_mask, cbuffers, images_save_path)

# ==============================================================================================
# evaluation: numbers
# ==============================================================================================  
def quantitative_eval(args, mesh, dataloader_validate, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer,
                        experiment_dir, images_eval_save_path, lgt=None, save_each=False):

    for it, views_subset in enumerate(dataloader_validate):
        with torch.no_grad():
            rgb_pred, gbuffer, cbuffer = run(args, mesh, views_subset, FLAMEServer, deformer_net, shader, renderer, device, 
                    channels_gbuffer, lgt=lgt)

        rgb_pred = rgb_pred * gbuffer["mask"]
        if save_each:
            save_individual_img(rgb_pred, views_subset, gbuffer["normal"], gbuffer["mask"], cbuffer, images_eval_save_path)

    ## ============== metrics ==============================
    gt_dir = Path(args.input_dir)
    if gt_dir is not None:
        eval_list = metrics.run(images_eval_save_path, gt_dir, args.eval_dir)

    with open(str(experiment_dir / "final_eval.txt"), 'a') as f:
        f.writelines("\n"+"w/o cloth result:"+"\n")
        f.writelines("\n"+"MAE | LPIPS | SSIM | PSNR"+"\n")
        if gt_dir is not None:
            eval_list = [str(e) for e in eval_list]
            f.writelines(" ".join(eval_list))
            
if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    # Create directories
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)
    flame_path = args.working_dir / 'flame/FLAME2020/generic_model.pkl'

    # ==============================================================================================
    # Create evalables: FLAME + Renderer + Views + Downsample
    # ==============================================================================================

    ### Read the views
    print("loading test views...")
    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=args.sample_idx_ratio, pre_load=True)
    dataloader_validate = torch.utils.data.DataLoader(dataset_val, batch_size=4, collate_fn=dataset_val.collate, shuffle=False)

    ### init flame and deformation
    flame_shape = dataset_val.shape_params
    FLAMEServer = FLAME(flame_path, n_shape=100, n_exp=50, shape_params=flame_shape).to(device)

    ### Obtain the initial mesh and compute its connectivity
    flame_canonical_mesh = Mesh(FLAMEServer.v_template, FLAMEServer.faces_tensor, device=device)
    flame_canonical_mesh.compute_connectivity()

    ### create bounding box from the mesh vertices
    aabb = AABB(flame_canonical_mesh.vertices.cpu().numpy())
    flame_mesh_aabb = [torch.min(flame_canonical_mesh.vertices, dim=0).values, torch.max(flame_canonical_mesh.vertices, dim=0).values]

    # init mesh is mouth open!!!
    FLAMEServer.canonical_exp = dataset_val.get_mean_expression_train(args.train_dir).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    FLAMEServer.canonical_verts, FLAMEServer.canonical_pose_feature, FLAMEServer.canonical_transformations = \
        FLAMEServer(expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose)
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)
    flame_canonical_mesh.vertices = FLAMEServer.canonical_verts.squeeze(0)

    # ==============================================================================================
    # mesh
    # ==============================================================================================
    
    mesh_path = Path(experiment_dir / "stage_2" / "meshes" / f"mesh_latest.obj")
    mesh = read_mesh(mesh_path, device=device)
    mesh.compute_connectivity()
    mesh.to(device)

    print("loaded mesh")
    # ==============================================================================================
    # Rendererrr
    # ==============================================================================================

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_val, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)

    # ==============================================================================================
    # deformation 
    # ==============================================================================================

    load_deformer = Path(experiment_dir / "stage_2" / "network_weights" / f"deformer_latest.pt")
    assert os.path.exists(load_deformer)

    multires = 0
    deformer_net = get_deformer_network(FLAMEServer, model_path=load_deformer, train=False, d_in=3, dims=[128, 128, 128, 128], 
                                           weight_norm=True, multires=multires, num_exp=50, aabb=aabb, ghostbone=args.ghostbone, device=device)
    if args.ghostbone:
        FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device), FLAMEServer.canonical_transformations], 1)
    # ==============================================================================================
    # shading
    # ==============================================================================================

    load_shader = Path(experiment_dir / "stage_2" / "network_weights" / f"shader_latest.pt")
    assert os.path.exists(load_shader)

    shader = NeuralShader.load(load_shader, device=device)

    lgt = light.create_env_rnd()    

    print("=="*50)
    shader.eval()
    deformer_net.eval()

    batch_size = args.batch_size
    print("Batch Size:", batch_size)
    
    # ==============================================================================================
    # evaluation: intrinsic materials and relighting
    # ==============================================================================================  
    lgt_list = light.load_target_cubemaps(args.working_dir)
    for i in range(len(lgt_list)):
        Path(images_eval_save_path / "qualitative_results" / f"env_map_{i}" ).mkdir(parents=True, exist_ok=True)

    for it, views_subset in enumerate(dataloader_validate):
        with torch.no_grad():
            run_relight(args, mesh, views_subset, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, lgt_list, images_eval_save_path / "qualitative_results")
            
    # # ==============================================================================================
    # # evaluation: qualitative and quantitative - animation
    # # ==============================================================================================  
    quantitative_eval(args, mesh, dataloader_validate, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, experiment_dir
                    , images_eval_save_path  / "qualitative_results", lgt=lgt, save_each=True)