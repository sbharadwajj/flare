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
import numpy as np
from pathlib import Path
from gpytoolbox import remesh_botsch
import torch
from tqdm import tqdm
from flame.FLAME import FLAME
from flare.dataset import *
from flare.dataset import dataset_util

from flare.core import (
    Mesh, Renderer
)
from flare.losses import *
from flare.modules import (
    NeuralShader, get_deformer_network, Displacement
)
from flare.utils import (
    AABB, read_mesh, write_mesh,
    visualize_training,
    make_dirs, set_defaults_finetune
)
import nvdiffrec.render.light as light
from test import run, quantitative_eval

import time

def main(args, device, dataset_train, dataloader_train, debug_views, FLAMEServer):
    ## ============== Dir ==============================
    run_name = args.run_name if args.run_name is not None else args.input_dir.parent.name
    images_save_path, images_eval_save_path, meshes_save_path, shaders_save_path, experiment_dir = make_dirs(args, run_name, args.finetune_color)

    ## ============== load mesh/train mesh ==============================
    if args.finetune_color:
        mesh_path = experiment_dir / "stage_1" / "meshes" / f"mesh_latest.obj"
        print("loading mesh from:", mesh_path)
        flame_canonical_mesh = read_mesh(mesh_path, device=device)
        flame_canonical_mesh.compute_connectivity()
        flame_canonical_mesh.to(device)
    else:
        if args.downsample:
            v_down, f_down = remesh_botsch(FLAMEServer.canonical_verts.squeeze(0).cpu().detach().numpy().astype(np.float64), 
                                                                    FLAMEServer.faces_tensor.cpu().numpy().astype(np.int32), h=float(args.downsample_ratio))
            verts = np.ascontiguousarray(v_down)
            faces = np.ascontiguousarray(f_down)
            print("Downsampled:", verts.shape, faces.shape)
        else:
            verts = FLAMEServer.canonical_verts.squeeze(0)
            faces = FLAMEServer.faces_tensor

        flame_canonical_mesh: Mesh = None
        flame_canonical_mesh = Mesh(verts, faces, device=device)
        flame_canonical_mesh.compute_connectivity()
        write_mesh(Path(meshes_save_path / "init_mesh.obj"), flame_canonical_mesh.to('cpu'))

    ## ============== renderer ==============================
    aabb = AABB(flame_canonical_mesh.vertices.cpu().numpy())
    flame_mesh_aabb = [torch.min(flame_canonical_mesh.vertices, dim=0).values, torch.max(flame_canonical_mesh.vertices, dim=0).values]

    renderer = Renderer(device=device)
    renderer.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)
    channels_gbuffer = ['mask', 'position', 'normal', "canonical_position"]
    print("Rasterizing:", channels_gbuffer)
    
    renderer_visualization = Renderer(device=device)
    renderer_visualization.set_near_far(dataset_train, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)

    # ==============================================================================================
    # vertices
    # ==============================================================================================

    lr_vertices = args.lr_vertices
    displacements = Displacement(vertices_shape=flame_canonical_mesh.vertices.shape)
    
    displacements.to(device=device)
    optimizer_vertices = torch.optim.Adam(list(displacements.parameters()), lr=lr_vertices)

    # ==============================================================================================
    # deformation 
    # ==============================================================================================
    if args.train_deformer:
        model_path = None
        print("=="*50)
        print("Training Deformer")
    else:
        print("=="*50)
        print("Loading deformer network trained in the previous stage")
        args.weight_flame_regularization = 0.0

        model_path = Path(experiment_dir / "stage_1" / "network_weights" / f"deformer_latest.pt")
        assert os.path.exists(model_path)

    deformer_net = get_deformer_network(FLAMEServer, model_path=model_path, train=args.train_deformer, d_in=3, dims=args.deform_dims, 
                                           weight_norm=True, multires=0, num_exp=50, aabb=flame_mesh_aabb, ghostbone=args.ghostbone, device=device)
    if args.train_deformer:
        optimizer_deformer = torch.optim.Adam(list(deformer_net.parameters()), lr=args.lr_deformer)

    # ==============================================================================================
    # shading
    # ==============================================================================================

    lgt = light.create_env_rnd()    
    disentangle_network_params = {
        "material_mlp_ch": args.material_mlp_ch,
        "light_mlp_ch":args.light_mlp_ch,
        "material_mlp_dims":args.material_mlp_dims,
        "light_mlp_dims":args.light_mlp_dims
    }

    # Create the optimizer for the neural shader
    shader = NeuralShader(fourier_features=args.fourier_features,
                          activation=args.activation,
                          last_activation=torch.nn.Sigmoid(), 
                          disentangle_network_params=disentangle_network_params,
                          bsdf=args.bsdf,
                          aabb=flame_mesh_aabb,
                          device=device)
    params = list(shader.parameters()) 

    if args.weight_albedo_regularization > 0:
        from robust_loss_pytorch.adaptive import AdaptiveLossFunction
        _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
        params += list(_adaptive.parameters()) ## need to train it

    optimizer_shader = torch.optim.Adam(params, lr=args.lr_shader)

    # ==============================================================================================
    # Loss Functions
    # ==============================================================================================
    # Initialize the loss weights and losses
    loss_weights = {
        "mask": args.weight_mask,
        "normal": args.weight_normal,
        "laplacian": args.weight_laplacian,
        "shading": args.weight_shading,
        "perceptual_loss": args.weight_perceptual_loss,
        "albedo_regularization": args.weight_albedo_regularization,
        "roughness_regularization": args.weight_roughness_regularization,
        "white_light_regularization": args.weight_white_lgt_regularization,
        "fresnel_coeff": args.weight_fresnel_coeff
    }

    if args.train_deformer:
        loss_weights["flame_regularization"] = 1.0 # we use the weight directly in loss function
    else:
        loss_weights["flame_regularization"] = 0.0

    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}
    print(loss_weights)
    if loss_weights["perceptual_loss"] > 0.0:
        VGGloss = VGGPerceptualLoss().to(device)

    print("=="*50)
    shader.train()
    if args.train_deformer:
        deformer_net.train()
    displacements.train()
    print("Batch Size:", args.batch_size)
    print("=="*50)

    # ==============================================================================================
    # T R A I N I N G
    # ==============================================================================================
    epochs = (args.iterations // len(dataloader_train)) + 1
    iteration = 0

    progress_bar = tqdm(range(epochs))
    start = time.time()
    for epoch in progress_bar:
        for iter_, views_subset in enumerate(dataloader_train):
            iteration += 1
            progress_bar.set_description(desc=f'Epoch {epoch}, Iter {iteration}')
            
            # ==============================================================================================
            # upsample + remesh + reduce lr + freeze if required
            # ==============================================================================================
            if iteration in args.upsample_iterations and not args.finetune_color:
                print("=="*50)
                print("Upsampling at iteration:", iteration)
                # Upsample the mesh by remeshing the surface with half the average edge length
                e0, e1 = mesh.edges.unbind(1)

                average_edge_length = torch.linalg.norm(canonical_offset_vertices[e0] - canonical_offset_vertices[e1], dim=-1).mean()
                v_upsampled, f_upsampled = remesh_botsch(canonical_offset_vertices.cpu().detach().numpy().astype(np.float64), 
                                                        mesh.indices.cpu().numpy().astype(np.int32), h=float(average_edge_length/1.5))
                v_upsampled = np.ascontiguousarray(v_upsampled)
                f_upsampled = np.ascontiguousarray(f_upsampled)
                flame_canonical_mesh = Mesh(v_upsampled, f_upsampled, device=device)
                flame_canonical_mesh.compute_connectivity()

                print("Vertices:", v_upsampled.shape)
                print("Faces:", f_upsampled.shape)
                del v_upsampled, f_upsampled
                if iteration == args.upsample_iterations[0]:
                    lr_vertices *= 0.75
                    # Adjust weights and step size
                    loss_weights['laplacian'] *= 4
                    loss_weights['normal'] *= 4
                print("laplacian weight", loss_weights['laplacian'])
                print("normal consistency weight", loss_weights['normal'])
                print("lr vertices", lr_vertices)

                displacements.register_parameter('vertex_offsets', torch.nn.Parameter(torch.zeros(flame_canonical_mesh.vertices.shape), requires_grad=True))
                displacements.canonical_vertices = flame_canonical_mesh.vertices
                displacements.vertices_shape = flame_canonical_mesh.vertices.shape
                displacements.to(device=device)
                optimizer_vertices = torch.optim.Adam(list(displacements.parameters()), lr=lr_vertices)
                print("=="*50)

            # ==============================================================================================
            # update/displace vertices
            # ==============================================================================================
            v_off = displacements()
            canonical_offset_vertices = flame_canonical_mesh.vertices + v_off
            mesh = flame_canonical_mesh.with_vertices(canonical_offset_vertices)

            # ==============================================================================================
            # deformation of canonical mesh
            # ==============================================================================================      
            shapedirs, posedirs, lbs_weights = deformer_net.query_weights(mesh.vertices)
            
            batched_verts = mesh.vertices.unsqueeze(0).repeat(args.batch_size, 1, 1)
            _, pose_features, transformations = FLAMEServer(expression_params=views_subset["flame_expression"], full_pose=views_subset["flame_pose"])
            if args.ghostbone:
                transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).expand(args.batch_size, -1, -1, -1).float().to(device), transformations], 1)
            deformed_vertices = FLAMEServer.forward_pts_batch(pnts_c=batched_verts, betas=views_subset["flame_expression"], transformations=transformations, pose_feature=pose_features, 
                                                shapedirs=shapedirs, posedirs=posedirs, lbs_weights=lbs_weights, dtype=torch.float32, map2_flame_original=True)
            d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)

            # ==============================================================================================
            # R A S T E R I Z A T I O N
            # ==============================================================================================
            gbuffers = renderer.render_batch(views_subset['camera'], deformed_vertices.contiguous(), d_normals, 
                                    channels=channels_gbuffer, with_antialiasing=True, 
                                    canonical_v=mesh.vertices, canonical_idx=mesh.indices) 
            
            # ==============================================================================================
            # loss function 
            # ==============================================================================================
            ## ============== geometry regularization ==============================
            losses['normal'] = normal_consistency_loss(mesh)
            losses['laplacian'] = laplacian_loss(mesh)

            ## ============== color + regularization for color ==============================
            pred_color_masked, cbuffers, gbuffer_mask = shader.shade(gbuffers, views_subset, mesh, args.finetune_color, lgt)

            losses['shading'], pred_color, tonemapped_colors = shading_loss_batch(pred_color_masked, views_subset, args.batch_size)
            losses['perceptual_loss'] = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)
            
            losses['mask'] = mask_loss(views_subset["mask"], gbuffer_mask)

            ## ======= regularization color ========
            losses['albedo_regularization'] = albedo_regularization(_adaptive, shader, mesh, device, displacements, iteration)
            losses['white_light_regularization'] = white_light(cbuffers)
            losses['roughness_regularization'] = roughness_regularization(cbuffers["roughness"], views_subset["skin_mask"], views_subset["mask"], r_mean=args.r_mean)
            losses["fresnel_coeff"] = spec_intensity_regularization(cbuffers["ko"], views_subset["skin_mask"], views_subset["mask"])
            
            ## ============== flame regularization ==============================
            if loss_weights['flame_regularization'] > 0:
                losses['flame_regularization'], gt_nn = flame_regularization(FLAMEServer, lbs_weights, shapedirs, posedirs, mesh.vertices, args.ghostbone, 
                                                                      iteration, args.flame_mask, views_subset=views_subset, gbuffer=gbuffers, 
                                                                      weight_lbs=args.weight_flame_regularization)
            
                if iteration in args.decay_flame:
                    print("Decaying flame regularization")
                    loss_weights['flame_regularization'] *= 0.5

            loss = torch.tensor(0., device=device) 
            for k, v in losses.items():
                loss += v * loss_weights[k]

            # ==============================================================================================
            # Optimizer step
            # ==============================================================================================
            optimizer_shader.zero_grad()
            optimizer_vertices.zero_grad()
            if args.train_deformer:
                optimizer_deformer.zero_grad()

            loss.backward()
            torch.cuda.synchronize()

            ### increase the gradients of positional encoding following tinycudnn
            if args.grad_scale and args.fourier_features == "hashgrid":
                shader.fourier_feature_transform.params.grad /= 8.0

            optimizer_shader.step()
            optimizer_vertices.step()
            if args.train_deformer:
                optimizer_deformer.step()

            progress_bar.set_postfix({'loss': loss.detach().cpu().item()})

            # ==============================================================================================
            # warning: check if light mlp diverged
            # ==============================================================================================
            '''
            We do not use an activation function for the output layer of light MLP because we are learning in sRGB space where the values 
            are not restricted between 0 and 1. As a result, the light MLP diverges sometimes and predicts only zero values. 
            Hence, we have included the try and catch block to automatically restart the training during this case. 
            '''
            if iteration == 100:
                convert_uint = lambda x: torch.from_numpy(np.clip(np.rint(dataset_util.rgb_to_srgb(x).detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)).to(device)
                try:
                    diffuse_shading = convert_uint(cbuffers["shading"])
                    specular_shading = convert_uint(cbuffers["specu"])
                    if torch.count_nonzero(diffuse_shading) == 0 or torch.count_nonzero(specular_shading) == 0:
                        raise ValueError("All values predicted from light MLP are zero")
                except ValueError as e:
                    print(f"Error: {e}")
                    raise  # Raise the exception to exit the current execution of main()
            
            # ==============================================================================================
            # V I S U A L I Z A T I O N S
            # ==============================================================================================
            if (args.visualization_frequency > 0) and (iteration == 1 or iteration % args.visualization_frequency == 0):
            
                with torch.no_grad():
                    debug_rgb_pred, debug_gbuffer, debug_cbuffers = run(args, mesh, debug_views, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, lgt)
                    ## ============== visualize ==============================
                    visualize_training(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, images_save_path, iteration)
                    del debug_gbuffer, debug_cbuffers

            ## ============== save intermediate ==============================
            if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
                with torch.no_grad():
                    write_mesh(meshes_save_path / f"mesh_{iteration:06d}.obj", mesh.detach().to('cpu'))                                
                    shader.save(shaders_save_path / f'shader_{iteration:06d}.pt')
                    displacements.save(shaders_save_path / f'displacement_{iteration:06d}.pt')
                    deformer_net.save(shaders_save_path / f'deformer_{iteration:06d}.pt')

    end = time.time()
    total_time = ((end - start) % 3600)
    print("TIME TAKEN (mins):", int(total_time // 60))
    # ==============================================================================================
    # s a v e
    # ==============================================================================================
    with open(experiment_dir / "args.txt", "w") as text_file:
        print(f"{args}", file=text_file)
    write_mesh(meshes_save_path / f"mesh_latest.obj", mesh.detach().to('cpu'))
    shader.save(shaders_save_path / f'shader_latest.pt')
    displacements.save(shaders_save_path / f'displacement_latest.pt')
    deformer_net.save(shaders_save_path / f'deformer_latest.pt')

    # ==============================================================================================
    # FINAL: qualitative and quantitative results
    # ==============================================================================================
    if args.finetune_color:        
        ## ============== free memory before evaluation ==============================
        del dataset_train, dataloader_train, debug_views, views_subset

        print("=="*50)
        print("E V A L U A T I O N")
        print("=="*50)
        dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=1, pre_load=True)
        dataloader_validate = torch.utils.data.DataLoader(dataset_val, batch_size=4, collate_fn=dataset_val.collate)

        quantitative_eval(args, mesh, dataloader_validate, FLAMEServer, deformer_net, shader, renderer, device, channels_gbuffer, experiment_dir
                        , images_eval_save_path / "qualitative_results", lgt=lgt, save_each=True)

if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    print(f"Using device {device}")

    # ==============================================================================================
    # load data
    # ==============================================================================================
    print("loading train views...")
    dataset_train    = DatasetLoader(args, train_dir=args.train_dir, sample_ratio=args.sample_idx_ratio, pre_load=True)
    dataset_val      = DatasetLoader(args, train_dir=args.eval_dir, sample_ratio=24, pre_load=True)
    dataloader_train    = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=dataset_train.collate, shuffle=True, drop_last=True)
    view_indices = np.array(args.visualization_views).astype(int)
    d_l = [dataset_val.__getitem__(idx) for idx in view_indices[2:]]
    d_l.append(dataset_train.__getitem__(view_indices[0]))
    d_l.append(dataset_train.__getitem__(view_indices[1]))
    debug_views = dataset_val.collate(d_l)

    del dataset_val
    # ==============================================================================================
    # Create trainables: FLAME + Renderer  + Downsample
    # ==============================================================================================
    ### ============== load FLAME mesh ==============================
    flame_path = args.working_dir / 'flame/FLAME2020/generic_model.pkl'
    flame_shape = dataset_train.shape_params
    FLAMEServer = FLAME(flame_path, n_shape=100, n_exp=50, shape_params=flame_shape).to(device)

    ## ============== canonical with mouth open (jaw pose 0.4) ==============================
    FLAMEServer.canonical_exp = (dataset_train.get_mean_expression()).to(device)
    FLAMEServer.canonical_pose = FLAMEServer.canonical_pose.to(device)
    FLAMEServer.canonical_verts, FLAMEServer.canonical_pose_feature, FLAMEServer.canonical_transformations = \
        FLAMEServer(expression_params=FLAMEServer.canonical_exp, full_pose=FLAMEServer.canonical_pose)
    if args.ghostbone:
        FLAMEServer.canonical_transformations = torch.cat([torch.eye(4).unsqueeze(0).unsqueeze(0).float().to(device), FLAMEServer.canonical_transformations], 1)
    FLAMEServer.canonical_verts = FLAMEServer.canonical_verts.to(device)
    
    # ==============================================================================================
    # main run
    # ==============================================================================================
    while True:
        try:
            main(args, device, dataset_train, dataloader_train, debug_views, FLAMEServer)
            break  # Exit the loop if main() runs successfully
        except:
            print("--"*50)
            print("Re-initializing main() because the training of light MLP diverged and all the values are zero.")
            print("--"*50)

    ### ============== defaults: fine tune color ==============================
    set_defaults_finetune(args)

    while True:
        try:
            main(args, device, dataset_train, dataloader_train, debug_views, FLAMEServer)
            break  # Exit the loop if main() runs successfully
        except:
            print("--"*50)
            print("Re-initializing main() because the training of light MLP diverged and all the values are zero.")
            print("--"*50)