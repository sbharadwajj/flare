import configargparse
from pathlib import Path
import torch

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument('--run_name', type=str, default=None, help="Name of this run")
    parser.add_argument('--batch_size', type=int, default=8, help="Number of views used per iteration.")
    
    # path
    parser.add_argument('--input_dir', type=Path, default="data/yufeng/", help="Path to the input data")
    parser.add_argument('--train_dir', type=Path, nargs='+', default=["MVI_1814", "MVI_1810"], help="Path to the input data")
    parser.add_argument('--eval_dir', type=Path, nargs='+', default=["MVI_1812"], help="Path to the input data")        
    parser.add_argument('--working_dir', type=Path, default="/home/sbharadwaj/projects/neural-deferred-shading/", help="Path to the input data")
    parser.add_argument('--output_dir', type=Path, default="out", help="Path to the output directory")

    # misc
    parser.add_argument('--sample_idx_ratio', type=int, default=1, help="To sample less images (mainly for debugging purposes)")    
    parser.add_argument('--device', type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="GPU to use; -1 is CPU")
    parser.add_argument('--finetune_color', action='store_true', default=False, help="stop geometry training (in iterations)")

    # iters
    parser.add_argument('--iterations', type=int, default=2000, help="Total number of iterations")
    parser.add_argument('--final_iter', type=int, default=1500, help="Total number of iterations")
    parser.add_argument('--upsample_iterations', type=int, nargs='+', default=[500], help="Iterations at which to perform mesh upsampling")
    parser.add_argument('--save_frequency', type=int, default=300, help="Frequency of mesh and shader saving (in iterations)")
    parser.add_argument('--visualization_frequency', type=int, default=100, help="Frequency of shader visualization (in iterations)")
    parser.add_argument('--visualization_views', type=int, nargs='+', default=[15, 25, 27, 21, 26], help="Views to use for visualization.")
    parser.add_argument('--downsample', action='store_true', help="Downsample the initial flame mesh")
    parser.add_argument('--no-downsample', dest='downsample', action='store_false')
    parser.set_defaults(downsample=False)
    parser.add_argument('--downsample_ratio', type=float, default=0.03, help="downsample ratio/size")
    parser.add_argument('--grad_scale', action='store_true', help="mlp for vertex displacements")
    
    # flame
    parser.add_argument('--decay_flame', type=int, nargs='+', default=[100], help="Iterations at which to perform mesh upsampling")
    parser.add_argument('--flame_mask', action='store_true', default=False, help="Flame mask")
    # lr
    parser.add_argument('--lr_vertices', type=float, default=1e-3, help="Step size/learning rate for the vertex positions")
    parser.add_argument('--lr_shader', type=float, default=1e-3, help="Step size/learning rate for the shader parameters")
    parser.add_argument('--lr_deformer', type=float, default=1e-3, help="Step size/learning rate for the deformation parameters")

    # loss weights
    parser.add_argument('--weight_mask', type=float, default=2.0, help="Weight of the mask term")
    parser.add_argument('--weight_normal', type=float, default=0.1, help="Weight of the normal term")
    parser.add_argument('--weight_laplacian', type=float, default=60.0, help="Weight of the laplacian term")
    parser.add_argument('--weight_shading', type=float, default=1.0, help="Weight of the shading term")
    parser.add_argument('--weight_perceptual_loss', type=float, default=0.1, help="Weight of the perceptual loss")
    parser.add_argument('--weight_albedo_regularization', type=float, default=0.01, help="Weight of the albedo regularization")
    parser.add_argument('--weight_flame_regularization', type=float, default=10.0, help="Weight of the flame regularization")
    parser.add_argument('--weight_white_lgt_regularization', type=float, default=1.0, help="Weight of the white light")
    parser.add_argument('--weight_roughness_regularization', type=float, default=0.1, help="Weight of the roughness regularization")
    parser.add_argument('--weight_fresnel_coeff', type=float, default=0.01, help="Weight of the specular intensity regularization")
    parser.add_argument('--r_mean', type=float, default=0.500, help="mean roughness")


    # neural shader
    parser.add_argument('--fourier_features', type=str, default='positional', choices=(['positional', 'hashgrid']), help="Input encoding used in the neural shader")
    parser.add_argument('--activation', type=str, default='relu', choices=(['relu', 'sine']), help="Activation function used in the neural shader")
    parser.add_argument('--bsdf', type=str, default='pbr_shading', choices=(['pbr', 'pbr_shading']), help="bsdf")
    parser.add_argument('--deform_d_out', type=int, default=128, help="output layer size")
    parser.add_argument('--light_mlp_ch', type=int, default=3, help="channels for light MLP")
    parser.add_argument('--light_mlp_dims', type=int, nargs='+', default=[64, 64], help="Views to use for visualization. By default, a random view is selected each time")
    parser.add_argument('--material_mlp_dims', type=int, nargs='+', default=[128, 128, 128, 128], help="Views to use for visualization. By default, a random view is selected each time")
    parser.add_argument('--material_mlp_ch', type=int, default=4, help="channels for material MLP")   

    parser.add_argument('--ghostbone', action='store_true', help="mlp for vertex displacements")
    parser.add_argument('--no-ghostbone', dest='ghostbone', action='store_false')
    parser.set_defaults(ghostbone=True)
    parser.add_argument('--train_deformer', action='store_true', help="mlp for vertex displacements")
    parser.add_argument('--no-train_deformer', dest='train_deformer', action='store_false')
    parser.set_defaults(train_deformer=True)
    parser.add_argument('--deform_dims', type=int, nargs='+', default=[128, 128, 128, 128], help="deformer dimensions")
    return parser