# FLARE: Fast Learning of Animatable and Relightable Mesh Avatars

## [Paper](https://drive.google.com/file/d/122xvzW7cGoInDZCzkmN1NkxESV4wVw1h/view?usp=sharing) | [Project Page](https://flare.is.tue.mpg.de/)

Official repository for SIGGRAPH Asia 2023 (Journal tack) paper [*FLARE: Fast Learning of Animatable and Relightable Mesh Avatars*](https://drive.google.com/file/d/122xvzW7cGoInDZCzkmN1NkxESV4wVw1h/view?usp=sharing)

## Citation
If you find our code or paper useful, please cite as:

```
@article{bharadwaj2023flare,
  title = {{FLARE}: Fast learning of Animatable and Relightable Mesh Avatars},
  author = {Bharadwaj, Shrisha and Zheng, Yufeng and Hilliges, Otmar and Black, Michael J. and Abrevaya, Victoria Fernandez},
  journal = {ACM Transactions on Graphics},
  volume = {42},
  pages = {15},
  month = dec,
  year = {2023},
  doi = {https://doi.org/10.1145/3618401},
  month_numeric = {12}
}
```

## Getting Started

Clone the repository:
```
git clone https://github.com/sbharadwajj/flare
cd flare
```

- Download [FLAME model](https://flame.is.tue.mpg.de/download.php), choose **FLAME 2020** and unzip it, copy 'generic_model.pkl' into `./flame/FLAME2020`

## Environment
- create a conda environment and install pytorch and pytorch3d as follows:
```
conda create -n flare python=3.9
conda activate flare
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

- install `nvdiffrast` and `tinycudann` as follows: 
Note that the NVIDIA GPU architecture of your specific GPU must be set before building tiny-cuda-nn. 

This code is tested on a single Nvidia 80GB A100 GPU and NVIDIA RTX A5000 24 GB, both of which have NVIDIA GPU architecture `sm_80`. We used cuda 11.7 and cudnn 8.4.1.
```
pip install ninja imageio PyOpenGL glfw xatlas gdown
pip install git+https://github.com/NVlabs/nvdiffrast/
export TCNN_CUDA_ARCHITECTURES="70;75;80" 
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/gcc-9'
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
imageio_download_bin freeimage
pip install gpytoolbox opencv-python trimesh matplotlib chumpy lpips tqdm
```

## Dataset

We follow the same data format and preprocessing used by IMavatar. Some of the preprocessed subjects can be found [here]().

Please refer to this [section](https://github.com/zhengyuf/IMavatar#preparing-dataset) to preprocess your own data. 
Note that, we follow OpenGL format for the camera and the conversion directly takes place while training. 

## Training and Evaluation

Config file:
- `input_dir`: set the path to the dataset folder
- `working_dir`: path to the code base 
- `output_dir`: path to save the outputs
- set CUDA_HOME path

### Training

```
python train.py  --config configs/001.txt
```
### Testing

The test code saves qualitative results of the intrinsic materials, performs quantitative evaluation once again (the train script is self contained and the final metrical evaluations are saved after training) and relit+animated results according to the `eval_dir`. Additional environment maps can be added in `assets/env_maps` folder. 

```
python test.py  --config configs/001.txt
```


Please refer to the config files to tweak individual arguments:
- `downsample`: downsamples the mesh before training. In the final paper, we do not downsample (and that is the default argument), but to additionally improve the results, this argument can be used
- `upsample_iterations`: For the final paper, we upsample once at 500th iteration. But an additional upsampling step can be added at 1000th iteration, if the mesh is initially downsampled. Upsampling the mesh improves small details, but is also suseptible to high frequency artifacts if overused. 
- `sample_idx_ratio`: Default is 1 i.e samples all the images. But for faster debugging cycles, it can be set to an arbitrary nth value (e.g. 6) to sample only every nth (6th) image uniformly.

## Evaluation

## GPU requirement
We train our models with a single Nvidia 80GB A100 GPU. It is possible to train on GPU's with less memory (e.g. 24 GB) by reducing the batch size. 

## License

This code and model are available for non-commercial scientific research purposes as defined in the LICENSE file. By downloading and using the code and model you agree to the terms in the LICENSE.

## Acknowledgements
For functions or scripts that are based on external sources, we acknowledge the origin individually in each file. 
But we specifically benefit a lot from [Nvdiffrec](https://github.com/NVlabs/nvdiffrec). Please consider citing their work if you find ours helpful [bibtex](https://github.com/NVlabs/nvdiffrec#citation). 

Other repositories that have been helpful: 
- [IMavatar](https://github.com/zhengyuf/IMavatar)
- [Neural deferred shading](https://github.com/fraunhoferhhi/neural-deferred-shading)
- [DECA](https://github.com/yfeng95/DECA)
- [FLAME](https://github.com/soubhiksanyal/FLAME_PyTorch).

