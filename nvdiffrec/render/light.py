# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.
#
# CODE MODIFIED/ADAPTED BY SHRISHA BHARADWAJ

import os
import numpy as np
import torch
import nvdiffrast.torch as dr
from . import util
from . import renderutils as ru
import torchvision.utils as torch_utils
from PIL import Image
from pathlib import Path
import math
######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################  
class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(EnvironmentLight, self).__init__()
        self.mtx = None
        if base is not None:
            # self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
            # self.base = base
            # self.env_base = base
            # print("requires grad false")    
            self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
            self.register_parameter('env_base', self.base)
        else:
            self.base = None
    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)


    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def bin_roughness(self, roughness, level_bin=5):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (level_bin - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + level_bin - 2)
    
    def shading_pbr(self, gb_pos, gb_normal, kd, roughness, view_pos, ko, canon_norm, fresnel_constant):
        batch_size = gb_pos.shape[0]
        wo = util.safe_normalize(view_pos - gb_pos)

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)
       
        # Diffuse lookup (supports batching)
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        shaded_col = diffuse * kd

        # Lookup FG term from lookup texture
        NdotV = torch.clamp(util.dot(wo, canon_norm), min=1e-4)
        fg_uv = torch.cat((NdotV, roughness), dim=-1)
        if not hasattr(self, '_FG_LUT'):
            self._FG_LUT = torch.as_tensor(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
        fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

        # Roughness adjusted specular env lookup
        miplevel = self.get_mip(roughness)
        spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

        # Compute aggregate lighting
        reflectance = fresnel_constant * fg_lookup[...,0:1] + fg_lookup[...,1:2]

        if ko is not None:
            shaded_col += ko * spec * reflectance
        else:
            shaded_col += spec * reflectance

        buffers = {
                    'k_r': roughness,
                    'specu': spec,
                    'reflec': reflectance,
                    'shading':diffuse,
                    }
        
        if ko is not None:
            buffers['ko'] = ko
        
        return shaded_col, buffers # Modulate by hemisphere visibility
    
    def shade_pbr_ipe(self, gb_pos, diffuse, enc_func, shading_net, gb_normal, kd, kr, view_pos, ko, canon_norm, fresnel_constant):
        bz, h, w, ch = gb_pos.shape
        roughness = kr

        # color
        shaded_col = diffuse * kd

        # reflec vector
        wo = util.safe_normalize(view_pos - gb_pos)
        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))        
        enc_reflvec_kr = enc_func(reflvec.view(-1, 3), roughness.view(-1, 1))
        spec = shading_net(enc_reflvec_kr)
        spec = spec.view(bz, h, w, 3)

        # Lookup FG term from lookup texture
        NdotV = torch.clamp(util.dot(wo, canon_norm), min=1e-4)
        fg_uv = torch.cat((NdotV, roughness), dim=-1)
        if not hasattr(self, '_FG_LUT'):
            self._FG_LUT = torch.as_tensor(np.fromfile('assets/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
        ## self._FG_LUT is (256, 256, 2) and we sample (wo * n_d) and roughness from here
        fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

        # # Compute aggregate lighting
        # ## 0.04 * (look up of wo * n_d) + (look up of roughness)
        reflectance = fresnel_constant * fg_lookup[...,0:1] + fg_lookup[...,1:2]

        if ko is not None:
            shaded_col += ko * spec * reflectance
        else:
            shaded_col += spec * reflectance

        buffers = {'k_r': roughness,
                    'specu': spec,
                    'reflec': reflectance,
                    'shading': diffuse,
                    }
        
        if ko is not None:
            buffers['ko'] = ko
            
        return shaded_col, buffers # Modulate by hemisphere visibility
        
    def shade_diffuse(self, gb_normal, kd):
        
        # Diffuse lookup (supports batching)
        diffuse = dr.texture(self.diffuse[None, ...], gb_normal.contiguous(), filter_mode='linear', boundary_mode='cube')
        shaded_col = diffuse #* kd

        return shaded_col # Modulate by hemisphere visibility


######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0, dim=512):
    latlong_img = torch.tensor(util.reinhard(util.load_image(fn)), dtype=torch.float32, device='cuda')*scale
    cubemap = util.latlong_to_cubemap(latlong_img, [dim, dim])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

def load_env(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() == ".hdr":
        return _load_env_hdr(fn, scale)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def load_target_cubemaps(working_dir):      

    scale_fac = 2.0

    
    lgt_list = []

    env_map_path = working_dir / "assets" / "env_maps"
    cube_maps = env_map_path.glob("*.hdr")

    print("=="*50)
    print("loading the following environment maps:")
    for cube_map in cube_maps:
        try:
            env_map = _load_env_hdr(cube_map, scale=scale_fac, dim=512)
            lgt_list.append(env_map)
            print(cube_map)
        except ValueError:
            print("Please provide a valid environment map")
            continue
    print("=="*50)

    return lgt_list

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    return EnvironmentLight(base)

def create_env_rnd():
    return EnvironmentLight(base=None)
