## Code: https://github.com/zhengyuf/IMavatar
# Modified/Adapted by: Shrisha Bharadwaj

import torch
from flare.modules.embedder import *
import numpy as np
import torch.nn as nn
import os

class ForwardDeformer(nn.Module):
    def __init__(self,
                FLAMEServer,
                d_in,
                dims,
                multires,
                num_exp=50,
                aabb=None,
                weight_norm=True,
                ghostbone=False):
        super().__init__()

        self.FLAMEServer = FLAMEServer
        #  ============================== pose correctives, expression blendshapes and linear blend skinning weights  ==============================
        d_out = 36 * 3 + num_exp * 3 

        self.num_exp = num_exp
        dims = [d_in] + dims + [d_out]
        self.embed_fn = None
        if multires > 0:
            self.embed_fn, input_ch = get_embedder(multires)
            dims[0] = input_ch
            
        self.num_layers = len(dims)
        self.init_bones = [0, 1, 2] if not ghostbone else [0, 2, 3]  # shoulder/identity, head and jaw

        for l in range(0, self.num_layers - 2):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if multires > 0 and l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.blendshapes = nn.Linear(dims[self.num_layers - 2], d_out)
        self.skinning_linear = nn.Linear(dims[self.num_layers - 2], dims[self.num_layers - 2])
        self.skinning = nn.Linear(dims[self.num_layers - 2], 6 if ghostbone else 5)
        torch.nn.init.constant_(self.skinning_linear.bias, 0.0)
        torch.nn.init.normal_(self.skinning_linear.weight, 0.0, np.sqrt(2) / np.sqrt(dims[self.num_layers - 2]))
        if weight_norm:
            self.skinning_linear = nn.utils.weight_norm(self.skinning_linear)
        ## ==============  initialize blendshapes to be zero, and skinning weights to be equal for every bone (after softmax activation)  ==============================
        torch.nn.init.constant_(self.blendshapes.bias, 0.0)
        torch.nn.init.constant_(self.blendshapes.weight, 0.0)
        torch.nn.init.constant_(self.skinning.bias, 0.0)
        torch.nn.init.constant_(self.skinning.weight, 0.0)

        self.ghostbone = ghostbone
        self.aabb = aabb


    def query_weights(self, pnts_c):
        ## ============== normalize PE input ==============================
        if self.embed_fn is not None:
            pnts_c = (pnts_c.view(-1, 3) - self.aabb[0][None, ...]) / (self.aabb[1][None, ...] - self.aabb[0][None, ...])
            pnts_c = torch.clamp(pnts_c, min=0, max=1)
            pnts_c = self.embed_fn(pnts_c.contiguous()).to(torch.float32)

        x = pnts_c

        for l in range(0, self.num_layers - 2):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            x = self.softplus(x)

        blendshapes = self.blendshapes(x)
        posedirs = blendshapes[:, :36 * 3]
        shapedirs = blendshapes[:, 36 * 3: 36 * 3 + self.num_exp * 3]
        lbs_weights = self.skinning(self.softplus(self.skinning_linear(x)))
        lbs_weights_exp = torch.exp(20 * lbs_weights)
        lbs_weights = lbs_weights_exp / torch.sum(lbs_weights_exp, dim=-1, keepdim=True)

        return shapedirs.reshape(-1, 3, self.num_exp), posedirs.reshape(-1, 4*9, 3), lbs_weights.reshape(-1, 6 if self.ghostbone else 5)


    def save(self, path):
        data = {
            'state_dict': self.state_dict()
        }
        torch.save(data, path)  

def get_deformer_network(FLAMEServer, model_path=None, train=True, d_in=3, dims=[128, 128, 128, 128], weight_norm=True, multires=0, aabb=None, num_exp=50, ghostbone=False, device='cuda'):
    deformer_net = ForwardDeformer(FLAMEServer, d_in=d_in, dims=dims, weight_norm=weight_norm, multires=multires, aabb=aabb, num_exp=num_exp, ghostbone=ghostbone)
    deformer_net.to(device)

    if train:
        deformer_net.train()
    else:
        assert(os.path.exists(model_path))
        params = torch.load(model_path)

        deformer_net.load_state_dict(params["state_dict"])
        deformer_net.eval()
    
    return deformer_net