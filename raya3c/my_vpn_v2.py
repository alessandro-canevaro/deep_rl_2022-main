import sys, os
sys.path.append(os.path.normpath( os.path.dirname(__file__) +"/../" ))


import numpy as np
from ray import tune
from ray.tune.logger import pretty_print

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch


vin_label = "vin_network_model"


class VPNNetwork(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)
        self.num_outputs = int(np.product(self.obs_space.shape))
        self.maze_w = 4
        self.maze_h = 4

        self._last_batch_size = None
        self.Phi = torch.nn.Linear(self.num_outputs, self.num_outputs) #48, 48 #Tue set a breakpoint here 3x3x4
        self.Logit = torch.nn.Linear(3*3*3+3*3, self.action_space.n)

        self.padder = torch.nn.ZeroPad2d(1)

    def forward(self, input_dict, state, seq_lens):
        obs_flat = input_dict["obs"]
        B = obs_flat.shape[0]
        self._last_batch_size = B
        obs = obs_flat.reshape(B, self.maze_h, self.maze_w, 3)
    
        phi_out = self.Phi(obs_flat).reshape((B, self.maze_h, self.maze_w, 3))
        rin, rout, p = phi_out[:, :, :, 0], phi_out[:, :, :, 1], phi_out[:, :, :, 2]
        Values = self.VIP(rin, rout, p, K=20)
        
        obs_val = torch.cat((obs, Values.unsqueeze(dim=-1)), dim=3)
        padded_obs_val = torch.nn.functional.pad(obs_val, (0, 0, 1, 1, 1, 1, 0, 0), "constant", 0)

        pos = torch.nonzero(obs_val[:, :, :, 1])[:, 1:]
        if pos.numel() == 0:
            pos = torch.ones((B, 2), dtype=torch.int64)

        selected_obs_val = torch.zeros((B, 3, 3, 4))
        for b in range(B):
            i, j = pos[b, 0]+1, pos[b, 1]+1
            selected_obs_val[b, :, :, :] = padded_obs_val[b, i-1:i+2, j-1:j+2, :]

        logit = self.Logit(selected_obs_val.reshape((B, 3*3*4)))

        Values = Values.reshape((B, 4*4))
        pos = (pos[:, 0] * pos[:, 1]).reshape((B, 1))
        self.V_ = torch.gather(Values, 1, pos).reshape((B))
        
        return logit, []

    def VIP(self, rin_full, rout_full, p_full, K=20):
        final_v = torch.zeros((rin_full.shape[0], 4, 4))
        for obs_idx in range(rin_full.shape[0]):
            rin, rout, p = rin_full[obs_idx, :, :],rout_full[obs_idx, :, :], p_full[obs_idx, :, :]
            
            rin_padded = self.padder(rin)
            v = torch.zeros((self.maze_h, self.maze_w), dtype=torch.float32)#self.value_function()

            for __ in range(K):
                v_padded = self.padder(v)
                for h_offset, w_offset in [(0, 1), (2, 1), (1, 0), (1, 2)]:
                    v_shifted = v_padded[h_offset:h_offset+self.maze_h, w_offset:w_offset+self.maze_w]
                    rin_shifted = rin_padded[h_offset:h_offset+self.maze_h, w_offset:w_offset+self.maze_w]
                    mask = torch.eq(rin_shifted, torch.zeros((self.maze_h, self.maze_w)))
                    rout_masked = rout * (1-mask.int().float())

                    nv = p * v_shifted + rin_shifted - rout_masked
                    v = v.maximum(nv)

            #v = v.transpose(0, 1)# do we need this???? should it be a flip rows?
            final_v[obs_idx] = v

        return final_v

    def value_function(self):
        return self.V_
