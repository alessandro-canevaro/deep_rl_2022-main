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

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict, state, seq_lens):
        #print(input_dict)
        obs = input_dict["obs"] #obs.shape = (B, 48)
        obs = obs.reshape(obs.shape[0], 4*4*3)
        assert len(obs.shape) == 2, "dimensions are not 2, {}".format(obs.shape)
        B = obs.shape[0] #batch size
        V_ = torch.zeros((B, ))
        #print("SELF V_ and B", V_, B)
        #print("size.V_, shape", self.V_.shape)

        # Store last batch size for value_function output.
        self._last_batch_size = B

        #V = self.VIP(obs)
        phi_out = self.Phi(obs).reshape((obs.shape[0], self.maze_h, self.maze_w, 3))
        V = self.VIP_diff(phi_out, K=20)
        #V_old = self.VIP_old(phi_out, K=20)
        #for b_idx in range(B):
        #    assert torch.allclose(V[b_idx], V_old[b_idx], atol=0.0001), f"V != Vold at b_idx={b_idx}: \n{V[b_idx]}, \n{V_old[b_idx]}"

        logit = torch.zeros((B, self.action_space.n))
        obs = obs.reshape(B, self.maze_h, self.maze_w, 3)

        for b_idx in range(B): 

            if not torch.any(obs[b_idx, :, :, 1]):#.sum().item() == 0: if all zeros
                i, j = 1, 1
                goal_i, goal_j = 0, 0
            else:
                i, j = torch.nonzero(obs[b_idx, :, :, 1])[0] #the [0] is because nonzero is returning a tuple of indexes
                goal_i, goal_j = torch.nonzero(obs[b_idx, :, :, 2])[0]
                #assert False, f"obs {obs[b_idx][:, :, 2]} {obs[b_idx].swapaxes(2, 1).swapaxes(1, 0)[2, :, :]}, {obs[b_idx].reshape((3, 4, 4))[2, :, :]}"
                #a = torch.Tensor(list(range(48))).reshape((3, 4, 4))
                #assert False, f"a {a[1, :, :]}, {a.reshape((4, 4, 3))[:, :, 1]}, {a.swapaxes(0, 1).swapaxes(1, 2)[:, :, 1]}"
            
            #assert goal_i == 3 and goal_j == 0, "Goal is not in top left corner. It is in {}, {}".format(goal_i, goal_j)
            assert V_.shape[0] > b_idx, f"trying to access element {b_idx} of V_, but V_ has size {V_.shape[0]}, {V_}, {B}, {obs.shape}"
            V_[b_idx] = V[b_idx][i, j]


            sub_state = torch.nn.functional.pad(obs[b_idx, :, :, :], (0, 0, 1, 1, 1, 1), "constant", 0)[i-1+1:i+2+1, j-1+1:j+2+1, :]
            sub_v = self.padder(V[b_idx])[i-1+1:i+2+1, j-1+1:j+2+1]
            logit_input = torch.concatenate((sub_state.flatten(), sub_v.flatten()))

            assert logit_input.shape[0] == 3*3*3 + 3*3, f"Logit input size is {logit_input.shape[0]} (expected 36)"
            logit[b_idx] = self.Logit(logit_input)
            
            #logit[b_idx] = self.Logit(obs[b_idx].flatten())

        #print("logit.shape", logit.shape)
        self.V_ = V_
        return logit, []

    def VIP_diff(self, phi_out, K=20):
        """
        Parallel and differentiable version of VIP function
        obs has shape (B, 48)
        """
        final_v = []
        # rewards in , out and probabilities
        for obs_idx in range(phi_out.shape[0]):
            #assert False, f"obs {obs[0][:, :, 2]}"
            rin, rout, p = phi_out[obs_idx, :, :, 0], phi_out[obs_idx, :, :, 1], phi_out[obs_idx, :, :, 2]
            
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
            final_v.append(v)

        return final_v

    def VIP_old(self, phi_out, K=20):
        final_v = []
        # rewards in , out and probabilities
        for obs_idx in range(phi_out.shape[0]):
            rin, rout, p = phi_out[obs_idx, :, :, 0], phi_out[obs_idx, :, :, 1], phi_out[obs_idx, :, :, 2]
            
            h, w = 4, 4#obs.shape[0], obs.shape[1]
            v = torch.zeros((h, w, K+1), dtype=torch.float32)#self.value_function()
            for k in range(K):
                for i in range(h):
                    for j in range(w):
                        v[i, j, k+1] = v[i,j,k]
                        for di, dj in [ (-1, 0), (1, 0), (0, -1), (0, 1)]:
                            if di + i < 0 or dj + j < 0:
                                continue
                            if di + i >= h or dj + j >= w:
                                continue
                            ip = i + di
                            jp = j + dj
                            nv = p[i,j] * v[ip, jp,k] + rin[ip, jp] - rout[i,j]
                            #print("nv", obs_idx, i, j, nv)
                            v[i,j,k+1] = max( v[i,j,k+1], nv)
            #print("v.shape",v.shape)
            vv = v[:,:,min(K, v.shape[2]-1)]
            #v_out = vv.copy()
            #for i,j in vv:
            #    v_out[i,j] = vv[j, i] # annoying transpose
            #print("v_out.shape",v_out.shape)
            #vv = torch.from_numpy(vv)
            #vv.requires_grad=True
            final_v.append(vv)

        return final_v


    def value_function(self):
        return self.V_ #torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))
