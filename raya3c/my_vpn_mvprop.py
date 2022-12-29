import sys, os

sys.path.append(os.path.normpath(os.path.dirname(__file__) + "/../"))

import matplotlib.pyplot as plt
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
        self.maze_w = model_config["custom_model_config"]["maze_w"]
        self.maze_h = model_config["custom_model_config"]["maze_h"]

        self.is_train = model_config["custom_model_config"]["is_train"]
        self.vin_k = model_config["custom_model_config"]["vin_k"]

        self._last_batch_size = None
        self.Phi = torch.nn.Linear(self.num_outputs, 2*(self.maze_w*self.maze_h))  # 48, 48 #Tue set a breakpoint here 3x3x4
        
        self.Logit = torch.nn.Linear(3 * 3 * 3 + 3 * 3, 16)
        self.Logit2 = torch.nn.Linear(16, self.action_space.n)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_dict, state, seq_lens):
        obs_flat = input_dict["obs"]
        B = obs_flat.shape[0]
        self._last_batch_size = B
        obs = obs_flat.reshape(B, self.maze_h, self.maze_w, 3)
        obs_flat = obs_flat.reshape((B, self.maze_h * self.maze_w * 3))

        # VIN
        phi_out = self.sigmoid(self.Phi(obs_flat)).reshape((B, self.maze_h, self.maze_w, 2))
        r = phi_out[:, :, :, 0]
        p = phi_out[:, :, :, 1]#self.softmax(phi_out[:, :, :, 2].reshape((B, self.maze_h * self.maze_w))).reshape(
            #(B, self.maze_h, self.maze_w)
        #)
        # assert abs(p.sum()/B - 1.0) < 0.01 , f"sum = {p.sum()/B}"
        
        Values = self.MVProp(r, p, K=self.vin_k)
        #print(f"WALLS {obs[0, :, :, 0]}, AGENT {obs[0, :, :, 1]} GOAL {obs[0, :, :, 2]} Values {Values}")
        #print(f"RIN {rin}, ROUT {rout} P {p} ")#Values {Values}")
        # Policy net
        obs_val = torch.cat((obs, Values.unsqueeze(dim=-1)), dim=3)
        padded_obs_val = torch.nn.functional.pad(obs_val, (0, 0, 1, 1, 1, 1, 0, 0), "constant", 0)

        pos = torch.nonzero(obs_val[:, :, :, 1])[:, 1:]
        if pos.numel() == 0:
            pos = torch.ones((B, 2), dtype=torch.int64)
        if not self.is_train:
            goal_pos = torch.nonzero(obs_val[:, :, :, 2])[:, 1:]
            if goal_pos.numel() == 0:
                goal_pos = torch.ones((B, 2), dtype=torch.int64)

        selected_obs_val = torch.zeros((B, 3, 3, 4))
        for b in range(B):
            i, j = pos[b, 0] + 1, pos[b, 1] + 1
            selected_obs_val[b, :, :, :] = padded_obs_val[b, i - 1 : i + 2, j - 1 : j + 2, :]

        logit = self.Logit(selected_obs_val.reshape((B, 3 * 3 * 4)))
        logit = self.relu(logit)
        logit = self.Logit2(logit)

        #if not self.is_train:
        #    self.plotValues(Values, pos, goal_pos, p)

        self.my_stats = (Values.reshape((B, self.maze_h, self.maze_w))[:, :, :], pos[0, 0], pos[0, 1], p)

        # Value at the agent position
        Values = Values.reshape((B, self.maze_h * self.maze_w))
        pos = (self.maze_h *pos[:, 0] + pos[:, 1]).reshape((B, 1))
        self.V_ = torch.gather(Values, 1, pos).reshape((B))

        return logit, []

    def plotValues(self, values, agent_pos, goal_pos, p, sleep_time=2):
        # print(values.shape)
        if values.shape[0] == 1:
            p = p.reshape((self.maze_h, self.maze_w))
            p = p.detach().numpy()

            values = values.reshape((self.maze_h, self.maze_w))
            values = values.detach().numpy()

            #values = np.rot90(values, 2).T

            #print(values, agent_pos[0, 0], agent_pos[0, 1])
            # values = np.rot90(values)
            #plt.imshow(values, cmap='hot', interpolation='nearest')
            #plt.pause(0.05)
            
            plt.clf()
            heatmap = plt.pcolor(values)
            plt.text(agent_pos[0, 1] + 0.5,agent_pos[0, 0] + 0.5, 'A',
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
            plt.text(goal_pos[0, 1] + 0.5, goal_pos[0, 0] + 0.5, 'G',
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
            """
            for y in range(values.shape[0]):
                for x in range(values.shape[1]):
                    plt.text(x + 0.5, y + 0.5, '%.4f' % values[y, x],
                            horizontalalignment='center',
                            verticalalignment='center',
                            )
            """
            plt.colorbar(heatmap)
            plt.pause(sleep_time)
            #plt.show()
            

    def MVProp(self, r, p, K=20):
        B = r.shape[0]
        Values = r

        for __ in range(K):
            padded_v = torch.nn.functional.pad(Values, (1, 1, 1, 1, 0, 0), "constant", 0)

            for h_offset, w_offset in [(0, 1), (2, 1), (1, 0), (1, 2)]:
                shifted_v = padded_v[:, h_offset : h_offset + self.maze_h, w_offset : w_offset + self.maze_w]
                nv = r + p * (shifted_v - r)
                Values = Values.maximum(nv)

        return Values

    def get_my_values(self):
        return self.my_stats

    def value_function(self):
        return self.V_
