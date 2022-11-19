from collections import defaultdict
import numpy as np
from irlc import Agent
from mazeenv.maze_environment import MazeEnvironment
from irlc.gridworld.gridworld_environments import CliffGridEnvironment, BookGridEnvironment
from irlc import PlayWrapper, VideoMonitor
from irlc import Agent, train


# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.nn.parameter import Parameter


"""
steps:
- go to example_vin.py
- change model,, inspect obs dict
- implement forward:
    implement value function->first use his VPN (=VIP function)
    try with convolutions instead of 3 for loops
    final linear layer
- use a3c
- plot r1,r2,p check if sensible results
- inspect how to create custom maze check maze_register.py
- once trained apply action
"""


class VIAgent(Agent):
    def __init__(self, env):
        self.v = defaultdict(lambda: 0)
        # self.k = 0
        super().__init__(env)

    def pi(self, s, k=None):
        # take random actions and randomize the value function V for visualization.
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if s[i,j,0] == 0:
                    self.v[(i,j)] = np.random.rand()-0.5

        return self.env.action_space.sample()
    
    # def Phi(self, s):
    #     rout = s[:, :, 2]
    #     rin = s[:, :, 2] - 0.05  # Simulate a small transition cost.
    #     p = 1 - s[:, :, 0]
    #     return (rin, rout, p)
    
    def train(self, s, a, r, sp, done):
        # """
        # Called at each step of the simulation after a = pi(s,k) and environment transition to sp. 
        
        # Allows the agent to learn from experience  #!s

        # :param s: Current state x_k
        # :param a: Action taken
        # :param r: Reward obtained by taking action a_k in x_k
        # :param sp: The state that the environment transitioned to :math:`{\\bf x}_{k+1}`
        # :param done: Whether environment terminated when transitioning to sp
        # :return: None
        # """
        # do training stuff here (save to buffer, call torch, etc.)

        # w = VIP(s, self.Phi(s))
        # vv = w[:,:,min(self.k, w.shape[2]-1)]
        # for i,j in self.v:
        #     self.v[i,j] = vv[j, i] # annoying transpose
        # self.k += 1
        # if done:
            # self.k = 0
        pass

# class Model(nn.Module):
#     def __init__(self, config):
#         super(VIN, self).__init__()
#         self.config = config
#         self.h = nn.Conv2d(
#             in_channels=config.l_i,
#             out_channels=config.l_h,
#             kernel_size=(3, 3),
#             stride=1,
#             padding=1,
#             bias=True)
#         self.r = nn.Conv2d(
#             in_channels=config.l_h,
#             out_channels=1,
#             kernel_size=(1, 1),
#             stride=1,
#             padding=0,
#             bias=False)
#         self.q = nn.Conv2d(
#             in_channels=1,
#             out_channels=config.l_q,
#             kernel_size=(3, 3),
#             stride=1,
#             padding=1,
#             bias=False)
#         self.fc = nn.Linear(in_features=config.l_q, out_features=8, bias=False)
#         self.w = Parameter(
#             torch.zeros(config.l_q, 1, 3, 3), requires_grad=True)
#         self.sm = nn.Softmax(dim=1)

#     def forward(self, input_view, state_x, state_y, k):
#         """
#         :param input_view: (batch_sz, imsize, imsize)
#         :param state_x: (batch_sz,), 0 <= state_x < imsize
#         :param state_y: (batch_sz,), 0 <= state_y < imsize
#         :param k: number of iterations
#         :return: logits and softmaxed logits
#         """
#         h = self.h(input_view)  # Intermediate output
#         r = self.r(h)           # Reward
#         q = self.q(r)           # Initial Q value from reward
#         v, _ = torch.max(q, dim=1, keepdim=True)

#         def eval_q(r, v):
#             return F.conv2d(
#                 # Stack reward with most recent value
#                 torch.cat([r, v], 1),
#                 # Convolve r->q weights to r, and v->q weights for v. These represent transition probabilities
#                 torch.cat([self.q.weight, self.w], 1),
#                 stride=1,
#                 padding=1)

#         # Update q and v values
#         for i in range(k - 1):
#             q = eval_q(r, v)
#             v, _ = torch.max(q, dim=1, keepdim=True)

#         q = eval_q(r, v)
#         # q: (batch_sz, l_q, map_size, map_size)
#         batch_sz, l_q, _, _ = q.size()
#         q_out = q[torch.arange(batch_sz), :, state_x.long(), state_y.long()].view(batch_sz, l_q)

#         logits = self.fc(q_out)  # q_out to actions

#         return logits, self.sm(logits)

class ShittyVIAgent(Agent):
    def __init__(self, env):
        self.v = defaultdict(lambda: 0)
        self.k = 0
        super().__init__(env)

    def pi(self, s, k=None):
        # take random actions and randomize the value function V for visualization.
        return self.env.action_space.sample()

    def Phi(self, s):
        rout = s[:, :, 2]
        rin = s[:, :, 2] - 0.05  # Simulate a small transition cost.
        p = 1 - s[:, :, 0]
        return (rin, rout, p)

    def train(self, s, a, r, sp, done=False):
        # do training stuff here (save to buffer, call torch, whatever)
        w = VIP(s, self.Phi(s))
        vv = w[:,:,min(self.k, w.shape[2]-1)]
        for i,j in self.v:
            self.v[i,j] = vv[j, i] # annoying transpose
        self.k += 1
        if done:
            self.k = 0

def VIP(s, Phi, K=20):
    (rin, rout, p) = Phi
    h, w = s.shape[0], s.shape[1]
    v = np.zeros((h,w, K+1))
    for k in range(K):
        for i in range(h):
            for j in range(w):
                v[i,j, k+1] = v[i,j,k]
                for di, dj in [ (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if di + i < 0 or dj + j < 0:
                        continue
                    if di + i >= h or dj + j >= w:
                        continue
                    ip = i + di
                    jp = j + dj
                    nv = p[i,j] * v[ip, jp,k] + rin[ip, jp] - rout[i,j]
                    v[i,j,k+1] = max( v[i,j,k+1], nv)
    return v


if __name__ == "__main__":
    env = MazeEnvironment(size=10, render_mode='human')
    agent = ShittyVIAgent(env)
    agent = PlayWrapper(agent, env, autoplay=False)
    experiment = "experiments/q1_value_iteration"
    env = VideoMonitor(env, agent=agent, fps=100, continious_recording=True, agent_monitor_keys=('v', ), render_kwargs={'method_label': 'VI-K'})
    train(env, agent, experiment_name=experiment, num_episodes=10, max_steps=100)
    
    
    
    env.close()
