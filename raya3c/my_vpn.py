import sys, os
sys.path.append(os.path.normpath( os.path.dirname(__file__) +"/../" ))
import gym
from mazeenv import maze_register
from a3c import A3CConfig
# import farmer
#from dtufarm import DTUCluster
from irlc import Agent, train, VideoMonitor
import numpy as np
from ray import tune
from ray.tune.logger import pretty_print
from raya3c.my_callback import MyCallbacks

# The custom model that will be wrapped by an LSTM.
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch

"""
import ray
import ray.rllib.agents.ppo as ppo

ray.shutdown()
ray.init(ignore_reinit_error=True)
"""
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


# def pi(self, s, k=None):
#     # take random actions and randomize the value function V for visualization.
#     return self.env.action_space.sample()

# def Phi(self, s):
#     rout = s[:, :, 2]
#     rin = s[:, :, 2] - 0.05  # Simulate a small transition cost.
#     p = 1 - s[:, :, 0]
#     return (rin, rout, p)

# def train(self, s, a, r, sp, done=False):
#     # do training stuff here (save to buffer, call torch, whatever)
#     w = VIP(s, self.Phi(s))
#     vv = w[:,:,min(self.k, w.shape[2]-1)]
#     for i,j in self.v:
#         self.v[i,j] = vv[j, i] # annoying transpose
#     self.k += 1
#     if done:
#         self.k = 0

# def VIP(s, Phi, K=20):
#     # rewards in , out and probabilities
#     (rin, rout, p) = Phi
#     h, w = s.shape[0], s.shape[1]
#     v = np.zeros((h,w, K+1))
#     for k in range(K):
#         for i in range(h):
#             for j in range(w):
#                 v[i, j, k+1] = v[i,j,k]
#                 for di, dj in [ (-1, 0), (1, 0), (0, -1), (0, 1)]:
#                     if di + i < 0 or dj + j < 0:
#                         continue
#                     if di + i >= h or dj + j >= w:
#                         continue
#                     ip = i + di
#                     jp = j + dj
#                     nv = p[i,j] * v[ip, jp,k] + rin[ip, jp] - rout[i,j]
#                     v[i,j,k+1] = max( v[i,j,k+1], nv)
#     return v


vin_label = "vin_network_model"
# Kig paa: FullyConnectedNetwork som er den Model-klassen bruger per default.
# alt. copy-paste FullyConnectedNetwork-koden ind i denne klasse og modififer gradvist (check at den virker paa simple gridworld)
class VINNetwork(TorchModelV2, torch.nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)
        self.num_outputs = 16#int(np.product(self.obs_space.shape))
        print("self.num_outputs", self.num_outputs)
        self._last_batch_size = None
        self.Phi = torch.nn.Linear(48, 48) #Tue set a breakpoint here
        self.Logit = torch.nn.Linear(48, 3)

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict, state, seq_lens):
        
        obs = input_dict["obs"]
        print("input_dict", input_dict)
        print("obs", obs.shape)
        print("obs[0]", obs[0].shape)
        # Store last batch size for value_function output.
        self._last_batch_size = obs.shape[0]
        """
        V = self.VIP(input_dict)
        print("V", V)
        V = V.flatten()
        print("V flatten", V)

        # i, j = obs[0][:, :, 1]
        # self.V_ = V[i[0], j[0]]

        # sub_state = obs[i[0]-1:i[0]+1, j[0]-1:j[0]+1, :]
        # sub_v = V[i[0]-1:i[0]+1, j[0]-1:j[0]+1]
        # logit = self.Logit(torch.concatenate(sub_state.flatten(), sub_v.flatten()))
        """
        return torch.rand([32, 16]), []


    def VIP(self, input_dict, K=20):
        obs = input_dict["obs"]

        # rewards in , out and probabilities
        output = self.Phi(obs[0])
        (rin, rout, p) = output[:16].reshape(4, 4), output[16:32].reshape(4, 4), output[32:].reshape(4, 4)
        print("(rin, rout, p)", (rin.shape, rout.shape, p.shape))
        h, w = 4, 4#obs.shape[0], obs.shape[1]
        v = np.zeros((h, w, K+1))#self.value_function()
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
                        v[i,j,k+1] = max( v[i,j,k+1], nv)
        print("v.shape",v.shape)
        vv = v[:,:,min(K, v.shape[2]-1)]
        #v_out = vv.copy()
        #for i,j in vv:
        #    v_out[i,j] = vv[j, i] # annoying transpose
        #print("v_out.shape",v_out.shape)
        return vv


    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))

class TestVINNetwork(torch.nn.Module):
    def __init__(self):
        # super().__init__(obs_space, action_space, num_outputs, model_config, name)
        torch.nn.Module.__init__(self)
        self.Phi = torch.nn.Linear(48, 48) #Tue set a breakpoint here
        self.Logit = torch.nn.Linear(48, 3)

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict):
        obs = input_dict["obs"]
        print("input_dict", input_dict)
        print("obs", obs.shape)
        print("obs[0]", obs[0].shape)
        # Store last batch size for value_function output.
        self._last_batch_size = obs.shape[0]
        
        V = self.VIP(input_dict)

        # i, j = obs[0][:, :, 1]
        # self.V_ = V[i[0], j[0]]

        # sub_state = obs[i[0]-1:i[0]+1, j[0]-1:j[0]+1, :]
        # sub_v = V[i[0]-1:i[0]+1, j[0]-1:j[0]+1]
        # logit = self.Logit(torch.concatenate(sub_state.flatten(), sub_v.flatten()))

        return V, []


    def VIP(self, input_dict, K=20):
        obs = input_dict["obs"]

        # rewards in , out and probabilities
        output = self.Phi(obs[0])
        (rin, rout, p) = output[:16].reshape(4, 4), output[16:32].reshape(4, 4), output[32:].reshape(4, 4)

        h, w = 4, 4#obs.shape[0], obs.shape[1]
        v = np.zeros((h, w, K+1))#self.value_function()
        
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
                        v[i,j,k+1] = max( v[i,j,k+1], nv)

        return v


    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))


ModelCatalog.register_custom_model(vin_label, VINNetwork)


def my_experiment():
    print("Hello world")
    # see https://docs.ray.io/en/latest/rllib/rllib-training.html
    mconf = dict(custom_model=vin_label, use_lstm=False)
    config = A3CConfig().training(lr=0.01/10, grad_clip=30.0, model=mconf).resources(num_gpus=0).rollouts(num_rollout_workers=1)
    config = config.framework('torch')
    # config.
    #config = config.callbacks(MyCallbacks)
    # Set up alternative model (gridworld).
    # config = config.model(custom_model="my_torch_model", use_lstm=False)

    config.model['fcnet_hiddens'] = [24, 24]
    #env = gym.make("MazeDeterministic_empty4-v0")

    trainer = config.build(env="MazeDeterministic_empty4-v0")
    EPOCHS = 1

    #https://discuss.ray.io/t/error-when-setting-done-true-eval-data-i-env-id-yields-indexerror-list-index-out-of-range/867

    for t in range(EPOCHS):
        print("Main training step", t)
        result = trainer.train()
        rewards = result['hist_stats']['episode_reward']
        print("training epoch", t, len(rewards), max(rewards), result['episode_reward_mean'])

    config.save
    

if __name__ == "__main__":
    """
    network = TestVINNetwork()
    network.forward({'obs': torch.rand(32, 48)})
    """
    res = []
    DISABLE = True
    my_experiment()
    print("Job done")
    sys.exit()
    

