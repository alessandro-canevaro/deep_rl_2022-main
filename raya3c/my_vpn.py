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

#torch.autograd.set_detect_anomaly(True)


#POSTER
#introduction
#theory
#results
#reflection
#a1 format


#6 pages report

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
        self.num_outputs = 16 #int(np.product(self.obs_space.shape))
        print("self.num_outputs", self.num_outputs)
        self._last_batch_size = None
        self.Phi = torch.nn.Linear(48, 48) #Tue set a breakpoint here 3x3x4
        self.Logit = torch.nn.Linear(48, 5)
        self.V_ = torch.zeros((32))
        self.padder = torch.nn.ZeroPad2d(1)

    # Implement your own forward logic, whose output will then be sent
    # through an LSTM.
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        B = obs.shape[0] #batch size
        self.V_ = torch.zeros((B))
        # print("input_dict", input_dict)
        # print("obs", obs.shape)
        # print("obs[0]", obs[0].shape)
        # Store last batch size for value_function output.
        self._last_batch_size = B

        #V = self.VIP(obs)
        V = self.VIP_diff(obs)
        # print("AAAAAAAAAAAAAAaaobs", obs)
        # _, I, J, _ = obs.nonzero()
        # for i, j, obs_k in zip(I, J, obs)
        #agent_locations = []
        logit = torch.zeros((B, 5))
        obs = obs.reshape(B,4,4,3)

        for b_idx in range(B): 

            if not torch.any(obs[b_idx, :, :, 1]):#.sum().item() == 0: if all zeros
                i, j = 1, 1
            else:
                # print("obs[b_idx, :, :, 1]",obs[b_idx, :, :, 1])
                # print("torch.nonzero(obs[b_idx, :, :, 1])", torch.nonzero(obs[b_idx, :, :, 1]))
                i, j = torch.nonzero(obs[b_idx, :, :, 1])[0] #the [0] is because nonzero is returning a tuple of indexes
            #agent_locations.append((i[0], j[0]))
            # print("V", V)

            # return V.reshape((1, 16)), []
            # print("i j bidx", type(i),type(j), type(b_idx))
            print("SHAPE", self.V_.shape, b_idx)
            self.V_[b_idx] = V[b_idx][i, j]

            sub_state = obs[b_idx, i-1:i+2, j-1:j+2, :]
            sub_v = V[b_idx][i-1:i+2, j-1:j+2]
            # print("sub_state and sub_v", sub_state.shape, sub_v.shape)
            # print("torch.concatenate((sub_state.flatten(), sub_v.flatten())) shape", torch.concatenate((sub_state.flatten(), sub_v.flatten())).shape)
            # logit_input = torch.concatenate((sub_state.flatten(), sub_v.flatten()))
            # logit[b_idx] = self.Logit(logit_input.reshape((1, 36)).squeeze())
            
            logit[b_idx] = self.Logit(obs[b_idx].flatten())

        #print("logit.shape", logit.shape)
        return logit, []

    def VIP_diff(self, obs, K=20): #parallel differentiable version of VIP function
        final_v = []
        # rewards in , out and probabilities
        for obs_idx in range(obs.shape[0]):
            output = self.Phi(obs[obs_idx].flatten())
            (rin, rout, p) = output[:16].reshape(4, 4), output[16:32].reshape(4, 4), output[32:].reshape(4, 4)
            #print("(rin, rout, p)", (rin.shape, rout.shape, p.shape))
            
            rin_padded = self.padder(rin)
            h, w = 4, 4#obs.shape[0], obs.shape[1]
            v = torch.zeros((h, w), dtype=torch.float32)#self.value_function()
            for k in range(K):
                v_padded = self.padder(v)
                for w_offset, h_offset in [(0, 1), (2, 1), (1, 0), (1, 2)]:
                    nv = p * v_padded[w_offset:w_offset+w, h_offset:h_offset+h] + rin_padded[w_offset:w_offset+w, h_offset:h_offset+h] - rout
                    v.maximum(nv)

            final_v.append(v)

        return final_v

    def VIP(self, obs, K=20):
        final_v = []
        # rewards in , out and probabilities
        for obs_idx in range(obs.shape[0]):
            output = self.Phi(obs[obs_idx].flatten())
            (rin, rout, p) = output[:16].reshape(4, 4), output[16:32].reshape(4, 4), output[32:].reshape(4, 4)
            #print("(rin, rout, p)", (rin.shape, rout.shape, p.shape))
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
        return self.V_#torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))

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
        print("RESULT", result)
        rewards = result['hist_stats']['episode_reward']
        print("training epoch", t, len(rewards), rewards, result['episode_reward_mean'])
        # print("training epoch", t, len(rewards), max(rewards), result['episode_reward_mean'])

    
    

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
    

