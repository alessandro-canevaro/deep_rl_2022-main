import sys, os

sys.path.append(os.path.normpath(os.path.dirname(__file__) + "/../"))
import gym
from mazeenv import maze_register
from a3c import A3CConfig

from irlc import Agent, train, VideoMonitor

import numpy as np
from ray import tune
from ray.tune.logger import pretty_print
from raya3c.my_callback import MyCallbacks

# The custom model that will be wrapped by an LSTM.
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from raya3c.experiments_config import config as my_cfg
from collections import defaultdict
from my_vpn import VPNNetwork
# Uncomment the following line if you want to use the MVProp VPN
#from my_vpn_mvprop import VPNNetwork


class MyAgent(Agent):
    def __init__(self, env, trainer):
        #self.v = defaultdict(lambda: 0)
        self.k = 0
        super().__init__(env)
        self.trainer = trainer

    def pi(self, s, k=None):
        a = self.trainer.compute_single_action(s)
        return a
    
    def plotvalues(self):
        policy = self.trainer.get_policy()
        agent_val = policy.model.value_function()
        values, a_i, a_j, p = policy.model.get_my_values()
        assert torch.allclose(agent_val, values[0, a_i, a_j]), f"agent val {agent_val}, agent pos ({a_i}, {a_j}), values {values}"

        print(f"agent position: i={a_i} j={a_j}")

        p = p[0, :, :].detach().numpy()

        vv = values[0, :, :].detach().numpy()
        print(vv)
        for i,j in self.v:
            self.v[i,j] = vv[my_cfg["maze_size"] -1 - j, i]

    def train(self, s, a, r, sp, done=False):
        #self.plotvalues()
        self.k += 1
        if done:
            self.k = 0


vin_label = "vin_network_model"
ModelCatalog.register_custom_model(vin_label, VPNNetwork)

checkpoint_dir = my_cfg["models_dir"]+my_cfg["run_name"]+my_cfg["chk_point"] 
TRAIN_ENV = "MazeDeterministic_empty4-train-v0"
TEST_ENV = "MazeDeterministic_empty4-test-v0"


def my_experiment():
    print("Setting up testing")

    net_config={"is_train": False,
                "maze_w": my_cfg["maze_size"],
                "maze_h": my_cfg["maze_size"],
                "vin_k": my_cfg["vin_k"]}
    mconf = dict(custom_model=vin_label, use_lstm=False, custom_model_config=net_config)
    config = (
        A3CConfig()
        .training(lr=my_cfg["lr"], train_batch_size=my_cfg["batch_size"], grad_clip=30.0, model=mconf)
        .resources(num_gpus=0)
        .rollouts(num_rollout_workers=1)
    )
    config = config.framework("torch")

    trainer = config.build(env=TRAIN_ENV)

    trainer.restore(checkpoint_dir)

    env = gym.make(TEST_ENV)

    agent = MyAgent(env, trainer)
    env = VideoMonitor(env, agent=agent)#, agent_monitor_keys=('v', ), render_kwargs={'method_label': 'VI-K'})

    train(env, agent, num_episodes=10, sleep_time=0.1)

    env.close()


if __name__ == "__main__":
    my_experiment()
    print("Job done")
    sys.exit()
