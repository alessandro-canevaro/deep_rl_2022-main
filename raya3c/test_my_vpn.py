import sys, os

sys.path.append(os.path.normpath(os.path.dirname(__file__) + "/../"))
import gym
from mazeenv import maze_register
from a3c import A3CConfig

# import farmer
# from dtufarm import DTUCluster
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

from my_vpn import VPNNetwork


class MyAgent(Agent):
    def __init__(self, env, trainer):
        super().__init__(env)
        self.trainer = trainer

    def pi(self, s, k=None):
        # print("state in agent", s)
        # Wreturn self.env.action_space.sample()
        a = self.trainer.compute_single_action(s)
        # print("ACTION", a)
        return a


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
    env = VideoMonitor(env)
    # added sleep in agent.py line 199
    # removed render_as_text in maze_environment.py line 101, 176
    train(env, MyAgent(env, trainer), num_episodes=5, sleep_time=0)

    env.close()


if __name__ == "__main__":
    my_experiment()
    print("Job done")
    sys.exit()
