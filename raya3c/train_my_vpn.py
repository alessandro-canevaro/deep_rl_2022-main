import sys, os

sys.path.append(os.path.normpath(os.path.dirname(__file__) + "/../"))
import gym
from mazeenv import maze_register
from a3c import A3CConfig

# import farmer
# from dtufarm import DTUCluster
from irlc import Agent, train, VideoMonitor

import numpy as np
import ray
from ray import tune
from ray.tune.logger import pretty_print
from raya3c.my_callback import MyCallbacks

# The custom model that will be wrapped by an LSTM.
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch

from my_vpn import VPNNetwork
from datetime import datetime
from tqdm import tqdm
from functools import partial

from raya3c.experiments_config import config as my_cfg

vin_label = "vin_network_model"
ModelCatalog.register_custom_model(vin_label, VPNNetwork)

def my_experiment():
    print("Setting up training")

    net_config={"is_train": True,
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
    config = config.callbacks(partial(MyCallbacks, my_cfg["run_name"]))

    trainer = config.build(env="MazeDeterministic_empty4-train-v0")

    print("training started")
    with tqdm(total=my_cfg["epochs"], desc="Epochs", bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}") as bar:
        for t in range(my_cfg["epochs"]):
            result = trainer.train()
            #print("RESULT", result)
            rewards = result["hist_stats"]["episode_reward"]
            #print(f"training epoch: {t}, rewards: {len(rewards)}, max: {max(rewards)}, avg: {result['episode_reward_mean']}")
            bar.update()

    checkpoint_dir = trainer.save(f"./saved_models/"+my_cfg["run_name"])
    print(f"Saved model in {checkpoint_dir}")


if __name__ == "__main__":
    my_experiment()
    print("Job done")
    sys.exit()
