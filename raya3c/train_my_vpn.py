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

#from my_vpn import VPNNetwork
from my_vpn_v2 import VPNNetwork
from datetime import datetime

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



vin_label = "vin_network_model"
ModelCatalog.register_custom_model(vin_label, VPNNetwork)

def my_experiment():
    print("Hello world")
    # see https://docs.ray.io/en/latest/rllib/rllib-training.html
    mconf = dict(custom_model=vin_label, use_lstm=False)
    config = A3CConfig().training(lr=0.01/10, grad_clip=30.0, model=mconf).resources(num_gpus=0).rollouts(num_rollout_workers=1)
    config = config.framework('torch')
    # config.
    config = config.callbacks(MyCallbacks)
    # Set up alternative model (gridworld).
    # config = config.model(custom_model="my_torch_model", use_lstm=False)

    # config.model['fcnet_hiddens'] = [24, 24]
    #env = gym.make("MazeDeterministic_empty4-v0")

    trainer = config.build(env="MazeDeterministic_empty4-train-v0")
    EPOCHS = 150

    print("training started")
    for t in range(EPOCHS):
        result = trainer.train()
        #print("RESULT", result)
        rewards = result['hist_stats']['episode_reward']
        print(f"training epoch: {t}, rewards: {len(rewards)}, max: {max(rewards)}, avg: {result['episode_reward_mean']}")

    date = str(datetime.now()).split(".")[0].replace("-", "_").replace(":", "_").replace(" ", "_")
    checkpoint_dir = trainer.save(f"./saved_models/check_{date}")
    print(f"Saved model in {checkpoint_dir}")
        

if __name__ == "__main__":
    my_experiment()
    print("Job done")
    sys.exit()
    

