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

from my_vpn import VPNNetwork

class MyAgent(Agent):
    def __init__(self, env, trainer):
        super().__init__(env)
        self.trainer = trainer

    def pi(self, s, k=None):
        #print("state in agent", s)
        #Wreturn self.env.action_space.sample()
        a = self.trainer.compute_action(s)
        #print("ACTION", a)
        return a
    
    def train(self, s, a, r, sp, done=False):
        pass
        #print(f"TRAIN s, a, r, sp, done, {s, a, r, sp, done}")

vin_label = "vin_network_model"
ModelCatalog.register_custom_model(vin_label, VPNNetwork)


def my_experiment():
    print("Hello world testing")
    # see https://docs.ray.io/en/latest/rllib/rllib-training.html
    mconf = dict(custom_model=vin_label, use_lstm=False)
    config = A3CConfig().training(lr=0.01/10, grad_clip=30.0, model=mconf).resources(num_gpus=0).rollouts(num_rollout_workers=1)
    config = config.framework('torch')
    # config.
    #config = config.callbacks(MyCallbacks)
    # Set up alternative model (gridworld).
    # config = config.model(custom_model="my_torch_model", use_lstm=False)

    # config.model['fcnet_hiddens'] = [24, 24]
    #env = gym.make("MazeDeterministic_empty4-v0")

    trainer = config.build(env="MazeDeterministic_empty4-train-v0")


    checkpoint_dir = "./saved_models/100epochs_working"#"./saved_models/checkpoint_000001"
    trainer.restore(checkpoint_dir)

    env = gym.make("MazeDeterministic_empty4-test-v0")
    env = VideoMonitor(env)
    #added sleep in agent.py line 199
    #removed render_as_text in maze_environment.py line 101, 176
    train(env, MyAgent(env, trainer), num_episodes=5)
    
    env.close()
        

if __name__ == "__main__":
    my_experiment()
    print("Job done")
    sys.exit()
    
