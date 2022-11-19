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


class DummyAgent(Agent):
    def __init__(self, env, trainer):
        super().__init__(env)
        self.trainer = trainer

    def pi(self, s, k=None):
        return self.trainer.compute_action(s)

class MyClass:
    pass

def my_experiment(a):
    print("Hello world")
    # see https://docs.ray.io/en/latest/rllib/rllib-training.html
    config = A3CConfig().training(lr=0.01/10, grad_clip=30.0).resources(num_gpus=0).rollouts(num_rollout_workers=1)
    config = config.framework('torch')
    config = config.callbacks(MyCallbacks)
    # Set up alternative model (gridworld).

    print(config.to_dict())
    config.model['fcnet_hiddens'] = [24, 24]
    # env = gym.make("MazeDeterministic_empty4-v0")

    trainer = config.build(env="MazeDeterministic_empty4-v0")

    for t in range(30):
        print("Main trainign step", t)
        result = trainer.train()
        rewards = result['hist_stats']['episode_reward']
        print("training epoch", t, len(rewards), max(rewards), result['episode_reward_mean'])



    # return
    # print(pretty_print(result1))
    # import matplotlib.pyplot as plt
    # plt.plot(rewards)
    # plt.show()

    # print( rewards )
    # env = gym.make("CartPole-v1")
    # env.reset()
    env = gym.make("MazeDeterministic_empty4-v0")
    env = VideoMonitor(env)
    train(env, DummyAgent(env, trainer), num_episodes=10)
    a = 234
    #
    # config = A3CConfig()
    # # Print out some default values.
    # print(config.sample_async)
    # # Update the config object.
    # config.training(lr=tune.grid_search([0.001, 0.0001]), use_critic=False)
    # # Set the config object's env.
    # config.environment(env="CartPole-v1")
    # # Use to_dict() to get the old-style python config dict
    # # when running with tune.
    # tune.run(
    #     "A3C",
    #     stop = {"episode_reward_mean": 200},
    #     config = config.to_dict(),
    # )




if __name__ == "__main__":
    #
    # import pickle
    #
    # with open('mydb', 'wb') as f:
    #     pickle.dump({'x': 344}, f)
    #
    # with open('mydb', 'rb') as f:
    #     s = pickle.load(f)
    #     # pickle.dump({'x': 344}, f)
    #
    # sys.exit()
    res = []
    DISABLE = False
    # key = "04ff52c3923a648c9c263246bf44cb955a8bf56d"

    # Optional
    # wandb.watch(model)

    # sys.exit()
    # my_experiment(34)
    # sys.exit()
    #with DTUCluster(job_group="myfarm/job0", nuke_group_folder=True, rq=False, disable=False, dir_map=['../../../mavi'],
    #                nuke_all_remote_folders=True) as cc:
        # my_experiment(2)
    #wfun = cc.wrap(my_experiment) if not DISABLE else my_experiment
    wfun = my_experiment
    for a in [1]:
        res.append(wfun(a))
    print(res)
    # res = cc.wrap(myfun)(args1, args2)
    # val2 = myexperiment(1,2)
    # wait_to_finish()
    print("Job done")
    sys.exit()

