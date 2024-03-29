import sys, os
sys.path.append(os.path.normpath( os.path.dirname(__file__) +"/../" ))
import gym
from mazeenv.maze_environment import MazeEnvironment
from raya3c.experiments_config import config

gym.envs.register(
     id="MazeDeterministic_empty4-train-v0",
     entry_point='mazeenv.maze_environment:MazeEnvironment',
     max_episode_steps=200,
     kwargs=dict(size=config["maze_size"], blockpct=config["wall_prob"], render_mode='native'),
)
gym.envs.register(
     id="MazeDeterministic_empty4-test-v0",
     entry_point='mazeenv.maze_environment:MazeEnvironment',
     max_episode_steps=25,
     kwargs=dict(size=config["maze_size"], blockpct=config["wall_prob"], render_mode='human'),
)
from ray.tune.registry import register_env

def env_creator(env_config):
     env = gym.make("MazeDeterministic_empty4-train-v0")
     from gym.wrappers.flatten_observation import FlattenObservation
     env = FlattenObservation(env)
     return env

register_env('MazeDeterministic_empty4-train-v0', env_creator)
