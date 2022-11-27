import sys, os
sys.path.append(os.path.normpath( os.path.dirname(__file__) +"/../" ))
import gym
from mazeenv.maze_environment import MazeEnvironment
gym.envs.register(
     id="MazeDeterministic_empty4-train-v0",
     entry_point='mazeenv.maze_environment:MazeEnvironment',
     max_episode_steps=200,
     kwargs=dict(size=4, blockpct=0, seed=0, render_mode='native'),
)
gym.envs.register(
     id="MazeDeterministic_empty4-test-v0",
     entry_point='mazeenv.maze_environment:MazeEnvironment',
     max_episode_steps=200,
     kwargs=dict(size=4, blockpct=0, seed=0, render_mode='human'),
)
from ray.tune.registry import register_env

def env_creator(env_config):
     env = gym.make("MazeDeterministic_empty4-train-v0")
     from gym.wrappers.flatten_observation import FlattenObservation
     env = FlattenObservation(env) # flat as fuck to avoid rllib interpreting it as an image.
     return env

register_env('MazeDeterministic_empty4-train-v0', env_creator)

# Apparently needed because #reasons.
# from ray import tune
# tune.register_env('MazeDeterministic_empty4-v0', lambda cfg: gym.make("MazeDeterministic_empty4-v0"))
# gym.register("MazeDeterministic_simple3-v0", MazeEnvironment)