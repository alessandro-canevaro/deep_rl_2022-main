# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from irlc.ex09.value_iteration import value_iteration
from irlc.ex09.mdp import GymEnv2MDP
from irlc import TabularAgent
from irlc import Agent
import numpy as np

class ValueIterationAgent(TabularAgent):
    def __init__(self, env, mdp=None, gamma=1, epsilon=0, **kwargs):
        super().__init__(env)
        # if mdp is None: # Try to see if MDP can easily be found from environment (if not specified):
        #     if hasattr(env, 'mdp'):
        #         mdp = env.mdp
        #     elif hasattr(env, 'P'):
        #         mdp = GymEnv2MDP(env)
        #     else:
        #         raise Exception("Must supply a MDP so I can plan!")
        self.epsilon = epsilon
        # TODO: 1 lines missing.
        raise NotImplementedError("Call the value_iteration function and store the policy for later.")
        self.Q = None  # This is slightly hacky; pay no attention to it. It is for visualization-purposes.

    def pi(self, s, k=0):
        """ With probability (1-epsilon), the take optimal action as computed using value iteration
         With probability epsilon, take a random action. You can do this using return self.random_pi(s)
        """
        if np.random.rand() < self.epsilon:
            return self.random_pi(s)
        else:
            """ Return the optimal action here. This should be computed using value-iteration. 
             To speed things up, I recommend calling value-iteration from the __init__-method and store the policy. """
            # TODO: 1 lines missing.
            raise NotImplementedError("Compute and return optimal action according to value-iteration.")
            return action


if __name__ == "__main__":
    from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
    env = SuttonCornerGridEnvironment(living_reward=-1)
    from irlc import VideoMonitor, train
    # Note you can access the MDP for a gridworld using env.mdp. The mdp will be an instance of the MDP class we have used for planning so far.
    agent = ValueIterationAgent(env, mdp=env.mdp) # Make a ValueIteartion-based agent
    # Let's have a cute little animation. Try to leave out the agent_monitor_keys line to see what happens.
    env = VideoMonitor(env, agent=agent)
    train(env, agent, num_episodes=20)                             # Train for 100 episodes
    env.savepdf("smallgrid.pdf") # Take a snapshot of the final configuration
    env.close() # Whenever you use a VideoMonitor, call this to avoid a dumb openglwhatever error message on exit
