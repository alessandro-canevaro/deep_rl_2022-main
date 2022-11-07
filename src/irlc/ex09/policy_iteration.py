# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf).
"""
import numpy as np
from irlc.ex09.small_gridworld import SmallGridworldMDP
import matplotlib.pyplot as plt
from irlc.ex09.policy_evaluation import policy_evaluation
from irlc.ex01.agent import Agent
from irlc.ex09.policy_evaluation import qs_

class PolicyIterationAgent(Agent):
    """ this is an old of how we can combine policy iteration into the Agent interface which we will
    use in the subsequent weeks. """
    def __init__(self, mdp, gamma):
        self.policy, self.v = policy_iteration(mdp, gamma)
        super().__init__(self)

    def pi(self, s, k=None): 
        """ Return the action to take in state s according to policy-iteration. Look at the __init__-function!
        If in doubt, insert a breakpoint and take a look at the self.policy-variable.
        """
        # TODO: 1 lines missing.
        raise NotImplementedError("Implement function body")
        return action

    def train(self, s, a, r, sp, done=False):
        pass

    def __str__(self):
        return 'PolicyIterationAgent'

def policy_iteration(mdp, gamma=1.0):
    """
    Implement policy iteration (see (SB18, Section 4.3)).

    Note that policy iteration only considers deterministic policies. we will therefore use the shortcut by representing the policy pi
    as a dictionary (similar to the DP-problem in week 2!) so that
    > a = pi[s]
    is the action in state s.

    """
    pi = {s: np.random.choice(mdp.A(s)) for s in mdp.nonterminal_states}
    policy_stable = False
    V = None # Sutton has an initialization-step, but it can actually be skipped if we intialize the policy randomly.
    while not policy_stable:
        # Evaluate the current policy using your code from the previous exercise.
        # The main complication is that we need to transform our deterministic policy, pi[s], into a stochastic one pi[s][a].
        # but this can be done in a single line using dictionary comprehension.
        V = policy_evaluation( {s: {pi[s]: 1} for s in mdp.nonterminal_states}, mdp, gamma)
        """ Implement the method. This is step (3) in (SB18). """
        policy_stable = True   # Will be set to False if the policy pi changes
        """ Implement the steps for policy improvement here. Start by writing a for-loop over all non-terminal states
        you can see the policy_evaluation function for how to do this, but 
        I recommend looking at the property mdp.nonterminal_states (see MDP class for more information). 
        Hints:
            * In the algorithm in (SB18), you need to perform an argmax_a over what is actually Q-values. The function
            qs_(mdp, s, gamma, V) can compute these. 
            * The argmax itself, assuming you follow the above procedure, involves a dictionary. It can be computed 
            using methods similar to those we saw in week2 of the DP problem. See the ex2/dp.py file. 
            It is not a coincidence these algorithms are very similar -- if you think about it, the maximization step closely resembles the DP algorithm!
        """
        # TODO: 6 lines missing.
        raise NotImplementedError("")
    return pi, V

if __name__ == "__main__":
    mdp = SmallGridworldMDP()
    pi, v = policy_iteration(mdp, gamma=0.99)
    expected_v = np.array([ 0, -1, -2, -3,
                           -1, -2, -3, -2,
                           -2, -3, -2, -1,
                           -3, -2, -1,  0])

    from irlc.ex09.small_gridworld import plot_value_function
    plot_value_function(mdp, v)
    plt.title("Value function using policy iteration to find optimal policy")
    from irlc import savepdf
    savepdf("policy_iteration")
    plt.show()
