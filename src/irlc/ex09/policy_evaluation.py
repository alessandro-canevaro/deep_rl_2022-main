# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf).
"""
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from irlc.ex09.small_gridworld import SmallGridworldMDP,plot_value_function
from irlc import savepdf

def qs_(mdp, s, gamma, v): 
    """
    Something you need to do very often is to compute the Q-values in a given state. I.e., given you are in state s,
    compute the Q-values:

    > Q(s,a) = E[r + gamma * v[s'] | s,a]

    for all available actions a. This helper function does exactly that, by returning the Q-values as a dictionary:
    > {a_1: q_1, a_2: q_2, ..., a_n: q_n}
    so that Q(s,a_k) = q_k

    In other words, the following should work
    > Qs = qs_(mdp, s, gamma, v)
    > Qs[a] # This is the Q-value Q(s,a)
    Don't be afraid to use more than one line in your solution!

    Hints:
        * The terminal states represents a special case. Remember that v[s] = 0 if s is a terminal state (this is explained in (SB18)).
    """
    # TODO: 1 lines missing.
    raise NotImplementedError("Implement function body")

def policy_evaluation(pi, mdp, gamma=.99, theta=0.00001):
    """ Implements the iterative policy-evaluation algorithm ((SB18, Section 4.1)).
    The algorithm is given a policy pi which is represented as a dictionary so that

    > pi[s][a] = p

    is the probability p of taking action a in state s. The 'mdp' is a MDP-instance and the other terms have the same meaning as in the algorithm.
    It should return a dictionary v so that
    > v[s]
    is the value-function evaluated in state s. I recommend using the qs_-function defined above.
    """
    v = defaultdict(float)
    Delta = theta #Initialize the 'Delta'-variable to a large value to make sure the first iteration of the method runs.
    while Delta >= theta: # Outer loop in (SB18)
        Delta = 0 # Remember to update Delta (same meaning as in (SB18))
        # Remember that 'S' in (SB18) is actually just the set of non-terminal states (NOT including terminal states!)
        for s in mdp.nonterminal_states: # See the MDP class if you are curious about how this variable is defined.
            """ Implement the main body of the policy evaluation algorithm here. You can do this directly, 
            or implement (and use) the qs_-function (consider what it does and compare to the algorithm).
            If you do so, note that qs_(mdp, s, gamma, v) computes the equivalent of Q(s,a) (as a dictionary), 
            and in the algorithm, you then need to compute the expectation over pi:
            > sum_a pi(a|s) Q(s,a) 
            In code it would be more akin to 
            q = qs_(...)
            sum_a pi[s][a] * q[a]
            
            Don't be afraid to use a few more lines than I do.             
            """
            # TODO: 2 lines missing.
            raise NotImplementedError("")
            """ stop condition. v_ is the old value of the value function (see algorithm listing in (SB18)) which you need to update. """
            Delta = max(Delta, np.abs(v_ - v[s]))
    return v


if __name__ == "__main__":
    mdp = SmallGridworldMDP()
    """
    Create the random policy pi0 below. The policy is defined as a nested dict, i.e. 
    
    > pi0[s][a] = (probability to take action a in state s)
     
    """
    pi0 = {s: {a: 1/len(mdp.A(s)) for a in mdp.A(s) } for s in mdp.nonterminal_states }
    V = policy_evaluation(pi0, mdp, gamma=1)
    plot_value_function(mdp, V)
    plt.title("Value function using random policy")
    savepdf("policy_eval")
    plt.show()

    expected_v = np.array([0, -14, -20, -22,
                           -14, -18, -20, -20,
                           -20, -20, -18, -14,
                           -22, -20, -14, 0])
