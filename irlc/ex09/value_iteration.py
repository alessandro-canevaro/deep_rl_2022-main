# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
"""
References:
  [SB18] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. The MIT Press, second edition, 2018. (See sutton2018.pdf).
"""
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from irlc.ex09.policy_evaluation import qs_
from irlc import savepdf

def value_iteration(mdp, gamma=.99, theta=0.0001, max_iters=10 ** 6, verbose=False):
    """ Implement the value-iteration algorithm defined in (SB18, Section 4.4).
    The inputs should be self-explanatory given the pseudo-code.

    I have also included a max_iters variable which represents an upper bound on the total number of iterations. This is useful
    if you want to check what the algorithm does after a certain (e.g. 1 or 2) steps.

    The verbose-variable makes the algorithm print out the biggest change in the value-function in a single step.
    This is useful if you run it on a large problem and want to know how much time remains, or simply get an idea of
    how quickly it converges.
    """
    V = defaultdict(lambda: 0)  # value function
    for i in range(max_iters):
        Delta = 0
        for s in mdp.nonterminal_states:
            """ Perform the update the value-function V[s] here for the given state. 
            Note that this has a lot of similarity to the policy-evaluation algorithm, and you can re-use 
            a lot of that solution, including qs_(...) (assuming you used that function). """
            # TODO: 2 lines missing.
            v, V[s] = V[s], max(qs_(mdp, s, gamma, V).values()) if len(mdp.A(s))>0 else 0    
            Delta = max(Delta, np.abs(v - V[s])) 
        if verbose:
            print(i, Delta)
        if Delta < theta:
            break
    # Turn the value-function into a policy. It implements the last line of the algorithm. 
    pi = values2policy(mdp, V, gamma)
    return pi, V

def values2policy(mdp, V, gamma):
    """ Turn the value-function V into a policy. The value function V is implemented as a dictionary so that 
    > value = V[s] 
    is the value-function in state s. 
    The procedure you implement is the very last line of the value-iteration algorithm (SB18, Section 4.4), and it should return
    a policy pi as a dictionary so that
    > a = pi[s]
    is the action in state s.

    Note once again you can re-use the qs_-function. and the argmax -- in fact, the solution is very similar to your solution to the 
    policy-iteration problem in policy_iteration.py. 
    As you have properly noticed, even though we implement different algorithms, they are all build using the same 
    building-block.
    """
    pi = {}
    for s in mdp.nonterminal_states:
        # Create the policy here. pi[s] = a is the action to be taken in state s.
        # You can use the qs_ helper function to simplify things and perhaps
        # re-use ideas from the dp.py problem from week 2.
        # TODO: 2 lines missing.
        Q = {a: v-(1e-8*a if isinstance(a, int) else 0) for a,v in qs_(mdp, s, gamma, V).items()} 
        pi[s] = max(Q, key=Q.get) 
    return pi

if __name__ == "__main__":
    import seaborn as sns
    from irlc.ex09.small_gridworld import SmallGridworldMDP, plot_value_function
    env = SmallGridworldMDP()
    policy, v = value_iteration(env, gamma=0.99, theta=1e-6)
    plot_value_function(env, v)

    plt.title("Value function obtained using value iteration to find optimal policy")
    savepdf("value_iteration")
    plt.show()
