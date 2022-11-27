import gym
from irlc import main_plot
import matplotlib.pyplot as plt
from irlc.gridworld.gridworld_environments import OpenGridEnvironment
from irlc import train
from irlc.ex11.q_agent import QAgent

class SarsaDelayNAgent(QAgent):
    """ Implement the N-step semi-gradient sarsa agent from \cite[Section 7.2]{sutton}"""
    def __init__(self, env, gamma=1, alpha=0.2, epsilon=0.1, n=1):
        # Variables for TD-n
        self.n = n  # as in n-step sarse
        # Buffer lists for previous (S_t, R_{t}, A_t) triplets
        self.R, self.S, self.A = [None] * (self.n + 1), [None] * (self.n + 1), [None] * (self.n + 1)
        super().__init__(env, gamma=gamma, alpha=alpha, epsilon=epsilon)

    def pi(self, s, k=None):
        self.t = k  # Save current step in episode for use in train.
        return self.pi_eps(s)

    def train(self, s, a, r, sp, done=False):
        # Recall we are given S_t, A_t, R_{t+1}, S_{t+1} and done is whether t=T+1.
        n, t = self.n, self.t
        # Store current observations in buffer.
        self.S[t%(n+1)] = s
        self.A[t%(n+1)] = a # self.pi_eps(sp) if not done else -1
        self.R[(t+1)%(n+1)] = r
        if done:
            T = t+1
            tau_steps_to_train = range(t - n, T)
        else:
            T = 1e10
            tau_steps_to_train = [t - n ] if t > 0 else []

        # Tau represent the current tau-steps which are to be updated. The notation is compatible with that in Sutton.
        for tau in tau_steps_to_train:
            if tau >= 0:
                """
                Compute the return for this tau-step and perform the relevant Q-update. 
                The first step is to compute the expected return G in the below section. 
                """
                G = sum([self.gamma**(i-tau-1)*self.R[i%(n+1)] for i in range(tau+1, min(tau+n, T)+1)])
                S_tau_n, A_tau_n = self.S[(tau+n)%(n+1)], self.A[(tau+n)%(n+1)]
                if tau+n < T:
                    G += self.gamma**n * self._q(S_tau_n, A_tau_n)
                S_tau, A_tau = self.S[tau%(n+1)], self.A[tau%(n+1)]
                delta = G - self._q(S_tau, A_tau)

                if n == 1: # Check your implementation is correct when n=1 by comparing it with regular Sarsa learning.
                    delta_Sarsa = (self.R[ (tau+1)%(n+1) ] + (0 if tau+n==T else self.gamma * self._q(S_tau_n,A_tau_n)) - self._q(S_tau,A_tau))
                    if abs(delta-delta_Sarsa) > 1e-10:
                        raise Exception("n=1 agreement with Sarsa learning failed. You have at least one bug!")
                self._upd_q(S_tau, A_tau, delta)

    def _q(self, s, a): return self.Q[s,a] # Using these helper methods will come in handy when we work with function approximators, but it is optional.
    def _upd_q(self, s, a, delta): self.Q[s,a] += self.alpha * delta

    def __str__(self):
        return f"SarsaN_{self.gamma}_{self.epsilon}_{self.alpha}_{self.n}"

from irlc.ex11.nstep_sarsa_agent import SarsaNAgent
from irlc.lectures.lec11.lecture_10_sarsa_open import open_play
if __name__ == "__main__":
    n = 8
    env = OpenGridEnvironment()
    agent = SarsaDelayNAgent(env, n=n)
    train(env, agent, num_episodes=100)

    open_play(SarsaDelayNAgent, method_label=f"Sarsa n={n}", n=n)

