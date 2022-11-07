from collections import defaultdict
import numpy as np
from irlc import Agent
from mazeenv.maze_environment import MazeEnvironment
from irlc.gridworld.gridworld_environments import CliffGridEnvironment, BookGridEnvironment
from irlc import PlayWrapper, VideoMonitor
from irlc import Agent, train

class VIAgent(Agent):
    def __init__(self, env):
        self.v = defaultdict(lambda: 0)
        super().__init__(env)

    def pi(self, s, k=None):
        # take random actions and randomize the value function V for visualization.
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                if s[i,j,0] == 0:
                    self.v[(i,j)] = np.random.rand()-0.5

        return self.env.action_space.sample()

    def train(self, s, a, r, sp, done):
        # do training stuff here (save to buffer, call torch, etc.)
        pass


class ShittyVIAgent(Agent):
    def __init__(self, env):
        self.v = defaultdict(lambda: 0)
        self.k = 0
        super().__init__(env)

    def pi(self, s, k=None):
        # take random actions and randomize the value function V for visualization.
        return self.env.action_space.sample()

    def Phi(self, s):
        rout = s[:, :, 2]
        rin = s[:, :, 2] - 0.05  # Simulate a small transition cost.
        p = 1 - s[:, :, 0]
        return (rin, rout, p)

    def train(self, s, a, r, sp, done=False):
        # do training stuff here (save to buffer, call torch, whatever)
        w = VIP(s, self.Phi(s))
        vv = w[:,:,min(self.k, w.shape[2]-1)]
        for i,j in self.v:
            self.v[i,j] = vv[j, i] # annoying transpose
        self.k += 1
        if done:
            self.k = 0

def VIP(s, Phi, K=20):
    (rin, rout, p) = Phi
    h, w = s.shape[0], s.shape[1]
    v = np.zeros((h,w, K+1))
    for k in range(K):
        for i in range(h):
            for j in range(w):
                v[i,j, k+1] = v[i,j,k]
                for di, dj in [ (-1, 0), (1, 0), (0, -1), (0, 1)]:
                    if di + i < 0 or dj + j < 0:
                        continue
                    if di + i >= h or dj + j >= w:
                        continue
                    ip = i + di
                    jp = j + dj
                    nv = p[i,j] * v[ip, jp,k] + rin[ip, jp] - rout[i,j]
                    v[i,j,k+1] = max( v[i,j,k+1], nv)
    return v


if __name__ == "__main__":
    env = MazeEnvironment(size=10, render_mode='human')
    agent = ShittyVIAgent(env)
    agent = PlayWrapper(agent, env, autoplay=False)
    experiment = "experiments/q1_value_iteration"
    env = VideoMonitor(env, agent=agent, fps=100, continious_recording=True, agent_monitor_keys=('v', ), render_kwargs={'method_label': 'VI-K'})
    train(env, agent, experiment_name=experiment, num_episodes=10, max_steps=100)
    
    
    
    env.close()
