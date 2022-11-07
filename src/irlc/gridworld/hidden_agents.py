from collections import defaultdict
import numpy as np
from irlc import TabularAgent, PlayWrapper, VideoMonitor, train
from irlc.ex09.policy_evaluation import qs_
from irlc.gridworld.gridworld_environments import BookGridEnvironment

class ValueIterationAgent2(TabularAgent):
    def __init__(self, env, gamma=.99, epsilon=0, theta=1e-5):
        self.v = defaultdict(lambda: 0)
        self.steps = 0
        self.mdp = env.mdp
        super().__init__(env, gamma, epsilon=epsilon)

    def pi(self, s, k=None): #!f
        q = qs_(self.mdp, s, self.gamma, self.v)
        a = max(q, key=q.get) # optimal action
        return self.random_pi(s) if np.random.rand() < self.epsilon else a

    @property
    def label(self):
        label = f"Value iteration after {self.steps} steps"
        return label

    def v2Q(self, s): # used for rendering right now
        return qs_(self.mdp, s, self.gamma, self.v)

    def train(self, s, a, r, sp, done=False):
        delta = 0
        v2 = {}
        for s in self.env.P.keys():
            v, v2[s] = self.v[s], max(qs_(self.mdp, s, self.gamma, self.v).values()) if len(self.mdp.A(s)) > 0 else 0
            delta = max(delta, np.abs(v - self.v[s]))

        self.v = v2

        for s in self.mdp.nonterminal_states:
            for a in self.mdp.A(s):
                self.Q[s,a] = self.v2Q(s)[a]

        self.delta = delta
        self.steps += 1

    def __str__(self):
        return f"VIAgent_{self.gamma}"


class PolicyEvaluationAgent2(TabularAgent):
    def __init__(self, env, mdp=None, gamma=0.99, steps_between_policy_improvement=10):
        if mdp is None:
            mdp = env.mdp
        self.mdp = mdp
        self.v = defaultdict(lambda: 0)
        self.imp_steps = 0
        self.steps_between_policy_improvement = steps_between_policy_improvement
        self.steps = 0
        self.policy = {}
        for s in mdp.nonterminal_states:
            self.policy[s] = {}
            for a in mdp.A(s):
                self.policy[s][a] = 1/len(mdp.A(s))
        super().__init__(env, gamma)


    def pi(self, s,k=None):  # !f
        a, pa = zip(*self.policy[s].items())
        return np.random.choice(a, p=pa)

    def v2Q(self, s):  # used for rendering right now
        return qs_(self.mdp, s, self.gamma, self.v)

    @property
    def label(self):
        if self.steps_between_policy_improvement is None:
            label = f"Policy evaluation after {self.steps} steps"
        else:
            dd = self.steps % self.steps_between_policy_improvement == 0
            print(dd)
            label = f"PI after {self.steps} steps/{self.imp_steps-dd} policy improvements"
        return label

    def train(self, s, a, r, sp, done=False):
        v2 = {}
        for s in self.mdp.nonterminal_states:
            q = qs_(self.mdp, s, self.gamma, self.v)
            if len(q) == 0:
                v2[s] = 0
            else:
                v2[s] = sum( [qv * self.policy[s][a] for a, qv in q.items()] )

        self.v = v2
        for s in self.mdp.nonterminal_states:
            for a,q in self.v2Q(s).items():
                self.Q[s,a] = q

        if self.steps_between_policy_improvement is not None and (self.steps+1) % self.steps_between_policy_improvement == 0:
            self.policy = {}
            for s in self.mdp.nonterminal_states:
                q = qs_(self.mdp, s, self.gamma, self.v)
                if len(q) == 0:
                    continue
                a_ = max(q, key=q.get)  # optimal action
                self.policy[s] = {}
                for a in self.env.P[s]:
                    self.policy[s][a] = 1 if q[a] == max(q.values()) else 0 #if a == a_ else 0

                n = sum(self.policy[s].values())
                for a in self.policy[s]:
                    self.policy[s][a] *= 1/n

            self.imp_steps += 1
        self.steps += 1

    def __str__(self):
        return f"PIAgent_{self.gamma}"


def peval():
    env = BookGridEnvironment()
    from irlc.ex11.q_agent import QAgent
    use_q = False
    if use_q:
        agent = QAgent(env, gamma=0.9)
    else:
        agent = PolicyEvaluationAgent2(env, gamma=0.9)

    agent = PlayWrapper(agent, env)
    if use_q:
        env = VideoMonitor(env, agent=agent, fps=100,
                           continious_recording=True, agent_monitor_keys=('Q',),
                           render_kwargs={'method_label': 'Qlearn'})
    else:
        env = VideoMonitor(env, agent=agent, fps=100,
                           continious_recording=True, agent_monitor_keys=('v', 'pi', 'v2Q', "label"),
                           render_kwargs={'method_label': 'PI'})

    env.reset()
    train(env, agent, num_episodes=10, max_steps=100)
    env.close()


##### new pol. eval agent.
"""
Replicate and solves experiments from

http://ai.berkeley.edu/reinforcement.html
"""

if __name__ == "__main__":
    peval()
