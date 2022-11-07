import numpy as np
from collections import defaultdict
from pyglet.window import key
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from gym.spaces.discrete import Discrete
from irlc.ex09.mdp import MDP2GymEnv
from irlc.gridworld.gridworld_mdp import GridworldMDP, FrozenGridMDP
from irlc.gridworld import gridworld_graphics_display
from irlc import Timer

grid_cliff_grid = [[' ',' ',' ',' ',' ', ' ',' ',' ',' ',' ', ' '],
                   [' ',' ',' ',' ',' ', ' ',' ',' ',' ',' ', ' '],
                   ['S',-100, -100, -100, -100,-100, -100, -100, -100, -100, 0]]

grid_cliff_grid2 = [[' ',' ',' ',' ',' '],
                    ['S',' ',' ',' ',' '],
                     [-100,-100, -100, -100, 0]]

grid_discount_grid = [[' ',' ',' ',' ',' '],
                    [' ','#',' ',' ',' '],
                    [' ','#', 1,'#', 10],
                    ['S',' ',' ',' ',' '],
                    [-10,-10, -10, -10, -10]]

grid_bridge_grid = [[ '#',-100, -100, -100, -100, -100, '#'],
        [   1, 'S',  ' ',  ' ',  ' ',  ' ',  10],
        [ '#',-100, -100, -100, -100, -100, '#']]

grid_book_grid = [[' ',' ',' ',+1],
        [' ','#',' ',-1],
        ['S',' ',' ',' ']]

grid_maze_grid = [[' ',' ',' ', +1],
                  ['#','#',' ','#'],
                  [' ','#',' ',' '],
                  [' ','#','#',' '],
                  ['S',' ',' ',' ']]

sutton_corner_maze = [[  1, ' ', ' ', ' '], #!s=corner
                      [' ', ' ', ' ', ' '],
                      [' ', 'S', ' ', ' '],
                      [' ', ' ', ' ',   1]] #!s=corner

# A big yafcport open maze.
grid_open_grid = [[' ']*8 for _ in range(5)]
grid_open_grid[0][0] = 'S'
grid_open_grid[-1][-1] = 1


class GridworldEnvironment(MDP2GymEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1000,
    }
    def get_keys_to_action(self):
        return {(key.LEFT,): GridworldMDP.WEST, (key.RIGHT,): GridworldMDP.EAST, (key.UP,): GridworldMDP.NORTH, (key.DOWN,): GridworldMDP.SOUTH}

    def _get_mdp(self, grid, uniform_initial_state=False):
        return GridworldMDP(grid, living_reward=self.living_reward)

    def __init__(self, grid=None, uniform_initial_state=True, living_reward=0,zoom=1, view_mode=0, **kwargs):
        self.living_reward = living_reward
        mdp = self._get_mdp(grid)
        super().__init__(mdp)
        self.display = None
        self.viewer = None
        self.action_space = Discrete(4)
        self.render_episodes = 0
        self.render_steps = 0
        self.zoom = zoom
        self.timer = Timer()
        self.view_mode = view_mode

        def _step(*args, **kwargs):
            o = type(self).step(self, *args, **kwargs)
            done = o[2]
            self.render_steps += 1
            self.render_episodes += done
            return o
        self.step = _step

    def keypress(self, key):
        if key == 116:
            self.view_mode += 1
            self.render()

    def render(self, mode='human', state=None, agent=None, v=None, Q=None, pi=None, policy=None, v2Q=None, gamma=0, method_label="", label=None):
        self.agent = agent
        if label is None:
            label = f"{method_label} AFTER {self.render_steps} STEPS"
        speed = 1
        gridSize = int(150 * self.zoom)
        # print("In environment - render")
        if self.display is None:
            self.display = gridworld_graphics_display.GraphicsGridworldDisplay(self.mdp, gridSize)
            self.viewer = self.display.ga.gc.viewer

        if state is None:
            state = self.state

        avail_modes = []
        if agent != None:
            label = (agent.label if hasattr(agent, 'label') else method_label) if label is None else label
            v = agent.v if hasattr(agent, 'v') else None
            Q = agent.Q if hasattr(agent, 'Q') else None
            policy = agent.policy if hasattr(agent, 'policy') else None
            v2Q = agent.v2Q if hasattr(agent, 'v2Q') else None
            avail_modes = []
            if Q is not None:
                avail_modes.append("Q")
                avail_modes.append("v")
            elif v is not None:
                avail_modes.append("v")

        if len(avail_modes) > 0:
            self.view_mode = self.view_mode % len(avail_modes)
            if avail_modes[self.view_mode] == 'v':
                preferred_actions = None

                if v == None:
                    preferred_actions = {}
                    v = {s: Q.max(s) for s in self.mdp.nonterminal_states}
                    for s in self.mdp.nonterminal_states:
                        acts, values = Q.get_Qs(s)
                        preferred_actions[s] = [a for (a,w) in zip(acts, values) if np.round(w, 2) == np.round(v[s], 2)]

                if v2Q is not None:
                    preferred_actions = {}
                    for s in self.mdp.nonterminal_states:
                        q = v2Q(s)
                        mv = np.round( max( q.values() ), 2)
                        preferred_actions[s] = [k for k, v in q.items() if np.round(v, 2) == mv]

                if agent != None and hasattr(agent, 'policy') and agent.policy is not None and state in agent.policy and isinstance(agent.policy[state], dict):
                    for s in self.mdp.nonterminal_states:
                        preferred_actions[s] = [a for a, v in agent.policy[s].items() if v == max(agent.policy[s].values()) ]

                if hasattr(agent, 'returns_count'):
                    returns_count = agent.returns_count
                else:
                    returns_count = None
                if hasattr(agent, 'returns_sum'):
                    returns_sum = agent.returns_sum
                else:
                    returns_sum = None
                self.display.displayValues(mdp=self.mdp, v=v, preferred_actions=preferred_actions, currentState=state, message=label, returns_count=returns_count, returns_sum=returns_sum)

            elif avail_modes[self.view_mode] == 'Q':

                if hasattr(agent, 'e') and isinstance(agent.e, defaultdict):
                    eligibility_trace = defaultdict(float)
                    for k, v in agent.e.items():
                        eligibility_trace[k] = v

                else:
                    eligibility_trace = None
                    # raise Exception("bad")
                # print(eligibility_trace)
                self.display.displayQValues(self.mdp, Q, currentState=state, message=label, eligibility_trace=eligibility_trace)
            else:
                raise Exception("No view mode selected")
        else:
            self.display.displayNullValues(self.mdp, currentState=state)

        self.display.end_frame()
        render_out = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return render_out

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class BookGridEnvironment(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_book_grid, *args, **kwargs)

class BridgeGridEnvironment(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_bridge_grid, *args, **kwargs)

class CliffGridEnvironment(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_cliff_grid, living_reward=-1, *args, **kwargs)

class CliffGridEnvironment2(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_cliff_grid2, living_reward=-1, *args, **kwargs)


class OpenGridEnvironment(GridworldEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(grid_open_grid, *args, **kwargs)

"""  #!s=corner
Implement Suttons little corner-maze environment (see \cite[Example 4.1]{sutton}).  
You can make an instance using:
> from irlc.gridworld.gridworld_environments import SuttonCornerGridEnvironment
> env = SuttonCornerGridEnvironment()
To get access the the mdp (as a MDP-class instance, for instance to see the states env.mdp.nonterminal_states) use
> env.mdp
You can make a visualization (allowing you to play) and train it as:
> from irlc import PlayWrapper, Agent, train
> agent = Agent()
> agent = PlayWrapper(agent, env) # allows you to play using the keyboard; omit to use agent as usual. 
> train(env, agent, num_episodes=1)
"""
class SuttonCornerGridEnvironment(GridworldEnvironment):
    def __init__(self, *args, living_reward=-1, **kwargs): # living_reward=-1 means the agent gets a reward of -1 per step.
        super().__init__(sutton_corner_maze, *args, living_reward=living_reward, **kwargs) #!s=corner

class SuttonMazeEnvironment(GridworldEnvironment):
    def __init__(self, *args, living_reward=0, **kwargs):
        sutton_maze_grid = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', '#',  +1],
                            [' ', ' ', '#', ' ', ' ', ' ', ' ', '#', ' '],
                            ['S', ' ', '#', ' ', ' ', ' ', ' ', '#', ' '],
                            [' ', ' ', '#', ' ', ' ', ' ', ' ', ' ', ' '],
                            [' ', ' ', ' ', ' ', ' ', '#', ' ', ' ', ' '],
                            [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '] ]

        super().__init__(sutton_maze_grid, *args, living_reward=living_reward, **kwargs)

class FrozenLake(GridworldEnvironment):
    def _get_mdp(self, grid, uniform_initial_state=False):
        return FrozenGridMDP(grid, is_slippery=self.is_slippery, living_reward=self.living_reward)

    def __init__(self, is_slippery=True, living_reward=0, *args, **kwargs):
        self.is_slippery = is_slippery
        menv = FrozenLakeEnv(is_slippery=is_slippery) # Load frozen-lake game layout and convert to our format 'grid'
        gym2grid = dict(F=' ', G=1, H=0)
        grid = [[gym2grid.get(s.decode("ascii"), s.decode("ascii")) for s in l] for l in menv.desc.tolist()]
        super().__init__(grid=grid, *args, living_reward=living_reward, **kwargs)
