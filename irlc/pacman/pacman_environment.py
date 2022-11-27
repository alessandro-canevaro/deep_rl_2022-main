from irlc.pacman.gamestate import Directions, ClassicGameRules
from irlc.pacman.layout import getLayout
from irlc.pacman.pacman_text_display import PacmanTextDisplay
from irlc.pacman.pacman_graphics_display import PacmanGraphics, FirstPersonPacmanGraphics
from irlc.pacman.pacman_utils import PacAgent, RandomGhost
from irlc.pacman.layout import Layout
import gym
from gym import RewardWrapper
from irlc.utils.common import ExplicitActionSpace
from pyglet.window import key


class GymPacmanEnvironment(gym.Env):
    """
    A fairly messy pacman environment class. I do not recommend reading this code.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 20
    }

    def __init__(self, animate_movement=False, layout='mediumGrid', zoom=2.0, num_ghosts=4, frames_per_second=30, ghost_agent=None, layout_str=None):
        self.metadata['video_frames_per_second'] = frames_per_second
        self.ghosts = [ghost_agent(i+1) if ghost_agent is not None else RandomGhost(i+1) for i in range(num_ghosts)]

        # Set up action space. Use this P-construction so action space can depend on state (grr. gym).
        class P:
            def __getitem__(self, state):
                return {pm_action: "new_state" for pm_action in state.A()}
        self.P = P()
        self.action_space = ExplicitActionSpace(self) # Wrapper environments copy the action space.

        # Load level layout
        if layout_str is not None:
            self.layout = Layout([line.strip() for line in layout_str.strip().splitlines()])
        else:
            self.layout = getLayout(layout)
            if self.layout is None:
                raise Exception("Layout file not found", layout)
        self.rules = ClassicGameRules(30)
        self.options_frametime = 1/frames_per_second
        self.game = None

        # Setup displays.
        self.first_person_graphics = False
        self.animate_movement = animate_movement
        self.options_zoom = zoom
        self.text_display = PacmanTextDisplay(1 / frames_per_second)
        self.graphics_display = None

        # temporary variables for animation/visualization. Don't remove.
        self.visitedlist = None
        self.ghostbeliefs = None
        self.path = None

    def reset(self):
        self.game = self.rules.newGame(self.layout, PacAgent(index=0), self.ghosts, quiet=True, catchExceptions=False)
        self.game.numMoves = 0
        return self.state, {}

    def close(self):
        if self.graphics_display is not None:
            if self.graphics_display.viewer is not None:
                self.graphics_display.viewer.close()
            self.graphics_display.viewer = None

    @property
    def state(self):
        if self.game is None:
            return None
        return self.game.state.deepCopy()

    def get_keys_to_action(self):
        return {(key.LEFT,): Directions.WEST,
                (key.RIGHT,): Directions.EAST,
                (key.UP,): Directions.NORTH,
                (key.DOWN,): Directions.SOUTH,
                (key.S,): Directions.STOP,
                }

    def step(self, action):
        r_ = self.game.state.getScore()
        done = False
        if action not in self.P[self.game.state]:
            raise Exception(f"Agent tried {action=} available actions {self.P[self.game.state]}")

        # Let player play `action`, then let the ghosts play their moves in sequence.
        for agent_index in range(len(self.game.agents)):
            a = self.game.agents[agent_index].getAction(self.game.state) if agent_index > 0 else action
            self.game.state = self.game.state.f(a)
            self.game.rules.process(self.game.state, self.game)

            if self.graphics_display is not None and self.animate_movement and agent_index == 0:
                self.graphics_display.update(self.game.state.data, animate=self.animate_movement, ghostbeliefs=self.ghostbeliefs, path=self.path, visitedlist=self.visitedlist)

            done = self.game.gameOver or self.game.state.isWin() or self.game.state.isLose()
            if done:
                break
        reward = self.game.state.getScore() - r_
        return self.state, reward, done, {}



    def render(self, mode='human', visitedlist=None, ghostbeliefs=None, path=None):
        if mode in ["human", 'rgb_array']:
            if self.graphics_display is None:
                if self.first_person_graphics:
                    self.graphics_display = FirstPersonPacmanGraphics(self.options_zoom, showGhosts=True,
                                                                      frameTime=self.options_frametime)
                    self.graphics_display.ghostbeliefs = self.ghostbeliefs
                else:
                    self.graphics_display = PacmanGraphics(self.options_zoom, frameTime=self.options_frametime)

            if not hasattr(self.graphics_display, 'viewer'):
                self.graphics_display.initialize(self.game.state.data)

            # We save these because the animation code may need it in step()
            self.visitedlist = visitedlist
            self.path = path
            self.ghostbeliefs = ghostbeliefs
            self.graphics_display.master_render(self.game.state.data, ghostbeliefs=ghostbeliefs, path=path, visitedlist=visitedlist)
            return self.graphics_display.viewer.render(return_rgb_array=mode == "rgb_array")

        elif mode in ['ascii']:
            return self.text_display.draw(self.game.state)
        else:
            raise Exception("Bad video mode", mode)

    @property
    def viewer(self):
        if self.graphics_display is not None and hasattr(self.graphics_display, 'viewer'):
            return self.graphics_display.viewer
        else:
            return None


class PacmanWinWrapper(RewardWrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self.env.game.state.isWin():
            reward = 1
        else:
            reward = 0
        return observation, reward, done, info


if __name__ == "__main__":
    from irlc import VideoMonitor
    import time
    from irlc.utils.player_wrapper import PlayWrapper
    from irlc.ex01.agent import Agent, train

    env = GymPacmanEnvironment(layout='mediumClassic', animate_movement=True)
    env = VideoMonitor(env)
    experiment = "experiments/pacman_q"
    if True:
        agent = Agent(env)
        agent = PlayWrapper(agent, env)
        train(env, agent, num_episodes=1)

    env.unwrapped.close()
    time.sleep(0.1)
    env.close()
# 230 174, 159
