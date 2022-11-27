import math
import numpy as np
from PIL import ImageColor
from pyglet.shapes import Circle, Rectangle, Polygon, Sector
from irlc.utils.pyglet_rendering import GroupedElement
from irlc.pacman.pacman_graphics_display import GHOST_COLORS, GHOST_SHAPE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Eye(GroupedElement):
    normal, cross = None, None

    def render(self):
        self.normal = [Circle(0, 0, .2, color=WHITE, batch=self.batch, group=self.group),
                       Circle(0, 0, 0.1, color=BLACK, batch=self.batch, group=self.group)]  # radius was 0.08
        ew = 0.6
        rw = ew/6
        self.cross = [Rectangle(x=-ew/2, y=-rw/2, width=ew, height=rw, color=BLACK, group=self.group, batch=self.batch),
                      Rectangle(x=-rw/2, y=-ew/2, width=rw, height=ew, color=BLACK, group=self.group, batch=self.batch)]
        self.set_eye_dir('stop')

    def set_eye_dir(self, direction='stop'):
        dead = direction.lower() == 'dead'
        for n in self.normal:
            n.visible = not dead
            pp = (0, 0)
            if direction.lower() == 'stop':
                pass
            dd = 0.1
            if direction.lower() == 'east':
                pp = (dd, 0)
                # self.group.translate(dd, 0)

            if direction.lower() == 'west':
                pp = (-dd, 0)
                # self.group.translate(-dd, 0)
            if direction.lower() == 'south':
                pp = (0, -dd)
                # self.group.translate(0, -dd)
            if direction.lower() == 'north':
                # self.group.translate(0, dd)
                pp = (0, dd)
            self.normal[1].x = pp[0]
            self.normal[1].y = pp[1]

        for e in self.cross:
            e.visible = dead
        self.group.rotate(np.pi/4 if dead else 0)


class Ghost(GroupedElement):
    body_, eyes_ = None, None

    def __init__(self, batch, agent_index=1, order=1):
        self.agentIndex = agent_index
        super().__init__(batch, order=order)

    def set_scared(self, scared):
        from irlc.pacman.devel.pyglet_pacman_graphics import SCARED_COLOR, GHOST_COLORS
        self.body_.color = SCARED_COLOR if scared else GHOST_COLORS[self.agentIndex]

    def eyes(self, direction):
        for e in self.eyes_:
            e.set_eye_dir(direction)

    def set_position(self, x, y):
        # print("setting position", x,y)
        self.group.x = x
        self.group.y = y
        # self.group.translate(x, y)

    def set_direction(self, direction):
        self.eyes(direction)

    def kill(self):
        self.eyes('dead')
        self.body_.color = ImageColor.getcolor(GHOST_COLORS[3], "RGB")
        self.group.rotate(-np.pi/2)

    def resurrect(self):
        self.eyes('straight')
        self.body_.color = ImageColor.getcolor(GHOST_COLORS[self.agentIndex], "RGB")
        self.group.rotate(0)

    def render(self):
        ghost_shape = tuple((x, -y) for (x, y) in GHOST_SHAPE)
        colour = ImageColor.getcolor(GHOST_COLORS[self.agentIndex], "RGB")
        self.body_ = Polygon(*ghost_shape, color=colour, batch=self.batch, group=self.group)
        self.eyes_ = [Eye(order=self.group.order+1+k, pg=self.group, batch=self.batch) for k in range(2)]
        for k, e in enumerate(self.eyes_):
            e.group.translate(-.3 if k == 0 else .3, .3)


PACMAN_COLOR = (255, 255, 61)


class Pacman(GroupedElement):
    body = None

    def __init__(self, grid_size, batch, pg=None, parent=None, order=0):
        self.delta = 0
        self.GRID_SIZE = grid_size
        super().__init__(batch, pg=pg, parent=parent, order=order)
        self.set_animation(0, 4)

    def set_animation(self, frame, frames):
        pos = frame/frames
        width = 30 + 80 * math.sin(math.pi * pos)
        delta = width / 2
        self.delta = delta * np.pi / 180
        self.body._angle = 2*np.pi-2*self.delta
        self.body._start_angle = self.delta
        self.body._update_position()

    def set_direction(self, direction):
        if direction == 'Stop':
            pass
        else:
            angle = 0
            if direction == 'East':
                angle = 0
            elif direction == 'North':
                angle = np.pi/2
            elif direction == 'West':
                angle = np.pi
            elif direction == 'South':
                angle = np.pi*1.5
            self.group.rotate(angle)

    def render(self):
        width = 30
        delta = width/2
        delta = delta/180 * np.pi
        self.body = Sector(0, 0, self.GRID_SIZE/2, angle=2*np.pi-2*delta, start_angle=delta,
                           color=PACMAN_COLOR, batch=self.batch, group=self.group)