# graphicsUtils.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import numpy as np
import pyglet
# import irlc.utils.gym21.pyglet_rendering as rendering
# from irlc.utils.gym21.pyglet_rendering import Viewer, glBegin, GL_LINES, glVertex2f, glEnd

from gym.envs.classic_control import rendering
from gym.envs.classic_control.rendering import Viewer
from gym.envs.classic_control.rendering import glBegin, GL_LINES, glVertex2f, glEnd

ghost_shape = [
    (0, - 0.5),
    (0.25, - 0.75),
    (0.5, - 0.5),
    (0.75, - 0.75),
    (0.75, 0.5),
    (0.5, 0.75),
    (- 0.5, 0.75),
    (- 0.75, 0.5),
    (- 0.75, - 0.75),
    (- 0.5, - 0.5),
    (- 0.25, - 0.75)
]

def _adjust_coords(coord_list, x, y):
    for i in range(0, len(coord_list), 2):
        coord_list[i] = coord_list[i] + x
        coord_list[i + 1] = coord_list[i + 1] + y
    return coord_list

def formatColor(r, g, b):
    return '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))

def colorToVector(color):
    return list(map(lambda x: int(x, 16) / 256.0, [color[1:3], color[3:5], color[5:7]]))

def h2rgb(color):
    if color is None:
        return None
    if color.startswith("#"):
        color = color[1:]
    return tuple(int(color[i:i + 2], 16) / 255 for i in (0, 2, 4))

class GraphicsCache:
    break_cache = False
    def __init__(self, viewer, verbose=False):
        self.viewer = viewer
        # self._items_in_viewer = {}
        # self._seen_things = set()
        self.clear()
        self.verbose = verbose

    def copy_all(self):
        self._seen_things.update( set( self._items_in_viewer.keys() ) )

    def clear(self):
        self._seen_things = set()
        self.viewer.geoms.clear()
        self._items_in_viewer = {}

    def prune_frame(self):
        s0 = len(self._items_in_viewer)
        self._items_in_viewer = {k: v for k, v in self._items_in_viewer.items() if k in self._seen_things }
        if self.verbose:
            print("removed", len(self._items_in_viewer) - s0,  "geom size", len(self._items_in_viewer))
        self.viewer.geoms = list( self._items_in_viewer.values() )
        self._seen_things = set()


    def add_geometry(self, name, geom):
        if self.break_cache:
            if self._items_in_viewer == None:
                self.viewer.geoms = []
                self._items_in_viewer = {}

        self._items_in_viewer[name] = geom
        self._seen_things.add(name)

class GraphicsUtilGym:
    viewer = None
    _canvas_xs = None      # Size of canvas object
    _canvas_ys = None
    _canvas_x = None      # Current position on canvas
    _canvas_y = None

    def begin_graphics(self, width=640, height=480, color=formatColor(0, 0, 0), title=None, local_xmin_xmax_ymin_ymax=None, verbose=False):
        """ Main interface for managing graphics.
        The local_xmin_xmax_ymin_ymax controls the (local) coordinate system which is mapped onto screen coordinates. I.e. specify this
        to work in a native x/y coordinate system. If not, it will default to screen coordinates familiar from Gridworld.
        """
        if height % 2 == 1:
            height += 1 # Must be divisible by 2.
        self._bg_color = color
        viewer = Viewer(width=int(width), height=int(height))
        viewer.window.set_caption(title)
        self.viewer = viewer
        self.gc = GraphicsCache(viewer, verbose=verbose)
        self._canvas_xs, self._canvas_ys = width - 1, height - 1
        self._canvas_x, self._canvas_y = 0, self._canvas_ys
        if local_xmin_xmax_ymin_ymax is None:
            # local_coordinates = []
            # This will by default flip the xy coordinate system in the y-direction. This is the
            # expected behavior in GridWorld.
            local_xmin_xmax_ymin_ymax = (0, width, height, 0)
        self._local_xmin_xmax_ymin_ymax = local_xmin_xmax_ymin_ymax

        return viewer

    def end_frame(self):
        self.gc.prune_frame()

    def close(self):
        self.viewer.close()

    def draw_background(self):
        # print("drawing bg")
        x1, x2, y1, y2 = self._local_xmin_xmax_ymin_ymax
        # corners = [(0,0), (0, self._canvas_ys), (self._canvas_xs, self._canvas_ys), (self._canvas_xs, 0)]
        corners = [ (x1, y1), (x2, y1), (x2, y2), (x1, y2)  ]
        # print(corners)
        # for s in corners:
        self.polygon(name="background", coords=corners, outlineColor=self._bg_color, fillColor=self._bg_color, filled=True, smoothed=False)

    def fixxy(self, xy):
        x,y = xy
        x = (x - self._local_xmin_xmax_ymin_ymax[0]) / (self._local_xmin_xmax_ymin_ymax[1] - self._local_xmin_xmax_ymin_ymax[0]) * self.viewer.width
        y = (y - self._local_xmin_xmax_ymin_ymax[2]) / (self._local_xmin_xmax_ymin_ymax[3] - self._local_xmin_xmax_ymin_ymax[2]) * self.viewer.height
        return int(x), int(y)


    def plot(self, name, x, y, color=None, width=1.0):
        coords = [(x_,y_) for (x_, y_) in zip(x,y)]
        if color is None:
            color = "#000000"
        return self.polygon(name, coords, outlineColor=color, filled=False, width=width)

    def polygon(self, name, coords, outlineColor, fillColor=None, filled=True, smoothed=1, behind=0, width=1.0, closed=False):
        c = []
        for coord in coords:
            c.append(coord[0])
            c.append(coord[1])

        if fillColor == None: fillColor = outlineColor
        if not filled: fillColor = ""
        from gym.envs.classic_control import rendering
        c = [self.fixxy(tuple(c[i:i+2])) for i in range(0, len(c), 2)]
        if not filled:
            poly = rendering.PolyLine(c, close=closed)
            poly.set_linewidth(width)
            poly.set_color(*h2rgb(outlineColor))
        else:
            poly = rendering.FilledPolygon(c)
            poly.set_color(*h2rgb(fillColor))
            poly.add_attr(rendering.LineWidth(width))

        if len(outlineColor) > 0 and filled: # Not sure why this cannot be merged with the filled case...
            outl = rendering.PolyLine(c, close=True)
            outl.set_linewidth(width)
            outl.set_color(*h2rgb(outlineColor))
        if poly is not None:
            self.gc.add_geometry(name, poly)
        else:
            raise Exception("Bad polyline")
        return poly

    def square(self, name, pos, r, color, filled=1, behind=0):
        x, y = pos
        coords = [(x - r, y - r), (x + r, y - r), (x + r, y + r), (x - r, y + r)]
        return self.polygon(name, coords, color, color, filled, 0, behind=behind)

    def circle(self, name, pos, r, outlineColor, fillColor, endpoints=None, style='pieslice', width=2):
        x, y = pos
        if endpoints == None:
            e = [0, 359]
        else:
            e = list(endpoints)
        while e[0] > e[1]: e[1] = e[1] + 360
        if endpoints is not None and len(endpoints) > 0:
            tt = np.linspace(e[0]/360 * 2*np.pi, e[-1]/360 * 2*np.pi, int(r*20) )
            px = np.cos(tt) * r
            py = -np.sin(tt) * r
            pp = list(zip(px.tolist(), py.tolist()))
            if style == 'pieslice':
                pp = [(0,0),] + pp + [(0,0),]
            pp = [( (x+a, y+b)) for (a,b) in pp  ]
            if style == 'arc':
                pp = [self.fixxy(p_) for p_ in pp]
                outl = rendering.PolyLine(pp, close=False)
                outl.set_linewidth(width)
                outl.set_color(*h2rgb(outlineColor))
                self.gc.add_geometry(name, outl)
            elif style == 'pieslice':
                self.polygon(name, pp, fillColor=fillColor, outlineColor=outlineColor, width=width)
            else:
                raise Exception("bad style", style)
        else:
            circ = rendering.make_circle(r)
            circ.set_color(*h2rgb(fillColor))
            tf = rendering.Transform(translation = self.fixxy(pos))
            circ.add_attr(tf)
            self.gc.add_geometry(name, circ)


    def moveCircle(self, id, pos, r, endpoints=None):
        # global _canvas_x, _canvas_y
        x, y = pos
        x0, x1 = x - r - 1, x + r
        y0, y1 = y - r - 1, y + r
        if endpoints == None:
            e = [0, 359]
        else:
            e = list(endpoints)
        while e[0] > e[1]: e[1] = e[1] + 360
        self.edit(id, ('start', e[0]), ('extent', e[1] - e[0]))

    def edit(id, *args):
        pass


    def text(self, name, pos, color, contents, font='Helvetica', size=12, style='normal', anchor="w"):
        x_, y_ = self.fixxy(pos)
        ax = "center"
        ax = "left" if anchor == "w" else ax
        ax = "right" if anchor == "e" else ax
        ay = "center"
        ay = "baseline" if anchor == "s" else ay
        ay = "top" if anchor == "n" else ay
        psz = int(-size * 0.75) if size < 0 else size
        cl = tuple(int(c*255) for c in h2rgb(color) )+(255,)
        label = pyglet.text.Label(contents, x=int(x_), y=int(y_),  font_name='Arial', font_size=psz, bold=style=="bold",
                                  color=cl,
                                  anchor_x=ax, anchor_y=ay)

        self.gc.add_geometry(name, TextGeom(label))

    def line(self, name, here, there, color=formatColor(0, 0, 0), width=2):
        poly = MyLine(self.fixxy(here), self.fixxy(there), width=width)
        poly.set_color(*h2rgb(color))
        poly.add_attr(rendering.LineWidth(width))
        self.gc.add_geometry(name, poly)
        # return None

class MyLine(rendering.Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), width=1):
        rendering.Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = rendering.LineWidth(width)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

class TextGeom(rendering.Geom):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def render(self):
        self.label.draw()

# 547 lines
# 556 lines
# 270