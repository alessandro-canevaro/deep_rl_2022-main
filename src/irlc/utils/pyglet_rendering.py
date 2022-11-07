# from gym.envs.classic_control.rendering import Viewer
from irlc.utils.gym21.pyglet_rendering import Viewer

from pyglet import gl as gl
from pyglet.graphics import OrderedGroup, Batch
from pyglet.gl import glLineWidth

class PG:
    def __init__(self, pg):
        self.render = pg.draw


class PygletViewer:
    def __init__(self, screen_width=800, xmin=0., xmax=800., ymin=0., ymax=600., title="Gym window"):
        screen_height = int(screen_width / (xmax - xmin) * (ymax-ymin))
        self.viewer = Viewer(screen_width, screen_height)
        self.viewer.window.set_caption(title)
        self.viewer.set_bounds(xmin, xmax, ymin, ymax)
        self.viewer.geoms.append(PG(self))
        self.batch = Batch()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def draw(self):
        self.batch.draw()

    def render(self, *args, **kwargs):
        return self.viewer.render(*args, **kwargs)

    @property
    def window(self):
        return self.viewer.window


class CameraGroup(OrderedGroup):
    RAD2DEG = 57.29577951308232

    def __init__(self, order=0, pg=None, *args, **kwargs):
        super().__init__(order=order, *args, **kwargs)
        self.pg = pg
        self.x, self.y, self.radians, self._scale = 0, 0, 0, 1

    def translate(self, x, y):
        self.x = x
        self.y = y

    def rotate(self, radians):
        self.radians = radians

    def scale(self, s):
        self._scale = s

    def set_state(self):
        if self.pg is not None:
            self.pg.set_state()
        gl.glPushMatrix()
        gl.glTranslatef(self.x, self.y, 0.0)
        gl.glRotatef(self.RAD2DEG * self.radians, 0, 0, 1.0)
        factor = self._scale
        gl.glScalef(factor, factor, 1.0)

    def unset_state(self):
        gl.glPopMatrix()
        if self.pg is not None:
            self.pg.unset_state()

    def __str__(self):
        return f"CameraGroup({self.order}, scale={self._scale})"

    def __repr__(self):
        return str(self)


class GroupedElement:
    def __init__(self, batch, pg=None, parent=None, order=None):
        # print("Init called", order)
        self.batch = batch
        if order is None:
            # print("ord is notne", pg, pg.order)
            if pg is not None:
                # print("Setting order from pg", pg, pg.order)
                order = pg.order + 1
            else:
                order = 0
        self.group = CameraGroup(order=order, pg=pg, parent=parent)
        self.render()

    def render(self):
        pass


class PolygonOutline(GroupedElement):
    def __init__(self, batch, coords, outlineColor, fillColor=None, filled=True, width=1.0, closed=False):
        if fillColor == None:
            fillColor = outlineColor
        if not filled:
            fillColor = ""

        self.coords = coords
        self.outlineColor = outlineColor
        self.fillColor = fillColor
        self.filled = filled
        self.width = width
        self.closed = closed
        super().__init__(batch, order=0)

    def render(self):
        # def DrawPolygon(self, vertices, color):
        """
        Draw a wireframe polygon given the world vertices (tuples) with the specified color.
        """
        vertices = self.coords
        r,g,b = self.outlineColor
        cl = (r/255, g/255, b/255) # tuple( (c/255 for c in self.outlineColor) )
        glLineWidth(self.width)
        if len(vertices) == 2:
            p1, p2 = vertices
            self.vl = self.batch.add(2, gl.GL_LINES, None,
                           ('v2f', (p1[0], p1[1], p2[0], p2[1])),
                           ('c3f', list(cl) * 2))
        else:
            ll_count, ll_vertices = line_loop(vertices)
            colors = cl * ll_count
            self.vl = self.batch.add(ll_count, gl.GL_LINES, None,
                           ('v2f', ll_vertices),
                           ('c3f', colors)
                                     )


def line_loop(vertices):
    """
    in: vertices arranged for gl_line_loop ((x,y),(x,y)...)
    out: vertices arranged for gl_lines (x,y,x,y,x,y...)
    see https://github.com/openai/box2d-py/blob/master/examples/backends/pyglet_framework.py
    """
    out = []
    for i in range(len(vertices) - 1):
        # 0,1  1,2  2,3 ... len-1,len  len,0
        out.extend(vertices[i])
        out.extend(vertices[i + 1])

    out.extend(vertices[len(vertices) - 1])
    out.extend(vertices[0])

    return len(out) // 2, out
