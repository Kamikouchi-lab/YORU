#!/usr/bin/python
# -*- coding: utf-8 -*-


try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from .utils import distance
# YORU's single definition of a rotated box: the annotation tool, the detector
# and the real-time drawing all reduce corners to (cx, cy, w, h, theta) the
# same way, so a box means the same thing at every stage of the pipeline.
from yoru.libs.obb import (
    corners_to_obb,
    is_rotated as _is_rotated,
    obb_corners,
    rotate_points,
)
import sys

DEFAULT_LINE_COLOR = QColor(0, 255, 0, 128)
DEFAULT_FILL_COLOR = QColor(255, 0, 0, 128)
DEFAULT_SELECT_LINE_COLOR = QColor(255, 255, 255)
DEFAULT_SELECT_FILL_COLOR = QColor(0, 128, 255, 155)
DEFAULT_VERTEX_FILL_COLOR = QColor(0, 255, 0, 255)
DEFAULT_HVERTEX_FILL_COLOR = QColor(255, 0, 0)


class Shape(object):
    P_SQUARE, P_ROUND = range(2)

    MOVE_VERTEX, NEAR_VERTEX = range(2)

    # The following class variables influence the drawing
    # of _all_ shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    h_vertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8
    scale = 1.0
    label_font_size = 8

    def __init__(self, label=None, line_color=None, difficult=False, paint_label=False):
        self.label = label
        self.points = []
        self.fill = False
        self.selected = False
        self.difficult = difficult
        self.paint_label = paint_label

        self._highlight_index = None
        self._highlight_mode = self.NEAR_VERTEX
        self._highlight_settings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

    def close(self):
        self._closed = True

    def reach_max_points(self):
        if len(self.points) >= 4:
            return True
        return False

    def add_point(self, point):
        if not self.reach_max_points():
            self.points.append(point)

    def pop_point(self):
        if self.points:
            return self.points.pop()
        return None

    def is_closed(self):
        return self._closed

    def set_open(self):
        self._closed = False

    def paint(self, painter):
        if self.points:
            color = self.select_line_color if self.selected else self.line_color
            pen = QPen(color)
            # Try using integer sizes for smoother drawing(?)
            pen.setWidth(max(1, int(round(2.0 / self.scale))))
            painter.setPen(pen)

            line_path = QPainterPath()
            vertex_path = QPainterPath()

            line_path.moveTo(self.points[0])
            # Uncommenting the following line will draw 2 paths
            # for the 1st vertex, and make it non-filled, which
            # may be desirable.
            # self.drawVertex(vertex_path, 0)

            for i, p in enumerate(self.points):
                line_path.lineTo(p)
                self.draw_vertex(vertex_path, i)
            if self.is_closed():
                line_path.lineTo(self.points[0])

            painter.drawPath(line_path)
            painter.drawPath(vertex_path)
            painter.fillPath(vertex_path, self.vertex_fill_color)

            # Draw text at the top-left
            if self.paint_label:
                min_x = sys.maxsize
                min_y = sys.maxsize
                min_y_label = int(1.25 * self.label_font_size)
                for point in self.points:
                    min_x = min(min_x, point.x())
                    min_y = min(min_y, point.y())
                if min_x != sys.maxsize and min_y != sys.maxsize:
                    font = QFont()
                    font.setPointSize(self.label_font_size)
                    font.setBold(True)
                    painter.setFont(font)
                    if self.label is None:
                        self.label = ""
                    if min_y < min_y_label:
                        min_y += min_y_label
                    painter.drawText(min_x, min_y, self.label)

            if self.fill:
                color = self.select_fill_color if self.selected else self.fill_color
                painter.fillPath(line_path, color)

    def draw_vertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlight_index:
            size, shape = self._highlight_settings[self._highlight_mode]
            d *= size
        if self._highlight_index is not None:
            self.vertex_fill_color = self.h_vertex_fill_color
        else:
            self.vertex_fill_color = Shape.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearest_vertex(self, point, epsilon):
        for i, p in enumerate(self.points):
            if distance(p - point) <= epsilon:
                return i
        return None

    def contains_point(self, point):
        return self.make_path().contains(point)

    def make_path(self):
        path = QPainterPath(self.points[0])
        for p in self.points[1:]:
            path.lineTo(p)
        return path

    def bounding_rect(self):
        return self.make_path().boundingRect()

    def move_by(self, offset):
        self.points = [p + offset for p in self.points]

    def move_vertex_by(self, i, offset):
        self.points[i] = self.points[i] + offset

    # -- Oriented bounding boxes ------------------------------------------
    #
    # A rotated box needs no new storage: it is still the four corners in
    # self.points, only no longer axis-aligned.  Everything that paints,
    # hit-tests or moves a shape already works corner by corner and therefore
    # needs no change; only the operations below and Canvas.bounded_move_vertex
    # care about the difference.

    def center(self):
        """Centre of the shape, as a QPointF."""
        if not self.points:
            return QPointF()
        x = sum(p.x() for p in self.points) / len(self.points)
        y = sum(p.y() for p in self.points) / len(self.points)
        return QPointF(x, y)

    def obb(self):
        """``(cx, cy, w, h, theta)`` for this shape, or ``None`` if not a box."""
        if len(self.points) != 4:
            return None
        return corners_to_obb([(p.x(), p.y()) for p in self.points])

    def is_rotated(self):
        """True when this shape is a rectangle that is not axis-aligned."""
        if len(self.points) != 4:
            return False
        return _is_rotated([(p.x(), p.y()) for p in self.points])

    def rotate(self, theta, center=None):
        """Turn the shape by *theta* radians about *center* (default: its own).

        Rotating about the centre rather than a corner is what makes the key
        held down to spin a box feel like turning the animal in place: the box
        stays on the animal instead of swinging away from it.
        """
        if len(self.points) < 2:
            return
        pivot = self.center() if center is None else center
        rotated = rotate_points(
            [(p.x(), p.y()) for p in self.points], theta, (pivot.x(), pivot.y())
        )
        self.points = [QPointF(x, y) for x, y in rotated]

    def set_obb(self, cx, cy, w, h, theta):
        """Replace the four corners with those of this oriented box."""
        self.points = [QPointF(x, y) for x, y in obb_corners((cx, cy, w, h, theta))]
        self._closed = True

    def highlight_vertex(self, i, action):
        self._highlight_index = i
        self._highlight_mode = action

    def highlight_clear(self):
        self._highlight_index = None

    def copy(self):
        shape = Shape("%s" % self.label)
        shape.points = [p for p in self.points]
        shape.fill = self.fill
        shape.selected = self.selected
        shape._closed = self._closed
        if self.line_color != Shape.line_color:
            shape.line_color = self.line_color
        if self.fill_color != Shape.fill_color:
            shape.fill_color = self.fill_color
        shape.difficult = self.difficult
        return shape

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value
