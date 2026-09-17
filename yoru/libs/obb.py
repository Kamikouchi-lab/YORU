# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Oriented bounding box (OBB) geometry, shared by every part of YORU.

An OBB is carried as ``(cx, cy, w, h, theta)`` -- centre, extent along the
box's own two axes, and the rotation of the ``w`` axis in **radians**,
measured from the image's +x axis and turning towards +y (i.e. clockwise on
screen, because image y points down).  The same convention ultralytics uses
for ``Results.obb.xywhr``, so an OBB coming back from a model needs no
conversion.

The one alternative representation is the **four corners**, in the order
``(-w/2,-h/2), (+w/2,-h/2), (+w/2,+h/2), (-w/2,+h/2)`` rotated by ``theta``.
That order is also labelImg's ``Shape.points`` order and the order the
YOLO-OBB (DOTA-style) label file stores, which is why the annotation tool can
keep working in corners while the detector works in ``xywhr``.

This module is deliberately dependency-free (``math`` only): labelImg imports
it inside a Qt process that has no torch, and the tests import it with no
OpenCV installed.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

__all__ = [
    "normalize_angle",
    "obb_corners",
    "corners_to_obb",
    "obb_to_aabb",
    "aabb_to_obb",
    "point_in_obb",
    "rotate_points",
    "is_rotated",
    "box_axes",
    "resize_corner",
    "ANGLE_SNAP",
]

_HALF_PI = math.pi / 2.0

#: Angles within this of +pi/2 are folded to -pi/2 instead, so that one
#: rectangle never has two canonical forms.  The float64 modulo in
#: :func:`normalize_angle` can round up to exactly pi, which would otherwise
#: pin the result at +pi/2.
ANGLE_SNAP = 1e-6

#: Below this many radians a box counts as axis-aligned.  Used only to decide
#: how to *draw* and *store* a box, never to change its geometry.
_UPRIGHT_EPS = 1e-4


def normalize_angle(theta: float) -> float:
    """Fold *theta* into ``[-pi/2, pi/2)``.

    A rectangle has period pi, so this never changes the box: a pi rotation
    maps each corner onto the opposite one and leaves ``w`` and ``h`` assigned
    to the same sides.  Folding mod *pi/2* would not be safe -- that swaps
    ``w`` and ``h`` -- which is why the period here is pi.
    """
    out = (float(theta) + _HALF_PI) % math.pi - _HALF_PI
    if out >= _HALF_PI - ANGLE_SNAP:
        out -= math.pi
    return out


def obb_corners(obb: Sequence[float]) -> list:
    """The four corners of ``(cx, cy, w, h, theta)``, in canonical order."""
    cx, cy, w, h, theta = (float(v) for v in obb)
    hw, hh = w / 2.0, h / 2.0
    c, s = math.cos(theta), math.sin(theta)
    return [
        (cx + lx * c - ly * s, cy + lx * s + ly * c)
        for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
    ]


def corners_to_obb(points: Sequence[Sequence[float]]) -> Tuple[float, float, float, float, float]:
    """Inverse of :func:`obb_corners`.

    The centre is the mean of the four corners rather than the midpoint of one
    diagonal: for a quadrilateral that a rounding error has left very slightly
    non-rectangular, the mean is the least-squares centre, and for an exact
    rectangle the two agree.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) != 4:
        raise ValueError(f"an OBB needs exactly 4 corners, got {len(pts)}")
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    ux, uy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    vx, vy = pts[3][0] - pts[0][0], pts[3][1] - pts[0][1]
    w = math.hypot(ux, uy)
    h = math.hypot(vx, vy)
    # A degenerate w leaves the axis undefined; take it from the other edge,
    # which is perpendicular to it by construction.
    if w > 1e-9:
        theta = math.atan2(uy, ux)
    elif h > 1e-9:
        theta = math.atan2(vy, vx) - _HALF_PI
    else:
        theta = 0.0
    return cx, cy, w, h, normalize_angle(theta)


def obb_to_aabb(
    obb: Sequence[float],
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    """Axis-aligned ``(x1, y1, x2, y2)`` enclosing *obb*, optionally clamped.

    Every consumer that predates OBB support -- the CSV writer, the trigger
    plugins, the evaluation IoU -- keeps reading these four numbers, so an OBB
    detection stays usable by them without any change on their side.
    """
    pts = obb_corners(obb)
    x1 = min(p[0] for p in pts)
    y1 = min(p[1] for p in pts)
    x2 = max(p[0] for p in pts)
    y2 = max(p[1] for p in pts)
    if width is not None:
        x1 = max(0.0, min(x1, float(width)))
        x2 = max(0.0, min(x2, float(width)))
    if height is not None:
        y1 = max(0.0, min(y1, float(height)))
        y2 = max(0.0, min(y2, float(height)))
    return x1, y1, x2, y2


def aabb_to_obb(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float, float]:
    """The zero-rotation OBB covering an axis-aligned box."""
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, abs(x2 - x1), abs(y2 - y1), 0.0)


def point_in_obb(x: float, y: float, obb: Sequence[float]) -> bool:
    """Is ``(x, y)`` inside this rotated rectangle?"""
    cx, cy, w, h, theta = (float(v) for v in obb)
    c, s = math.cos(-theta), math.sin(-theta)
    dx, dy = x - cx, y - cy
    lx, ly = dx * c - dy * s, dx * s + dy * c
    return abs(lx) <= w / 2.0 and abs(ly) <= h / 2.0


def rotate_points(points: Sequence[Sequence[float]], theta: float,
                  centre: Optional[Sequence[float]] = None) -> list:
    """Rotate *points* by *theta* radians about *centre* (default: their mean)."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if centre is None:
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
    else:
        cx, cy = float(centre[0]), float(centre[1])
    c, s = math.cos(float(theta)), math.sin(float(theta))
    return [
        (cx + (p[0] - cx) * c - (p[1] - cy) * s,
         cy + (p[0] - cx) * s + (p[1] - cy) * c)
        for p in pts
    ]


def is_rotated(points: Sequence[Sequence[float]], eps: float = _UPRIGHT_EPS) -> bool:
    """True when these four corners are not an axis-aligned rectangle.

    Decided on the box's own scale, not on an absolute pixel count: a 2000 px
    box tilted by a hundredth of a degree still has one edge a fifth of a pixel
    out of true, and calling that "rotated" would be noise.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) != 4:
        return False
    ux, uy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    span = max(math.hypot(ux, uy), 1e-9)
    return min(abs(ux), abs(uy)) / span > eps


def box_axes(points):
    """Unit vectors along a box's own width and height, as ``((ux, uy), (vx, vy))``.

    For an upright box these are ``(1, 0)`` and ``(0, 1)``, and every
    calculation that uses them collapses to the axis-aligned one it replaced.
    Deriving them from the shape rather than from a mode flag is what keeps one
    code path for both kinds of box.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    ux, uy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    vx, vy = pts[3][0] - pts[0][0], pts[3][1] - pts[0][1]
    nu = math.hypot(ux, uy)
    nv = math.hypot(vx, vy)
    if nu > 1e-9:
        u = (ux / nu, uy / nu)
        # v perpendicular to u rather than straight off the edge, so a box
        # squashed to zero height mid-drag keeps its orientation instead of
        # snapping back upright.
        v = (-u[1], u[0]) if nv <= 1e-9 else (vx / nv, vy / nv)
    elif nv > 1e-9:
        v = (vx / nv, vy / nv)
        u = (v[1], -v[0])
    else:
        u, v = (1.0, 0.0), (0.0, 1.0)
    return u, v


def resize_corner(points, index, pos, square=False):
    """Drag corner *index* to *pos*, keeping the four points a rectangle.

    The opposite corner is the anchor and never moves; the dragged corner lands
    exactly on the cursor; the two neighbours take one coordinate from each.
    All of it happens in the box's own rotated frame, which is what makes this
    the same gesture for an upright box and a tilted one — and the reason a
    rotated box is editable at all.

    With *square*, the two extents are equalised, as labelImg's "draw squares"
    option does; in a rotated frame that still yields a square, tilted.

    Returns a new list of four ``(x, y)`` tuples; *points* is not modified.
    """
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) != 4:
        raise ValueError(f"an OBB needs exactly 4 corners, got {len(pts)}")

    anchor = pts[(index + 2) % 4]
    u, v = box_axes(pts)

    dx, dy = float(pos[0]) - anchor[0], float(pos[1]) - anchor[1]
    a = dx * u[0] + dy * u[1]   # extent along the box's width axis
    b = dx * v[0] + dy * v[1]   # extent along its height axis

    if square:
        side = min(abs(a), abs(b))
        a = side if a >= 0 else -side
        b = side if b >= 0 else -side

    def at(local_a, local_b):
        return (anchor[0] + local_a * u[0] + local_b * v[0],
                anchor[1] + local_a * u[1] + local_b * v[1])

    # Which neighbour keeps the width coordinate and which keeps the height
    # alternates around the rectangle; this is the rotated-frame form of the
    # ``index % 2`` test the axis-aligned code used.
    if index % 2 == 0:
        first, second = (0.0, b), (a, 0.0)
    else:
        first, second = (a, 0.0), (0.0, b)

    out = list(pts)
    out[index] = at(a, b)
    out[(index + 1) % 4] = at(*first)
    out[(index + 3) % 4] = at(*second)
    return out
