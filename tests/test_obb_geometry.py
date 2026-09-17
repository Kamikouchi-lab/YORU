"""Tests for ``yoru.libs.obb`` — YORU's single definition of a rotated box.

Four parts of YORU reduce corners to ``(cx, cy, w, h, theta)`` and back: the
annotation canvas, the YOLO-OBB label files, the detector's results and the
real-time drawing.  If any two of them disagreed about the corner order or the
angle convention, a box would silently change shape between being drawn and
being trained on — so the round trips below are the load-bearing tests.
"""

from __future__ import annotations

import math

import pytest

from yoru.libs.obb import (
    ANGLE_SNAP,
    aabb_to_obb,
    box_axes,
    corners_to_obb,
    is_rotated,
    normalize_angle,
    obb_corners,
    obb_to_aabb,
    point_in_obb,
    resize_corner,
    rotate_points,
)


def approx(value):
    return pytest.approx(value, abs=1e-9)


# ---------------------------------------------------------------------------
# Angle convention
# ---------------------------------------------------------------------------

class TestNormalizeAngle:

    @pytest.mark.parametrize("degrees", [0, 1, 30, 44.9, -30, -89, 89])
    def test_angles_already_in_range_are_unchanged(self, degrees):
        theta = math.radians(degrees)
        assert normalize_angle(theta) == pytest.approx(theta, abs=1e-12)

    @pytest.mark.parametrize("degrees", [0, 17, 45, 80, -33, 123, -178])
    def test_period_is_pi_not_two_pi(self, degrees):
        """A rectangle maps onto itself under a pi rotation."""
        theta = math.radians(degrees)
        assert normalize_angle(theta) == approx(normalize_angle(theta + math.pi))

    def test_result_is_always_in_the_half_open_range(self):
        for degrees in range(-720, 721, 7):
            out = normalize_angle(math.radians(degrees))
            assert -math.pi / 2 <= out < math.pi / 2

    def test_plus_half_pi_folds_to_minus_half_pi(self):
        """One rectangle must never have two canonical forms."""
        assert normalize_angle(math.pi / 2) == approx(-math.pi / 2)
        assert normalize_angle(math.pi / 2 - 2 * ANGLE_SNAP) > 0


# ---------------------------------------------------------------------------
# corners <-> (cx, cy, w, h, theta)
# ---------------------------------------------------------------------------

class TestCornerRoundTrip:

    @pytest.mark.parametrize("degrees", [0, 1, 17, 30, 45, 70, -12, -60, -89])
    def test_corners_then_back_is_the_identity(self, degrees):
        box = (120.0, 80.0, 61.0, 18.0, math.radians(degrees))
        cx, cy, w, h, theta = corners_to_obb(obb_corners(box))
        assert (cx, cy, w, h) == pytest.approx(box[:4], abs=1e-9)
        assert theta == approx(normalize_angle(box[4]))

    def test_w_stays_the_first_edge_even_when_it_is_the_shorter_one(self):
        """``w`` is the body axis by convention, not "the long side"."""
        tall = (0.0, 0.0, 10.0, 90.0, math.radians(20))
        _cx, _cy, w, h, _theta = corners_to_obb(obb_corners(tall))
        assert w == approx(10.0)
        assert h == approx(90.0)

    def test_corner_order_matches_an_upright_box_drawn_by_hand(self):
        """labelImg builds its four points in this order; so must we."""
        corners = obb_corners((50.0, 40.0, 20.0, 10.0, 0.0))
        assert corners == pytest.approx(
            [(40.0, 35.0), (60.0, 35.0), (60.0, 45.0), (40.0, 45.0)], abs=1e-9
        )

    def test_four_corners_are_required(self):
        with pytest.raises(ValueError):
            corners_to_obb([(0, 0), (1, 0), (1, 1)])

    def test_a_degenerate_width_still_yields_a_usable_angle(self):
        """A box squashed flat mid-drag must not lose its orientation."""
        theta = math.radians(35)
        flat = obb_corners((10.0, 10.0, 0.0, 40.0, theta))
        _cx, _cy, _w, h, out = corners_to_obb(flat)
        assert h == approx(40.0)
        assert out == approx(normalize_angle(theta))


# ---------------------------------------------------------------------------
# The upright box around a rotated one
# ---------------------------------------------------------------------------

class TestAabb:

    def test_upright_box_is_its_own_bounding_box(self):
        assert obb_to_aabb((50.0, 40.0, 20.0, 10.0, 0.0)) == pytest.approx(
            (40.0, 35.0, 60.0, 45.0), abs=1e-9
        )

    def test_rotating_a_box_grows_its_bounding_box(self):
        upright = obb_to_aabb((0.0, 0.0, 60.0, 20.0, 0.0))
        tilted = obb_to_aabb((0.0, 0.0, 60.0, 20.0, math.radians(30)))
        assert (tilted[2] - tilted[0]) > (upright[2] - upright[0]) - 1e-9
        assert (tilted[3] - tilted[1]) > (upright[3] - upright[1])

    def test_a_45_degree_square_has_a_diagonal_bounding_box(self):
        x1, y1, x2, y2 = obb_to_aabb((0.0, 0.0, 10.0, 10.0, math.radians(45)))
        side = 10.0 * math.sqrt(2)
        assert (x2 - x1) == pytest.approx(side, abs=1e-9)
        assert (y2 - y1) == pytest.approx(side, abs=1e-9)

    def test_clamping_keeps_the_box_inside_the_frame(self):
        x1, y1, x2, y2 = obb_to_aabb((5.0, 5.0, 40.0, 40.0, 0.0), width=100, height=100)
        assert (x1, y1) == (0.0, 0.0)
        assert x2 <= 100 and y2 <= 100

    def test_aabb_to_obb_is_the_inverse_for_upright_boxes(self):
        box = aabb_to_obb(10, 20, 30, 60)
        assert box == pytest.approx((20.0, 40.0, 20.0, 40.0, 0.0), abs=1e-9)
        assert obb_to_aabb(box) == pytest.approx((10.0, 20.0, 30.0, 60.0), abs=1e-9)


# ---------------------------------------------------------------------------
# Hit testing and rotation
# ---------------------------------------------------------------------------

class TestPointInObb:

    def test_the_centre_is_always_inside(self):
        assert point_in_obb(100, 50, (100, 50, 60, 20, math.radians(30)))

    def test_a_corner_of_the_bounding_box_is_outside_a_tilted_box(self):
        box = (0.0, 0.0, 60.0, 20.0, math.radians(45))
        x1, y1, _x2, _y2 = obb_to_aabb(box)
        assert not point_in_obb(x1 + 0.1, y1 + 0.1, box)

    def test_a_point_along_the_body_axis_is_inside(self):
        theta = math.radians(30)
        box = (0.0, 0.0, 60.0, 20.0, theta)
        assert point_in_obb(25 * math.cos(theta), 25 * math.sin(theta), box)
        assert not point_in_obb(40 * math.cos(theta), 40 * math.sin(theta), box)


class TestRotatePoints:

    def test_rotating_by_a_full_turn_returns_the_same_points(self):
        pts = [(0, 0), (10, 0), (10, 5), (0, 5)]
        out = rotate_points(pts, 2 * math.pi)
        # Flattened: pytest.approx compares a list of tuples element-wise only
        # for flat sequences.
        flat_out = [v for p in out for v in p]
        flat_in = [float(v) for p in pts for v in p]
        assert flat_out == pytest.approx(flat_in, abs=1e-9)

    def test_rotation_preserves_the_size_and_turns_the_angle(self):
        box = (10.0, 10.0, 40.0, 12.0, 0.0)
        turned = corners_to_obb(rotate_points(obb_corners(box), math.radians(25)))
        assert turned[2] == approx(40.0)
        assert turned[3] == approx(12.0)
        assert turned[4] == approx(math.radians(25))

    def test_rotation_is_about_the_centre_by_default(self):
        box = (10.0, 10.0, 40.0, 12.0, 0.0)
        turned = corners_to_obb(rotate_points(obb_corners(box), math.radians(25)))
        assert turned[:2] == pytest.approx((10.0, 10.0), abs=1e-9)


class TestIsRotated:

    def test_an_upright_box_is_not_rotated(self):
        assert not is_rotated(obb_corners((0, 0, 40, 20, 0.0)))

    @pytest.mark.parametrize("degrees", [0.5, 5, 30, -20])
    def test_a_tilted_box_is_rotated(self, degrees):
        assert is_rotated(obb_corners((0, 0, 40, 20, math.radians(degrees))))

    def test_the_test_is_scale_free(self):
        """A huge box tilted by a rounding error is not 'rotated'."""
        assert not is_rotated(obb_corners((0, 0, 2000, 900, 1e-9)))

    def test_anything_that_is_not_four_points_is_not_a_rotated_box(self):
        assert not is_rotated([(0, 0), (1, 1)])


# ---------------------------------------------------------------------------
# Dragging a corner — the part the annotator's hand touches
# ---------------------------------------------------------------------------

UPRIGHT = [(10.0, 20.0), (50.0, 20.0), (50.0, 60.0), (10.0, 60.0)]
TILTED = obb_corners((100.0, 100.0, 60.0, 20.0, math.radians(30)))


class TestResizeCorner:

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_the_dragged_corner_lands_exactly_on_the_cursor(self, index):
        out = resize_corner(TILTED, index, (137.0, 91.0))
        assert out[index] == pytest.approx((137.0, 91.0), abs=1e-9)

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_the_opposite_corner_never_moves(self, index):
        out = resize_corner(TILTED, index, (137.0, 91.0))
        anchor = (index + 2) % 4
        assert out[anchor] == pytest.approx(TILTED[anchor], abs=1e-9)

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_the_result_is_still_a_rectangle(self, index):
        out = resize_corner(TILTED, index, (137.0, 91.0))
        # Adjacent edges perpendicular, opposite edges equal: a rectangle.
        e01 = (out[1][0] - out[0][0], out[1][1] - out[0][1])
        e12 = (out[2][0] - out[1][0], out[2][1] - out[1][1])
        e32 = (out[2][0] - out[3][0], out[2][1] - out[3][1])
        assert e01[0] * e12[0] + e01[1] * e12[1] == pytest.approx(0.0, abs=1e-7)
        assert e01 == pytest.approx(e32, abs=1e-7)

    @pytest.mark.parametrize("index", [0, 1, 2, 3])
    def test_dragging_does_not_turn_the_box(self, index):
        before = corners_to_obb(TILTED)[4]
        after = corners_to_obb(resize_corner(TILTED, index, (137.0, 91.0)))[4]
        assert after == pytest.approx(before, abs=1e-7)

    def test_an_upright_box_behaves_as_it_always_did(self):
        """The rotated-frame maths must collapse to the old axis-aligned case."""
        out = resize_corner(UPRIGHT, 0, (0.0, 0.0))
        assert out == pytest.approx(
            [(0.0, 0.0), (50.0, 0.0), (50.0, 60.0), (0.0, 60.0)], abs=1e-9
        )

    def test_square_mode_equalises_the_two_extents(self):
        out = resize_corner(TILTED, 0, (137.0, 91.0), square=True)
        _cx, _cy, w, h, _theta = corners_to_obb(out)
        assert w == pytest.approx(h, abs=1e-7)

    def test_the_input_is_not_modified(self):
        original = list(TILTED)
        resize_corner(TILTED, 2, (10.0, 10.0))
        assert TILTED == original

    def test_four_corners_are_required(self):
        with pytest.raises(ValueError):
            resize_corner([(0, 0), (1, 0), (1, 1)], 0, (2, 2))


class TestBoxAxes:

    def test_an_upright_box_has_the_image_axes(self):
        u, v = box_axes(UPRIGHT)
        assert u == pytest.approx((1.0, 0.0), abs=1e-9)
        assert v == pytest.approx((0.0, 1.0), abs=1e-9)

    def test_the_axes_are_orthonormal_for_a_tilted_box(self):
        u, v = box_axes(TILTED)
        assert math.hypot(*u) == approx(1.0)
        assert math.hypot(*v) == approx(1.0)
        assert u[0] * v[0] + u[1] * v[1] == pytest.approx(0.0, abs=1e-9)

    def test_a_degenerate_box_still_yields_axes(self):
        u, v = box_axes([(5, 5), (5, 5), (5, 5), (5, 5)])
        assert u == (1.0, 0.0)
        assert v == (0.0, 1.0)
