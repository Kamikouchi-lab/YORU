"""Tests for ``yoru.libs.click_segment`` — the Click to Box engine.

The engine is called straight from a Qt mouse handler, so two properties matter
as much as the fit itself: it never raises, and a failure comes back as a named
diagnosis with a sentence to show rather than as ``None``.  Both are asserted
here alongside the accuracy of the fit on synthetic animals whose true size and
angle are known.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from yoru.libs.click_segment import (
    DEFAULT_RADIUS,
    DIAGNOSES,
    MERGED_BACKGROUND,
    NO_FRAME,
    NOTHING_FOUND,
    OUT_OF_FRAME,
    TWO_ANIMALS,
    ClickRequest,
    ClickResult,
    mask_to_obb,
    segment,
    segment_local,
)

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")
np = pytest.importorskip("numpy")

MODULE = Path(__file__).resolve().parents[1] / "yoru" / "libs" / "click_segment.py"


# ---------------------------------------------------------------------------
# Synthetic animals
# ---------------------------------------------------------------------------

def make_fly(degrees=30.0, length=60, breadth=16, centre=(150, 150),
             size=(300, 300), legs=True, background=220, body=30):
    """A dark elongated body with thin appendages, on a light plate.

    The appendages are the point: a box fitted to the raw silhouette would
    swallow them, so a fit that comes back at the body's own size is evidence
    that the distance-transform step is doing its job.
    """
    img = np.full((size[1], size[0], 3), background, np.uint8)
    cv2.ellipse(img, centre, (length // 2, breadth // 2), degrees, 0, 360,
                (body, body, body), -1)
    if legs:
        a = math.radians(degrees)
        for k in (-2, -1, 0, 1, 2):
            for s in (-1, 1):
                x0 = int(centre[0] + k * 9 * math.cos(a))
                y0 = int(centre[1] + k * 9 * math.sin(a))
                x1 = int(x0 - s * 17 * math.sin(a))
                y1 = int(y0 + s * 17 * math.cos(a))
                cv2.line(img, (x0, y0), (x1, y1), (body, body, body), 1)
    return img


def make_larva_on_dark(centre=(150, 150)):
    """A pale body on a dark plate — the opposite polarity."""
    img = np.full((300, 300, 3), 25, np.uint8)
    cv2.ellipse(img, centre, (28, 9), 0, 0, 360, (230, 230, 230), -1)
    return img


# ---------------------------------------------------------------------------
# Module invariants
# ---------------------------------------------------------------------------

class TestModuleInvariants:
    """What the module must never import, asserted on its source."""

    @pytest.fixture(scope="class")
    def tree(self):
        return ast.parse(MODULE.read_text(encoding="utf-8"))

    def _imported_names(self, tree):
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        return names

    @pytest.mark.parametrize("banned", ["torch", "PyQt5", "PyQt4", "dearpygui"])
    def test_the_engine_pulls_in_no_gui_and_no_torch(self, tree, banned):
        assert banned not in self._imported_names(tree)

    @pytest.mark.parametrize("lazy", ["cv2", "numpy"])
    def test_opencv_and_numpy_are_imported_lazily(self, tree, lazy):
        """The module must import on a machine with neither installed."""
        top_level = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module.split(".")[0])
        assert lazy not in top_level

    def test_it_spawns_nothing(self, tree):
        assert "subprocess" not in self._imported_names(tree)

    def test_every_diagnosis_names_a_cause_and_a_next_step(self):
        for key, sentence in DIAGNOSES.items():
            assert sentence.endswith("."), key
            assert len(sentence.split()) >= 4, key


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

class TestSegmentLocal:

    def test_a_click_on_a_fly_returns_a_box(self):
        result = segment_local(make_fly(), ClickRequest(x=150, y=150))
        assert result.ok
        assert result.diagnosis == ""
        assert result.backend == "local"

    @pytest.mark.parametrize("degrees", [0, 15, 30, 45, 60, 80, -25, -55])
    def test_the_angle_is_recovered_to_within_a_few_degrees(self, degrees):
        img = make_fly(degrees=degrees)
        result = segment_local(img, ClickRequest(x=150, y=150))
        assert result.ok
        got = math.degrees(result.obb[4])
        # Compare mod 180: a rectangle has no head and no tail.
        error = abs((got - degrees + 90) % 180 - 90)
        assert error < 5.0, f"{degrees} deg -> {got:.1f} deg"

    def test_the_legs_do_not_inflate_the_box(self):
        """The whole reason mask_to_obb exists rather than cv2.minAreaRect."""
        with_legs = segment_local(make_fly(legs=True), ClickRequest(x=150, y=150))
        without = segment_local(make_fly(legs=False), ClickRequest(x=150, y=150))
        assert with_legs.ok and without.ok
        # The legs stick out 17 px either side; the box must not grow by that.
        assert with_legs.obb[3] < without.obb[3] + 4.0

    def test_the_size_is_close_to_the_true_body(self):
        result = segment_local(
            make_fly(degrees=30, length=60, breadth=16), ClickRequest(x=150, y=150)
        )
        assert result.ok
        _cx, _cy, w, h, _theta = result.obb
        assert 55 <= w <= 68, w
        assert 12 <= h <= 22, h

    def test_the_centre_is_near_the_animal(self):
        result = segment_local(make_fly(centre=(120, 180)), ClickRequest(x=120, y=180))
        assert result.ok
        assert abs(result.obb[0] - 120) < 6
        assert abs(result.obb[1] - 180) < 6

    def test_a_pale_animal_on_a_dark_plate_works_too(self):
        """Polarity is decided from the pixels under the cursor, not configured."""
        result = segment_local(make_larva_on_dark(), ClickRequest(x=150, y=150))
        assert result.ok, result.detail
        assert result.obb[2] > result.obb[3]

    def test_a_click_a_couple_of_pixels_off_the_edge_is_rescued(self):
        img = make_fly(degrees=0, length=60, breadth=16)
        # Just outside the ellipse's short axis, still clearly on this animal.
        result = segment_local(img, ClickRequest(x=150, y=159))
        assert result.ok, result.detail

    def test_the_bbox_encloses_the_obb(self):
        result = segment_local(make_fly(degrees=40), ClickRequest(x=150, y=150))
        assert result.ok
        from yoru.libs.obb import obb_corners
        x1, y1, x2, y2 = result.bbox
        for x, y in obb_corners(result.obb):
            assert x1 - 1 <= x <= x2 + 1
            assert y1 - 1 <= y <= y2 + 1

    def test_the_result_is_deterministic(self):
        img = make_fly(degrees=22)
        first = segment_local(img, ClickRequest(x=150, y=150))
        second = segment_local(img, ClickRequest(x=150, y=150))
        assert first.obb == second.obb

    def test_a_radius_hint_does_not_change_a_successful_fit(self):
        img = make_fly(degrees=22)
        default = segment_local(img, ClickRequest(x=150, y=150))
        hinted = segment_local(
            img, ClickRequest(x=150, y=150, radius_hint=DEFAULT_RADIUS)
        )
        assert default.obb == pytest.approx(hinted.obb, abs=1e-9)

    def test_an_explicit_roi_is_honoured(self):
        img = make_fly(degrees=0)
        result = segment_local(
            img, ClickRequest(x=150, y=150, roi=(100, 120, 200, 180))
        )
        assert result.ok, result.detail


# ---------------------------------------------------------------------------
# Failures are results, not exceptions
# ---------------------------------------------------------------------------

class TestFailuresAreResults:

    def test_no_frame(self):
        result = segment_local(None, ClickRequest(x=1, y=1))
        assert isinstance(result, ClickResult)
        assert not result.ok
        assert result.diagnosis == NO_FRAME
        assert result.detail == DIAGNOSES[NO_FRAME]

    def test_a_greyscale_array_is_refused_as_no_frame(self):
        result = segment_local(np.zeros((10, 10), np.uint8), ClickRequest(x=1, y=1))
        assert result.diagnosis == NO_FRAME

    @pytest.mark.parametrize("point", [(-1, 10), (10, -1), (400, 10), (10, 400)])
    def test_a_click_outside_the_image(self, point):
        result = segment_local(make_fly(), ClickRequest(x=point[0], y=point[1]))
        assert result.diagnosis == OUT_OF_FRAME

    def test_a_click_on_empty_background(self):
        result = segment_local(make_fly(), ClickRequest(x=20, y=20))
        assert not result.ok
        assert result.diagnosis in (NOTHING_FOUND, MERGED_BACKGROUND)
        assert result.detail

    def test_a_uniform_image_yields_a_named_failure_not_a_crash(self):
        flat = np.full((200, 200, 3), 128, np.uint8)
        result = segment_local(flat, ClickRequest(x=100, y=100))
        assert not result.ok
        assert result.diagnosis in DIAGNOSES

    def test_two_touching_animals_are_reported_as_such(self):
        """Two bodies of comparable size joined at a thin point are two animals.

        The ratio between the two body cores decides this, not their number: an
        animal with one small satellite blob is still one animal, so a count
        would misfire on every fly whose wing thresholded separately.
        """
        img = np.full((300, 300, 3), 220, np.uint8)
        cv2.circle(img, (122, 150), 14, (30, 30, 30), -1)
        cv2.circle(img, (178, 150), 14, (30, 30, 30), -1)
        cv2.line(img, (122, 150), (178, 150), (30, 30, 30), 5)
        result = segment_local(img, ClickRequest(x=150, y=150))
        assert not result.ok
        assert result.diagnosis == TWO_ANIMALS
        assert "Click nearer" in result.detail

    def test_one_animal_with_a_small_satellite_is_still_one_animal(self):
        """The counterpart: a detached speck must not read as a second animal."""
        img = make_fly(degrees=0, legs=False)
        cv2.circle(img, (150, 168), 2, (30, 30, 30), -1)
        result = segment_local(img, ClickRequest(x=150, y=150))
        assert result.ok, result.detail

    def test_a_failed_result_carries_no_geometry(self):
        result = segment_local(None, ClickRequest(x=1, y=1))
        assert result.obb is None
        assert result.bbox is None


# ---------------------------------------------------------------------------
# mask_to_obb on its own
# ---------------------------------------------------------------------------

class TestMaskToObb:

    def test_a_plain_rectangle_mask(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[40:60, 20:80] = 1
        obb, why = mask_to_obb(mask, (50, 50))
        assert why == ""
        cx, cy, w, h, theta = obb
        assert cx == pytest.approx(49.5, abs=1.0)
        assert cy == pytest.approx(49.5, abs=1.0)
        assert w == pytest.approx(60, abs=2.0)
        assert h == pytest.approx(20, abs=2.0)
        assert abs(math.degrees(theta)) < 2.0

    def test_w_is_the_body_axis_even_for_a_tall_blob(self):
        mask = np.zeros((120, 120), np.uint8)
        mask[20:100, 50:70] = 1
        obb, why = mask_to_obb(mask, (60, 60))
        assert why == ""
        # The long axis is vertical, so w (the body axis) is the long one and
        # theta is near +/- 90 degrees.
        assert obb[2] > obb[3]
        assert abs(abs(math.degrees(obb[4])) - 90) < 2.0

    def test_an_empty_mask_fails_cleanly(self):
        obb, why = mask_to_obb(np.zeros((50, 50), np.uint8), (25, 25))
        assert obb is None
        assert why == NOTHING_FOUND

    def test_a_non_2d_mask_fails_cleanly(self):
        obb, why = mask_to_obb(np.zeros((10, 10, 3), np.uint8), (5, 5))
        assert obb is None
        assert why == NOTHING_FOUND

    def test_a_blob_filling_the_window_is_reported_as_merged_background(self):
        mask = np.ones((60, 60), np.uint8)
        obb, why = mask_to_obb(mask, (30, 30))
        assert obb is None
        # Touching every edge is detected first; either answer names a real
        # cause and offers the same next step.
        assert why in DIAGNOSES

    def test_a_speck_is_not_an_animal(self):
        mask = np.zeros((60, 60), np.uint8)
        mask[30:32, 30:32] = 1
        obb, why = mask_to_obb(mask, (30, 30))
        assert obb is None
        assert why == NOTHING_FOUND


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestSegmentDispatch:

    def test_the_default_backend_is_local(self):
        img = make_fly()
        assert segment(img, ClickRequest(x=150, y=150)).backend == "local"

    def test_an_unknown_backend_falls_back_to_local_rather_than_failing(self):
        """The tool button is never disabled, so dispatch must never refuse."""
        img = make_fly()
        result = segment(img, ClickRequest(x=150, y=150), backend="sam")
        assert result.ok
        assert result.backend == "local"
