"""Tests for the oriented-box path through detection, drawing and training.

The pipeline carries a box in two parameterisations at once: ``x1..y2``, the
upright box, which everything written before OBB support reads; and
``cx, cy, w, h, angle``, the box itself.  These tests pin the contract between
them — the row layout, the derivation for backends that predict no rotation,
and the fact that the first eight columns never moved.
"""

from __future__ import annotations

import math

import pytest

from yoru.libs.detector_base import DETECTION_COLUMNS, detection_row, obb_of
from yoru.libs.init_train import OBB_CAPABLE_FAMILIES

np = pytest.importorskip("numpy")


UPRIGHT = {
    "x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 60.0,
    "conf": 0.9, "class_id": 1, "class_name": "fly",
}

ROTATED = {
    "x1": 69.0, "y1": 26.0, "x2": 131.0, "y2": 74.0,
    "cx": 100.0, "cy": 50.0, "w": 60.0, "h": 20.0, "angle": math.radians(30),
    "conf": 0.8, "class_id": 0, "class_name": "larva",
}


# ---------------------------------------------------------------------------
# The row layout
# ---------------------------------------------------------------------------

class TestDetectionColumns:

    def test_the_original_eight_columns_kept_their_positions(self):
        """Trigger plugins index a row by position; these must not move."""
        assert DETECTION_COLUMNS[:8] == (
            "x1", "y1", "x2", "y2",
            "confidence", "class", "class_name", "total_time",
        )

    def test_the_oriented_columns_are_appended_at_the_end(self):
        assert DETECTION_COLUMNS[8:] == ("cx", "cy", "w", "h", "angle")

    def test_a_row_has_one_value_per_column(self):
        assert len(detection_row(UPRIGHT, 1.25)) == len(DETECTION_COLUMNS)

    def test_the_confidence_and_class_name_are_where_the_trigger_looks(self):
        row = detection_row(UPRIGHT, 1.25)
        assert row[4] == 0.9
        assert row[6] == "fly"

    def test_the_total_time_is_carried_through(self):
        assert detection_row(UPRIGHT, 3.5)[7] == 3.5


# ---------------------------------------------------------------------------
# Deriving the oriented box
# ---------------------------------------------------------------------------

class TestObbOf:

    def test_a_backend_without_rotation_gets_the_upright_box(self):
        assert obb_of(UPRIGHT) == pytest.approx((20.0, 40.0, 20.0, 40.0, 0.0), abs=1e-9)

    def test_a_backend_with_rotation_is_taken_at_its_word(self):
        assert obb_of(ROTATED) == pytest.approx(
            (100.0, 50.0, 60.0, 20.0, math.radians(30)), abs=1e-9
        )

    def test_an_angle_of_exactly_zero_still_counts_as_supplied(self):
        """An OBB model may well predict a box that happens to be upright."""
        d = dict(ROTATED, angle=0.0, w=61.0, h=17.0)
        assert obb_of(d)[2:] == pytest.approx((61.0, 17.0, 0.0), abs=1e-9)

    def test_the_derived_centre_matches_the_upright_box(self):
        cx, cy, _w, _h, _a = obb_of(UPRIGHT)
        assert cx == (UPRIGHT["x1"] + UPRIGHT["x2"]) / 2
        assert cy == (UPRIGHT["y1"] + UPRIGHT["y2"]) / 2


# ---------------------------------------------------------------------------
# The ultralytics detector's OBB branch
# ---------------------------------------------------------------------------

class FakeTensor:
    """Just enough of a torch tensor for the detector's ``.cpu()[i, j]`` use."""

    def __init__(self, array):
        self._a = np.asarray(array, dtype=float)

    def cpu(self):
        return self._a

    def __len__(self):
        return len(self._a)


class FakeOBB:
    def __init__(self, xywhr, xyxy, conf, cls):
        self.xywhr = FakeTensor(xywhr)
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)

    def __len__(self):
        return len(self.xywhr)


class TestUltralyticsObbBranch:

    @pytest.fixture
    def detector(self):
        from yoru.libs.plugins.ultralytics_detector import UltralyticsDetector

        det = UltralyticsDetector()
        det._names = {0: "larva", 1: "fly"}
        return det

    def test_oriented_predictions_carry_both_parameterisations(self, detector):
        obb = FakeOBB(
            xywhr=[[100.0, 50.0, 60.0, 20.0, math.radians(30)]],
            xyxy=[[69.0, 26.0, 131.0, 74.0]],
            conf=[0.77],
            cls=[1],
        )
        out = detector._obb_detections(obb)
        assert len(out) == 1
        d = out[0]
        assert (d["x1"], d["y1"], d["x2"], d["y2"]) == (69.0, 26.0, 131.0, 74.0)
        assert (d["cx"], d["cy"], d["w"], d["h"]) == (100.0, 50.0, 60.0, 20.0)
        assert d["angle"] == pytest.approx(math.radians(30))
        assert d["conf"] == pytest.approx(0.77)
        assert d["class_id"] == 1
        assert d["class_name"] == "fly"

    def test_every_prediction_is_returned(self, detector):
        obb = FakeOBB(
            xywhr=[[10, 10, 4, 2, 0.1], [20, 20, 6, 3, -0.2], [30, 30, 8, 4, 0.3]],
            xyxy=[[8, 9, 12, 11], [17, 18, 23, 22], [26, 28, 34, 32]],
            conf=[0.9, 0.8, 0.7],
            cls=[0, 1, 0],
        )
        assert len(detector._obb_detections(obb)) == 3

    def test_an_unknown_class_id_falls_back_to_its_number(self, detector):
        obb = FakeOBB([[10, 10, 4, 2, 0.0]], [[8, 9, 12, 11]], [0.5], [9])
        assert detector._obb_detections(obb)[0]["class_name"] == "9"

    def test_an_oriented_detection_survives_the_row_conversion(self, detector):
        obb = FakeOBB(
            [[100.0, 50.0, 60.0, 20.0, math.radians(30)]],
            [[69.0, 26.0, 131.0, 74.0]], [0.77], [1],
        )
        row = detection_row(detector._obb_detections(obb)[0], 2.0)
        assert row[DETECTION_COLUMNS.index("angle")] == pytest.approx(
            math.radians(30)
        )
        assert row[DETECTION_COLUMNS.index("w")] == 60.0


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

class TestDrawing:

    @pytest.fixture(autouse=True)
    def _cv(self):
        pytest.importorskip("cv2")
        pytest.importorskip("matplotlib")

    def test_an_upright_box_draws_its_four_edges(self):
        from yoru.libs.drawing import draw_box

        img = np.zeros((200, 200, 3), np.uint8)
        draw_box(img, (100, 100, 60, 20, 0.0), (0, 255, 0), thickness=1)
        # The top and bottom edges are drawn, the interior is not filled.
        assert img[90, 100].any()
        assert img[110, 100].any()
        assert not img[100, 100].any()

    def test_a_rotated_box_puts_ink_off_the_upright_edges(self):
        from yoru.libs.drawing import draw_box

        upright = np.zeros((200, 200, 3), np.uint8)
        tilted = np.zeros((200, 200, 3), np.uint8)
        draw_box(upright, (100, 100, 60, 20, 0.0), (0, 255, 0), thickness=1)
        draw_box(tilted, (100, 100, 60, 20, math.radians(45)), (0, 255, 0), thickness=1)
        assert not np.array_equal(upright, tilted)
        # The tilted box reaches further up than the upright one's top edge.
        assert tilted[:90].any()
        assert not upright[:89].any()

    def test_draw_detections_reads_the_schema_columns(self):
        from yoru.libs.drawing import draw_detections

        img = np.zeros((200, 200, 3), np.uint8)
        rows = [detection_row(ROTATED, 0.0)]
        draw_detections(img, rows, {0: (255, 0, 0)})
        assert img.any()

    def test_drawing_nothing_leaves_the_frame_untouched(self):
        from yoru.libs.drawing import draw_detections

        img = np.zeros((50, 50, 3), np.uint8)
        draw_detections(img, [], {})
        assert not img.any()


# ---------------------------------------------------------------------------
# Weight names for an OBB project
# ---------------------------------------------------------------------------

class TestObbWeightNames:

    def build(self, **overrides):
        """``_build_weight`` on a bare instance — no GUI, no DearPyGui context."""
        from yoru.train_GUI import yoru_train

        gui = yoru_train.__new__(yoru_train)
        gui.m_dict = dict(
            {"model_family": "YOLO", "yolo_version": "YOLO11", "yolo_size": "s"},
            **overrides,
        )
        return gui._build_weight()

    def test_a_detection_project_keeps_the_plain_weight(self):
        assert self.build(obb=False) == "yolo11s.pt"
        assert self.build() == "yolo11s.pt"

    def test_an_obb_project_asks_for_the_obb_weight(self):
        assert self.build(obb=True) == "yolo11s-obb.pt"

    @pytest.mark.parametrize("version,size,expected", [
        ("YOLOv8", "n", "yolov8n-obb.pt"),
        ("YOLOv8", "x", "yolov8x-obb.pt"),
        ("YOLO11", "m", "yolo11m-obb.pt"),
    ])
    def test_every_yolo_version_and_size(self, version, size, expected):
        assert self.build(obb=True, yolo_version=version, yolo_size=size) == expected

    def test_every_obb_weight_it_can_build_is_offered_in_the_weight_list(self):
        from yoru.libs.init_train import init_train

        m = {}
        init_train(m)
        for version in ("YOLOv8", "YOLO11"):
            for size in m["yolo_size_list"]:
                weight = self.build(obb=True, yolo_version=version, yolo_size=size)
                assert weight in m["weight_list"], weight

    def test_only_yolo_is_offered_for_oriented_boxes(self):
        """RT-DETR and the torchvision detectors have no rotated-box head."""
        assert OBB_CAPABLE_FAMILIES == ("YOLO",)

    def test_a_non_yolo_family_ignores_the_flag_rather_than_inventing_a_weight(self):
        assert self.build(obb=True, model_family="RT-DETR", rtdetr_size="l") == "rtdetr-l.pt"
