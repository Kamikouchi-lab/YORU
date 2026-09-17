"""Tests for labelImg's oriented-box wiring.

These need PyQt5 but no display: ``QImage``, ``QPointF`` and ``Shape`` all work
without a ``QApplication``, so the parts that decide what ends up in a label
file are testable while the widgets are not.

The property under test throughout is that **the project decides the format**.
An OBB project that silently saved upright boxes would throw every angle away,
and the user would only find out when training produced a detection model.
"""

from __future__ import annotations

import math
from enum import Enum

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")

from PyQt5.QtCore import QPointF  # noqa: E402
from PyQt5.QtGui import QImage  # noqa: E402

from yoru.labelimg.labelimg import build_arg_parser  # noqa: E402
from yoru.labelimg.libs.constants import (  # noqa: E402
    FORMAT_YOLO,
    FORMAT_YOLO_OBB,
    SETTING_OBB_MODE,
)
from yoru.labelimg.libs.labelFile import (  # noqa: E402
    LabelFile,
    LabelFileFormat,
    coerce_label_file_format,
)
from yoru.labelimg.libs.shape import Shape  # noqa: E402
from yoru.libs.obb import corners_to_obb, obb_corners  # noqa: E402


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

class TestCommandLine:

    def test_no_arguments_leaves_every_choice_to_the_remembered_settings(self):
        args = build_arg_parser().parse_args([])
        assert args.image_dir is None
        assert args.class_file is None
        assert args.save_dir is None
        # None, not False: "nothing was said", so the last-used mode stands.
        assert args.obb is None

    def test_the_training_gui_passes_all_four(self):
        args = build_arg_parser().parse_args(
            ["/p/all_label_images", "/p/all_label_images/classes.txt",
             "/p/all_label_images", "--obb"]
        )
        assert args.image_dir == "/p/all_label_images"
        assert args.class_file.endswith("classes.txt")
        assert args.save_dir == "/p/all_label_images"
        assert args.obb is True

    def test_no_obb_is_an_explicit_answer_not_a_missing_one(self):
        """A detection project must override a remembered OBB setting."""
        assert build_arg_parser().parse_args(["/p", "--no-obb"]).obb is False

    def test_the_two_flags_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            build_arg_parser().parse_args(["--obb", "--no-obb"])

    def test_an_empty_class_file_argument_falls_back_to_the_bundled_one(self):
        """A project that has not been labelled yet has no classes.txt."""
        args = build_arg_parser().parse_args(["/p", "", "/p", "--obb"])
        assert args.class_file == ""
        assert (args.class_file or "default") == "default"


# ---------------------------------------------------------------------------
# The format
# ---------------------------------------------------------------------------

class TestFormatConstants:

    def test_yolo_obb_is_a_format_of_its_own(self):
        assert LabelFileFormat.YOLO_OBB not in (
            LabelFileFormat.YOLO, LabelFileFormat.PASCAL_VOC, LabelFileFormat.CREATE_ML
        )

    def test_the_two_yolo_formats_have_distinct_names(self):
        assert FORMAT_YOLO_OBB != FORMAT_YOLO
        assert "OBB" in FORMAT_YOLO_OBB

    def test_the_obb_mode_is_remembered_under_its_own_setting(self):
        assert SETTING_OBB_MODE


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def make_shape(box):
    shape = Shape(label="fly")
    shape.points = [QPointF(x, y) for x, y in obb_corners(box)]
    shape.close()
    return shape


class TestShapeRotation:

    def test_a_hand_drawn_box_is_not_rotated(self):
        assert not make_shape((100, 100, 60, 20, 0.0)).is_rotated()

    def test_rotating_turns_the_box_without_resizing_it(self):
        shape = make_shape((100.0, 100.0, 60.0, 20.0, 0.0))
        shape.rotate(math.radians(30))
        cx, cy, w, h, theta = shape.obb()
        assert (cx, cy) == pytest.approx((100.0, 100.0), abs=1e-9)
        assert (w, h) == pytest.approx((60.0, 20.0), abs=1e-9)
        assert theta == pytest.approx(math.radians(30), abs=1e-9)
        assert shape.is_rotated()

    def test_rotation_accumulates(self):
        shape = make_shape((0.0, 0.0, 40.0, 10.0, 0.0))
        for _ in range(15):
            shape.rotate(math.radians(1))
        assert shape.obb()[4] == pytest.approx(math.radians(15), abs=1e-6)

    def test_rotating_by_a_full_turn_leaves_the_corners_where_they_were(self):
        shape = make_shape((50.0, 50.0, 30.0, 10.0, 0.3))
        before = [(p.x(), p.y()) for p in shape.points]
        shape.rotate(2 * math.pi)
        after = [(p.x(), p.y()) for p in shape.points]
        assert [v for p in after for v in p] == pytest.approx(
            [v for p in before for v in p], abs=1e-9
        )

    def test_the_centre_is_the_mean_of_the_corners(self):
        shape = make_shape((123.0, 45.0, 30.0, 10.0, math.radians(20)))
        centre = shape.center()
        assert (centre.x(), centre.y()) == pytest.approx((123.0, 45.0), abs=1e-9)

    def test_set_obb_replaces_the_corners(self):
        shape = Shape(label="fly")
        shape.set_obb(10.0, 20.0, 40.0, 12.0, math.radians(15))
        assert len(shape.points) == 4
        assert shape.obb() == pytest.approx(
            (10.0, 20.0, 40.0, 12.0, math.radians(15)), abs=1e-9
        )

    def test_a_shape_that_is_not_a_box_has_no_obb(self):
        shape = Shape(label="fly")
        shape.points = [QPointF(0, 0), QPointF(1, 1)]
        assert shape.obb() is None
        assert not shape.is_rotated()

    def test_a_copied_shape_keeps_its_rotation(self):
        shape = make_shape((50.0, 50.0, 30.0, 10.0, math.radians(40)))
        assert shape.copy().obb() == pytest.approx(shape.obb(), abs=1e-9)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

class TestSaveYoloObbFormat:

    @pytest.fixture
    def image(self):
        return QImage(640, 480, QImage.Format_RGB888)

    def save(self, tmp_path, image, boxes):
        shapes = [
            {
                "label": name,
                "points": obb_corners(box),
                "difficult": 0,
                "line_color": None,
                "fill_color": None,
            }
            for box, name in boxes
        ]
        target = tmp_path / "img.txt"
        LabelFile().save_yolo_obb_format(
            str(target), shapes, str(tmp_path / "img.png"), image, ["fly", "larva"]
        )
        return target

    def test_a_rotated_box_is_written_as_eight_coordinates(self, tmp_path, image):
        path = self.save(
            tmp_path, image,
            [((320.0, 240.0, 60.0, 20.0, math.radians(30)), "fly")],
        )
        fields = path.read_text(encoding="utf-8").split()
        assert len(fields) == 9
        assert fields[0] == "0"

    def test_the_angle_survives_the_save(self, tmp_path, image):
        box = (320.0, 240.0, 61.0, 18.0, math.radians(37))
        path = self.save(tmp_path, image, [(box, "fly")])
        values = [float(v) for v in path.read_text(encoding="utf-8").split()[1:]]
        corners = [(values[i] * 640, values[i + 1] * 480) for i in range(0, 8, 2)]
        assert corners_to_obb(corners)[4] == pytest.approx(box[4], abs=1e-4)

    def test_a_shape_that_is_not_a_rectangle_is_skipped(self, tmp_path, image):
        """An OBB file has no way to express anything but a rectangle."""
        shapes = [{
            "label": "fly",
            "points": [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)],
            "difficult": 0,
            "line_color": None,
            "fill_color": None,
        }]
        target = tmp_path / "img.txt"
        LabelFile().save_yolo_obb_format(
            str(target), shapes, str(tmp_path / "img.png"), image, ["fly"]
        )
        assert target.read_text(encoding="utf-8").strip() == ""

    def test_classes_txt_is_written_beside_the_label(self, tmp_path, image):
        self.save(tmp_path, image, [((100.0, 100.0, 20.0, 20.0, 0.0), "larva")])
        assert (tmp_path / "classes.txt").read_text(
            encoding="utf-8"
        ).split() == ["fly", "larva"]


# ---------------------------------------------------------------------------
# The remembered format, which may have been written by another labelImg
# ---------------------------------------------------------------------------

class ForeignLabelFileFormat(Enum):
    """Stands in for the pip-installed labelImg's enum, whose members are
    unequal to ours however identical they look."""

    PASCAL_VOC = 1
    YOLO = 2
    CREATE_ML = 3


class TestCoerceLabelFileFormat:

    def test_our_own_members_pass_straight_through(self):
        for member in LabelFileFormat:
            assert coerce_label_file_format(member) is member

    def test_a_foreign_enum_is_recovered_by_name(self):
        """Without this the window crashes before it opens."""
        assert ForeignLabelFileFormat.YOLO != LabelFileFormat.YOLO
        assert coerce_label_file_format(ForeignLabelFileFormat.YOLO) is LabelFileFormat.YOLO
        assert (coerce_label_file_format(ForeignLabelFileFormat.PASCAL_VOC)
                is LabelFileFormat.PASCAL_VOC)

    def test_a_bare_integer_is_recovered_by_value(self):
        assert coerce_label_file_format(4) is LabelFileFormat.YOLO_OBB

    @pytest.mark.parametrize("junk", [None, "yolo", 99, object()])
    def test_anything_unrecognised_falls_back_to_yolo(self, junk):
        assert coerce_label_file_format(junk) is LabelFileFormat.YOLO

    def test_the_fallback_can_be_chosen(self):
        assert coerce_label_file_format(
            None, default=LabelFileFormat.YOLO_OBB
        ) is LabelFileFormat.YOLO_OBB
