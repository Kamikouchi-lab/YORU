"""Tests for YOLO-OBB label files and the project flag that selects them.

Two things have to hold for an OBB project to work end to end:

* a rotated box survives write → read → write unchanged, or every save/load
  cycle in labelImg would shear the dataset a little further; and
* an ``.txt`` is identified by its **contents**, because OBB and plain YOLO
  share the extension, and an empty file (an image with no animals on it) must
  not be read as evidence that the project is axis-aligned.
"""

from __future__ import annotations

import math

import pytest

from yoru.labelimg.libs.yolo_obb_io import (
    AABB_FIELDS,
    OBB_FIELDS,
    YoloOBBReader,
    YoloOBBWriter,
    is_obb_file,
    sniff_obb,
)
from yoru.libs.create_yaml_train import (
    TASK_DETECT,
    TASK_OBB,
    is_obb_project,
    task_of,
)
from yoru.libs.obb import corners_to_obb, obb_corners

WIDTH, HEIGHT = 640, 480


class FakeImage:
    """The three methods the readers use out of a QImage."""

    def __init__(self, w=WIDTH, h=HEIGHT):
        self._w, self._h = w, h

    def width(self):
        return self._w

    def height(self):
        return self._h

    def isGrayscale(self):
        return False


def write_classes(tmp_path, names=("fly", "larva")):
    (tmp_path / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def save_boxes(tmp_path, boxes, classes=("fly", "larva")):
    """Write *boxes* (each ``(cx, cy, w, h, theta)``) and return the file path."""
    write_classes(tmp_path, classes)
    writer = YoloOBBWriter("folder", "img", img_size=(HEIGHT, WIDTH, 3))
    for box, name in boxes:
        writer.add_obb(obb_corners(box), name, 0)
    target = tmp_path / "img.txt"
    writer.save(class_list=list(classes), target_file=str(target))
    return target


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

class TestYoloOBBWriter:

    def test_one_line_per_box_with_nine_fields(self, tmp_path):
        path = save_boxes(tmp_path, [
            ((320.0, 240.0, 60.0, 20.0, math.radians(30)), "fly"),
            ((100.0, 100.0, 40.0, 40.0, 0.0), "larva"),
        ])
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            assert len(line.split()) == OBB_FIELDS

    def test_coordinates_are_normalised(self, tmp_path):
        path = save_boxes(tmp_path, [((320.0, 240.0, 64.0, 48.0, 0.0), "fly")])
        fields = path.read_text(encoding="utf-8").split()
        assert fields[0] == "0"
        values = [float(v) for v in fields[1:]]
        assert all(0.0 <= v <= 1.0 for v in values)
        # Upright 64 x 48 box centred in a 640 x 480 image.
        assert values[0] == pytest.approx(0.45, abs=1e-4)   # x1
        assert values[1] == pytest.approx(0.45, abs=1e-4)   # y1
        assert values[4] == pytest.approx(0.55, abs=1e-4)   # x3
        assert values[5] == pytest.approx(0.55, abs=1e-4)   # y3

    def test_a_corner_past_the_edge_is_clamped_into_the_unit_square(self, tmp_path):
        """Ultralytics rejects a label file with a coordinate outside [0, 1]."""
        path = save_boxes(tmp_path, [((10.0, 10.0, 80.0, 40.0, math.radians(30)), "fly")])
        values = [float(v) for v in path.read_text(encoding="utf-8").split()[1:]]
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_a_box_at_the_edge_is_clamped_rather_than_moved_or_dropped(self, tmp_path):
        """Clamping keeps the box on the animal; the other options do not.

        Sliding it back inside would move it off the animal and dropping it
        would lose the annotation, so a corner poking past the frame is pinned
        to the nearest legal point even though that leaves the quadrilateral
        very slightly out of true.
        """
        path = save_boxes(tmp_path, [((10.0, 10.0, 80.0, 40.0, math.radians(30)), "fly")])
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1                       # not dropped
        values = [float(v) for v in lines[0].split()[1:]]
        assert min(values) == 0.0                    # pinned, not slid inwards
        # The corners that were already inside the frame are untouched.
        inside = [v for v in values if v > 0.0]
        assert inside

    def test_a_box_well_inside_the_frame_is_untouched_by_the_clamp(self, tmp_path):
        box = (320.0, 240.0, 61.0, 18.0, math.radians(37))
        path = save_boxes(tmp_path, [(box, "fly")])
        values = [float(v) for v in path.read_text(encoding="utf-8").split()[1:]]
        corners = [(values[i] * WIDTH, values[i + 1] * HEIGHT) for i in range(0, 8, 2)]
        assert corners_to_obb(corners) == pytest.approx(box, abs=1e-2)

    def test_class_indices_follow_the_class_list(self, tmp_path):
        path = save_boxes(tmp_path, [
            ((100.0, 100.0, 20.0, 20.0, 0.0), "larva"),
            ((200.0, 200.0, 20.0, 20.0, 0.0), "fly"),
        ])
        indices = [line.split()[0] for line in
                   path.read_text(encoding="utf-8").strip().splitlines()]
        assert indices == ["1", "0"]

    def test_an_unknown_class_is_appended_to_the_list(self, tmp_path):
        write_classes(tmp_path)
        writer = YoloOBBWriter("folder", "img", img_size=(HEIGHT, WIDTH, 3))
        writer.add_obb(obb_corners((100.0, 100.0, 20.0, 20.0, 0.0)), "beetle", 0)
        class_list = ["fly", "larva"]
        writer.save(class_list=class_list, target_file=str(tmp_path / "img.txt"))
        assert class_list == ["fly", "larva", "beetle"]

    def test_classes_txt_is_rewritten_beside_the_labels(self, tmp_path):
        save_boxes(tmp_path, [((100.0, 100.0, 20.0, 20.0, 0.0), "fly")])
        written = (tmp_path / "classes.txt").read_text(encoding="utf-8")
        assert written.split() == ["fly", "larva"]

    def test_no_boxes_writes_an_empty_file_not_a_missing_one(self, tmp_path):
        path = save_boxes(tmp_path, [])
        assert path.is_file()
        assert path.read_text(encoding="utf-8").strip() == ""


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class TestYoloOBBReader:

    @pytest.mark.parametrize("degrees", [0, 13, 30, 45, 67, -22, -80])
    def test_a_rotated_box_survives_the_round_trip(self, tmp_path, degrees):
        box = (320.0, 240.0, 61.0, 18.0, math.radians(degrees))
        path = save_boxes(tmp_path, [(box, "fly")])
        shapes = YoloOBBReader(str(path), FakeImage()).get_shapes()
        assert len(shapes) == 1
        label, points, _lc, _fc, difficult = shapes[0]
        assert label == "fly"
        assert difficult is False
        cx, cy, w, h, theta = corners_to_obb(points)
        # Tolerance is the 6-decimal normalised precision the writer uses,
        # scaled back up by the image size.
        assert (cx, cy) == pytest.approx(box[:2], abs=1e-2)
        assert (w, h) == pytest.approx(box[2:4], abs=1e-2)
        assert theta == pytest.approx(box[4], abs=1e-4)

    def test_corners_are_kept_as_floats(self, tmp_path):
        """Rounding each corner would shear the box a little on every save."""
        box = (320.0, 240.0, 61.0, 18.0, math.radians(37))
        path = save_boxes(tmp_path, [(box, "fly")])
        _label, points, *_ = YoloOBBReader(str(path), FakeImage()).get_shapes()[0]
        assert any(abs(v - round(v)) > 1e-6 for p in points for v in p)

    def test_two_round_trips_do_not_drift(self, tmp_path):
        box = (320.0, 240.0, 61.0, 18.0, math.radians(37))
        first = save_boxes(tmp_path, [(box, "fly")])
        points = YoloOBBReader(str(first), FakeImage()).get_shapes()[0][1]

        writer = YoloOBBWriter("folder", "img", img_size=(HEIGHT, WIDTH, 3))
        writer.add_obb(points, "fly", 0)
        second = tmp_path / "img2.txt"
        writer.save(class_list=["fly", "larva"], target_file=str(second))

        assert (first.read_text(encoding="utf-8").split()
                == second.read_text(encoding="utf-8").split())

    def test_multiple_boxes_keep_their_labels(self, tmp_path):
        path = save_boxes(tmp_path, [
            ((100.0, 100.0, 30.0, 10.0, 0.2), "fly"),
            ((300.0, 300.0, 30.0, 10.0, -0.4), "larva"),
        ])
        labels = [s[0] for s in YoloOBBReader(str(path), FakeImage()).get_shapes()]
        assert labels == ["fly", "larva"]

    def test_an_empty_file_yields_no_shapes(self, tmp_path):
        path = save_boxes(tmp_path, [])
        assert YoloOBBReader(str(path), FakeImage()).get_shapes() == []

    def test_a_stray_axis_aligned_line_is_skipped_not_fatal(self, tmp_path):
        """One bad line must not cost the user the whole image's annotations."""
        write_classes(tmp_path)
        path = tmp_path / "img.txt"
        path.write_text(
            "0 0.5 0.5 0.1 0.1\n"
            "1 0.10 0.10 0.20 0.10 0.20 0.20 0.10 0.20\n",
            encoding="utf-8",
        )
        shapes = YoloOBBReader(str(path), FakeImage()).get_shapes()
        assert len(shapes) == 1
        assert shapes[0][0] == "larva"

    def test_a_class_index_past_the_class_list_falls_back_to_the_number(self, tmp_path):
        write_classes(tmp_path)
        path = tmp_path / "img.txt"
        path.write_text(
            "7 0.10 0.10 0.20 0.10 0.20 0.20 0.10 0.20\n", encoding="utf-8"
        )
        assert YoloOBBReader(str(path), FakeImage()).get_shapes()[0][0] == "7"


# ---------------------------------------------------------------------------
# Telling the two .txt formats apart
# ---------------------------------------------------------------------------

class TestSniffObb:

    def test_nine_fields_is_obb(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8")
        assert sniff_obb(str(path)) is True
        assert is_obb_file(str(path))

    def test_five_fields_is_axis_aligned(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        assert sniff_obb(str(path)) is False
        assert not is_obb_file(str(path))

    def test_an_empty_file_says_nothing_either_way(self, tmp_path):
        """A frame with no animals must not flip an OBB project's format."""
        path = tmp_path / "a.txt"
        path.write_text("", encoding="utf-8")
        assert sniff_obb(str(path)) is None
        assert not is_obb_file(str(path))

    def test_a_file_of_blank_lines_says_nothing_either_way(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("\n\n   \n", encoding="utf-8")
        assert sniff_obb(str(path)) is None

    def test_a_missing_file_says_nothing_either_way(self, tmp_path):
        assert sniff_obb(str(tmp_path / "nope.txt")) is None

    def test_leading_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("\n\n0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8")
        assert sniff_obb(str(path)) is True

    def test_the_two_field_counts_are_what_the_formats_use(self):
        assert OBB_FIELDS == 9
        assert AABB_FIELDS == 5


# ---------------------------------------------------------------------------
# The project flag
# ---------------------------------------------------------------------------

class TestProjectTask:

    def test_an_obb_project_writes_task_obb(self):
        assert task_of({"obb": True}) == TASK_OBB
        assert task_of({"obb": False}) == TASK_DETECT
        assert task_of({}) == TASK_DETECT

    def test_a_config_with_task_obb_is_an_obb_project(self):
        assert is_obb_project({"task": "obb"})
        assert is_obb_project({"task": "OBB"})
        assert is_obb_project({"task": " obb "})

    def test_a_config_with_task_detect_is_not(self):
        assert not is_obb_project({"task": "detect"})

    def test_a_project_made_before_obb_support_reads_as_detection(self):
        """No ``task`` key at all: every project created before this feature."""
        assert not is_obb_project({"path": "/x", "train": "/x/train"})
        assert not is_obb_project({})
        assert not is_obb_project(None)

    def test_a_hand_written_obb_true_is_accepted(self):
        assert is_obb_project({"obb": True})
        assert not is_obb_project({"obb": False})

    def test_task_wins_over_a_bare_obb_key(self):
        assert not is_obb_project({"task": "detect", "obb": True})
