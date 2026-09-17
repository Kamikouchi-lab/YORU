"""End-to-end tests for Click to Box inside a real labelImg window.

These drive the actual ``MainWindow`` on Qt's ``offscreen`` platform: opening a
folder of images, arming the tool, emitting the click the canvas would emit,
and reading back the label file that lands on disk.  Everything between — the
segmentation, the shape, the label dialog bypass, the format choice, the writer
— runs for real, which is the only way to catch a break in the wiring rather
than in any one piece.

The window is made hermetic first: labelImg remembers its settings in the
user's home directory, and a test must neither depend on what is there nor
write to it.
"""

from __future__ import annotations

import math
import os

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 not installed")
cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")
np = pytest.importorskip("numpy")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPointF  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from yoru.labelimg.labelimg import MainWindow  # noqa: E402
from yoru.labelimg.libs.labelFile import LabelFileFormat  # noqa: E402
from yoru.labelimg.libs.settings import Settings  # noqa: E402
from yoru.labelimg.libs.yolo_obb_io import sniff_obb  # noqa: E402
from yoru.libs.obb import corners_to_obb  # noqa: E402

TRUE_ANGLE = 30.0
TRUE_LENGTH, TRUE_BREADTH = 60, 16


@pytest.fixture(scope="module")
def qapp():
    """One QApplication for the module; skip if no Qt platform can start."""
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as e:  # pragma: no cover - depends on the machine
            pytest.skip(f"cannot start Qt offscreen: {e}")
    return app


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch):
    """Never read or write the user's ``~/.labelImgSettings.pkl``."""
    monkeypatch.setattr(Settings, "load", lambda self: False)
    monkeypatch.setattr(Settings, "save", lambda self: True)


@pytest.fixture
def project(tmp_path):
    """A folder of frames, each with one tilted animal at a known angle."""
    for i in range(2):
        img = np.full((300, 400, 3), 215, np.uint8)
        cv2.ellipse(img, (200, 150), (TRUE_LENGTH // 2, TRUE_BREADTH // 2),
                    TRUE_ANGLE, 0, 360, (35, 35, 35), -1)
        a = math.radians(TRUE_ANGLE)
        for k in (-2, -1, 0, 1, 2):          # legs, which must not inflate the box
            for s in (-1, 1):
                x0 = int(200 + k * 9 * math.cos(a))
                y0 = int(150 + k * 9 * math.sin(a))
                x1 = int(x0 - s * 17 * math.sin(a))
                y1 = int(y0 + s * 17 * math.cos(a))
                cv2.line(img, (x0, y0), (x1, y1), (35, 35, 35), 1)
        cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"), img)
    (tmp_path / "classes.txt").write_text("fly\nlarva\n", encoding="utf-8")
    return tmp_path


def open_window(qapp, project, obb):
    win = MainWindow(str(project), str(project / "classes.txt"), str(project),
                     obb_mode=obb)
    qapp.processEvents()
    # Bypass the modal label dialog the way the "use default label" box does.
    win.use_default_label_checkbox.setChecked(True)
    win.default_label_text_line.setText("fly")
    return win


def click_at(qapp, win, x, y):
    win.actions.clickToBox.trigger()
    win.canvas.clickToBox.emit(QPointF(x, y))
    qapp.processEvents()


def shape_obb(shape):
    return corners_to_obb([(p.x(), p.y()) for p in shape.points])


# ---------------------------------------------------------------------------
# The project decides the format
# ---------------------------------------------------------------------------

class TestFormatFollowsTheProject:

    def test_an_obb_project_opens_in_yolo_obb(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        assert win.label_file_format == LabelFileFormat.YOLO_OBB
        assert win.obb_mode
        assert win.canvas.obb_mode

    def test_a_detection_project_opens_in_plain_yolo(self, qapp, project):
        win = open_window(qapp, project, obb=False)
        assert win.label_file_format == LabelFileFormat.YOLO
        assert not win.obb_mode
        assert not win.canvas.obb_mode

    def test_the_images_are_loaded(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        assert len(win.m_img_list) == 2
        assert win.file_path.endswith("frame_000.png")


# ---------------------------------------------------------------------------
# Arming, clicking, disarming
# ---------------------------------------------------------------------------

class TestClickToBox:

    def test_one_click_produces_one_labelled_box(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        click_at(qapp, win, 200, 150)
        assert len(win.canvas.shapes) == 1
        assert win.canvas.shapes[0].label == "fly"

    def test_the_box_fits_the_body_and_not_the_legs(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        click_at(qapp, win, 200, 150)
        cx, cy, w, h, theta = shape_obb(win.canvas.shapes[0])
        assert (cx, cy) == pytest.approx((200, 150), abs=3)
        assert w == pytest.approx(TRUE_LENGTH, abs=5)
        # The legs reach 17 px either side; the box must stay near the body.
        assert h == pytest.approx(TRUE_BREADTH, abs=5)
        assert math.degrees(theta) == pytest.approx(TRUE_ANGLE, abs=5)

    def test_the_tool_disarms_itself_once_a_box_lands(self, qapp, project):
        """The next click is a grab at a handle, not a request for a second box."""
        win = open_window(qapp, project, obb=True)
        click_at(qapp, win, 200, 150)
        assert not win.canvas.clicking()
        assert not win.actions.clickToBox.isChecked()

    def test_the_new_box_is_selected_ready_to_adjust(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        click_at(qapp, win, 200, 150)
        assert win.canvas.selected_shape is win.canvas.shapes[-1]

    def test_a_miss_leaves_the_tool_armed_and_explains_itself(self, qapp, project):
        """Re-arming the tool for each attempt would be its own chore."""
        win = open_window(qapp, project, obb=True)
        click_at(qapp, win, 20, 20)
        assert win.canvas.shapes == []
        assert win.canvas.clicking()
        assert win.statusBar().currentMessage()

    def test_escape_disarms_the_tool(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        win.actions.clickToBox.trigger()
        assert win.canvas.clicking()
        win.canvas.set_click_mode(False)
        qapp.processEvents()
        assert not win.actions.clickToBox.isChecked()

    def test_switching_to_draw_by_hand_takes_the_tool_down(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        win.actions.clickToBox.trigger()
        win.canvas.set_click_mode(False)
        win.canvas.set_editing(False)
        qapp.processEvents()
        assert not win.canvas.clicking()
        assert not win.actions.clickToBox.isChecked()

    def test_a_detection_project_gets_the_upright_box_around_the_fit(self, qapp, project):
        win = open_window(qapp, project, obb=False)
        click_at(qapp, win, 200, 150)
        shape = win.canvas.shapes[0]
        assert not shape.is_rotated()
        _cx, _cy, w, h, _theta = shape_obb(shape)
        # The bounding box of a 60 x 16 body tilted 30 degrees is wider and
        # much taller than the body itself.
        assert w > TRUE_LENGTH - 5
        assert h > TRUE_BREADTH + 10


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

class TestRotationActions:

    def test_rotation_is_available_for_a_selected_box_in_an_obb_project(
        self, qapp, project
    ):
        win = open_window(qapp, project, obb=True)
        assert not win.actions.rotateRight.isEnabled()
        click_at(qapp, win, 200, 150)
        assert win.actions.rotateRight.isEnabled()

    def test_rotation_stays_unavailable_in_a_detection_project(self, qapp, project):
        """There would be nowhere in a YOLO label file to put the angle."""
        win = open_window(qapp, project, obb=False)
        click_at(qapp, win, 200, 150)
        assert not win.actions.rotateRight.isEnabled()

    def test_the_big_step_turns_the_box_by_fifteen_degrees(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        click_at(qapp, win, 200, 150)
        before = math.degrees(shape_obb(win.canvas.shapes[0])[4])
        win.actions.rotateRightStep.trigger()
        after = math.degrees(shape_obb(win.canvas.shapes[0])[4])
        assert after - before == pytest.approx(15.0, abs=1e-6)

    def test_rotating_marks_the_file_as_unsaved(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        click_at(qapp, win, 200, 150)
        win.set_clean()
        win.actions.rotateLeft.trigger()
        assert win.dirty


# ---------------------------------------------------------------------------
# Save and reload
# ---------------------------------------------------------------------------

class TestSaveAndReload:

    def test_an_obb_project_writes_nine_fields_per_box(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        click_at(qapp, win, 200, 150)
        win.save_file()
        txt = project / "frame_000.txt"
        assert sniff_obb(str(txt)) is True
        assert len(txt.read_text(encoding="utf-8").split()) == 9

    def test_a_detection_project_writes_five(self, qapp, project):
        win = open_window(qapp, project, obb=False)
        click_at(qapp, win, 200, 150)
        win.save_file()
        txt = project / "frame_000.txt"
        assert sniff_obb(str(txt)) is False
        assert len(txt.read_text(encoding="utf-8").split()) == 5

    def test_a_rotated_box_comes_back_the_same(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        click_at(qapp, win, 200, 150)
        win.actions.rotateRightStep.trigger()
        saved = shape_obb(win.canvas.shapes[0])
        win.save_file()

        win.load_file(win.file_path)
        qapp.processEvents()
        assert len(win.canvas.shapes) == 1
        reloaded = shape_obb(win.canvas.shapes[0])
        assert reloaded[:4] == pytest.approx(saved[:4], abs=0.05)
        assert reloaded[4] == pytest.approx(saved[4], abs=1e-3)
        assert win.canvas.shapes[0].label == "fly"

    def test_reloading_keeps_the_obb_format(self, qapp, project):
        win = open_window(qapp, project, obb=True)
        click_at(qapp, win, 200, 150)
        win.save_file()
        win.load_file(win.file_path)
        qapp.processEvents()
        assert win.label_file_format == LabelFileFormat.YOLO_OBB

    def test_an_empty_label_file_does_not_flip_an_obb_project(self, qapp, project):
        """A frame with no animals must not switch the format to axis-aligned."""
        (project / "frame_001.txt").write_text("", encoding="utf-8")
        win = open_window(qapp, project, obb=True)
        win.load_file(str(project / "frame_001.png"))
        qapp.processEvents()
        assert win.label_file_format == LabelFileFormat.YOLO_OBB
        assert win.canvas.obb_mode

    def test_an_axis_aligned_file_opens_as_one_even_in_an_obb_session(
        self, qapp, project
    ):
        """The file's own contents decide; nine fields or five, nothing else."""
        (project / "frame_001.txt").write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
        win = open_window(qapp, project, obb=True)
        win.load_file(str(project / "frame_001.png"))
        qapp.processEvents()
        assert win.label_file_format == LabelFileFormat.YOLO
        assert len(win.canvas.shapes) == 1
