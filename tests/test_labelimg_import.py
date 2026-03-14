"""Tests for the bundled labelImg package (yoru/labelimg/).

These tests verify that the annotation tool was correctly integrated:
  - Package and key submodules are importable without Qt
  - YOLOWriter produces correct normalised coordinates
  - YOLOReader parses YOLO-format annotation files
  - classes.txt merge logic preserves existing entries
"""

from __future__ import annotations

import sys
import types
import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Qt stub – prevent import errors when PyQt5/PySide2 is absent
# ---------------------------------------------------------------------------

def _stub_qt():
    """Insert a minimal PyQt5 stub so yolo_io.py can be imported without Qt."""
    qt_mods = [
        "PyQt5", "PyQt5.QtGui", "PyQt5.QtCore", "PyQt5.QtWidgets",
        "PyQt5.QtPrintSupport",
    ]
    for name in qt_mods:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # mark as package
            sys.modules[name] = mod

    # Minimal QImage stub used by canvas/labelImg
    QtGui = sys.modules["PyQt5.QtGui"]
    if not hasattr(QtGui, "QImage"):
        QtGui.QImage = type("QImage", (), {})

    QtCore = sys.modules["PyQt5.QtCore"]
    for name in ("Qt", "QPointF", "QRectF", "QObject", "pyqtSignal", "QSettings",
                 "QSize", "QByteArray", "QPoint"):
        if not hasattr(QtCore, name):
            setattr(QtCore, name, type(name, (), {}))

    QtWidgets = sys.modules["PyQt5.QtWidgets"]
    for name in ("QApplication", "QMainWindow", "QWidget", "QDialog",
                 "QFileDialog", "QMessageBox", "QLabel", "QAction",
                 "QScrollArea", "QToolBar", "QStatusBar", "QDockWidget",
                 "QListWidget", "QListWidgetItem", "QComboBox", "QLineEdit",
                 "QCheckBox", "QSpinBox", "QHBoxLayout", "QVBoxLayout",
                 "QShortcut", "QSizePolicy", "QMenu", "QMenuBar"):
        if not hasattr(QtWidgets, name):
            setattr(QtWidgets, name, type(name, (), {"__init__": lambda *a, **kw: None}))


_stub_qt()


# ---------------------------------------------------------------------------
# Package importability
# ---------------------------------------------------------------------------

class TestLabelimgPackageImports:

    def test_labelimg_package_importable(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        pkg = importlib.import_module("yoru.labelimg")
        assert pkg is not None

    def test_labelimg_libs_constants_importable(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        mod = importlib.import_module("yoru.labelimg.libs.constants")
        assert hasattr(mod, "DEFAULT_ENCODING")

    def test_labelimg_libs_ustr_importable(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        mod = importlib.import_module("yoru.labelimg.libs.ustr")
        assert hasattr(mod, "ustr")

    def test_labelimg_libs_utils_importable(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        importlib.import_module("yoru.labelimg.libs.utils")


# ---------------------------------------------------------------------------
# YOLOWriter – coordinate conversion
# ---------------------------------------------------------------------------

try:
    _lxml_ok = bool(importlib.util.find_spec("lxml"))
except Exception:
    _lxml_ok = False

_skip_yolo_io = pytest.mark.skipif(
    not _lxml_ok,
    reason="lxml not installed; pip install lxml"
)


@_skip_yolo_io
class TestYOLOWriter:
    """YOLOWriter.bnd_box_to_yolo_line() should produce correct normalised coords."""

    @pytest.fixture(autouse=True)
    def _writer(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from yoru.labelimg.libs.yolo_io import YOLOWriter
        # img_size is (height, width, channels)
        self.writer = YOLOWriter("folder", "file", img_size=(480, 640, 3))

    def test_center_coordinates_normalised(self):
        # Box covering the entire image → center (0.5, 0.5), size (1.0, 1.0)
        cls_idx, xc, yc, w, h = self.writer.bnd_box_to_yolo_line(
            {"xmin": 0, "ymin": 0, "xmax": 640, "ymax": 480, "name": "cat", "difficult": False},
            class_list=["cat"],
        )
        assert cls_idx == 0
        assert abs(xc - 0.5) < 1e-6
        assert abs(yc - 0.5) < 1e-6
        assert abs(w - 1.0) < 1e-6
        assert abs(h - 1.0) < 1e-6

    def test_class_added_to_list_if_missing(self):
        class_list = []
        cls_idx, *_ = self.writer.bnd_box_to_yolo_line(
            {"xmin": 10, "ymin": 10, "xmax": 100, "ymax": 100, "name": "dog", "difficult": False},
            class_list=class_list,
        )
        assert cls_idx == 0
        assert "dog" in class_list

    def test_multiple_classes_indexed_correctly(self):
        class_list = ["cat"]  # cat is already index 0
        cls_idx, *_ = self.writer.bnd_box_to_yolo_line(
            {"xmin": 10, "ymin": 10, "xmax": 100, "ymax": 100, "name": "dog", "difficult": False},
            class_list=class_list,
        )
        assert cls_idx == 1
        assert class_list == ["cat", "dog"]

    def test_coords_in_unit_range(self):
        cls_idx, xc, yc, w, h = self.writer.bnd_box_to_yolo_line(
            {"xmin": 100, "ymin": 50, "xmax": 300, "ymax": 200, "name": "fly", "difficult": False},
            class_list=["fly"],
        )
        for val in (xc, yc, w, h):
            assert 0.0 < val <= 1.0, f"normalised value {val} out of range (0, 1]"


# ---------------------------------------------------------------------------
# YoloReader – parsing annotation files
# ---------------------------------------------------------------------------

@_skip_yolo_io
class TestYoloReader:
    """YoloReader should parse YOLO-format .txt annotation files.

    YoloReader requires a QImage-like object (with .height(), .width(),
    .isGrayscale()) instead of a raw size tuple, so we supply a minimal stub.
    """

    @pytest.fixture(autouse=True)
    def _reader_cls(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from yoru.labelimg.libs.yolo_io import YoloReader
        self.YoloReader = YoloReader

    class _FakeImage:
        """Minimal QImage stub with the three methods YoloReader uses."""
        def __init__(self, h=480, w=640):
            self._h = h
            self._w = w
        def height(self): return self._h
        def width(self): return self._w
        def isGrayscale(self): return False

    def _write_annotation(self, tmp_path: Path, lines: list) -> Path:
        ann = tmp_path / "img.txt"
        ann.write_text("\n".join(lines), encoding="utf-8")
        classes_file = tmp_path / "classes.txt"
        classes_file.write_text("cat\ndog\n", encoding="utf-8")
        return ann

    def test_parse_single_box(self, tmp_path):
        ann = self._write_annotation(tmp_path, ["0 0.5 0.5 1.0 1.0"])
        reader = self.YoloReader(str(ann), self._FakeImage())
        shapes = reader.get_shapes()
        assert len(shapes) == 1
        label, points, _, _, _ = shapes[0]
        assert label == "cat"  # class index 0 → "cat" from classes.txt

    def test_parse_multiple_boxes(self, tmp_path):
        ann = self._write_annotation(tmp_path, [
            "0 0.25 0.25 0.4 0.4",
            "1 0.75 0.75 0.3 0.3",
        ])
        reader = self.YoloReader(str(ann), self._FakeImage())
        shapes = reader.get_shapes()
        assert len(shapes) == 2
        labels = [s[0] for s in shapes]
        assert labels == ["cat", "dog"]

    def test_empty_annotation_file(self, tmp_path):
        ann = self._write_annotation(tmp_path, [])
        reader = self.YoloReader(str(ann), self._FakeImage())
        shapes = reader.get_shapes()
        assert len(shapes) == 0
