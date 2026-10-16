"""The names sister applications build on, and the rule that keeps YORU apart.

docs/external_api.md promises that the names below stay put.  YORU-Tracker
and YORU-Explain import them; nothing in this repository would otherwise
notice if one were renamed or had a parameter dropped, so each is pinned here.
The last test is the dependency rule itself: YORU never imports a sister
application.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

# module -> names it must export
PUBLIC_NAMES = {
    "yoru.libs.plugins": (
        "get_detector", "list_detector_backends",
        "DEFAULT_CONF_THRESH", "DEFAULT_IOU_THRESH",
    ),
    # The model behind get_detector("yolov5", ...), for YORU-Explain.
    "yoru.libs.plugins.yolov5_detector": (
        "load_yolov5_model", "rgb_input_requested", "V1_NMS_CONF",
    ),
    "yoru.libs.yolov5_support": ("yolov5_importable", "YOLOV5_DIR"),
    "yoru.libs.device": ("torch_device",),
    "yoru.libs.detector_base": (
        "DetectorBase", "DETECTION_COLUMNS", "obb_of", "detection_row",
    ),
    "yoru.libs.obb": (
        "normalize_angle", "obb_corners", "corners_to_obb", "obb_to_aabb",
        "aabb_to_obb", "point_in_obb", "rotate_points", "is_rotated",
        "box_axes", "resize_corner", "ANGLE_SNAP",
    ),
    "yoru.libs.camera": ("open_camera",),
    "yoru.gui_base": (
        "apply_default_theme", "process_frame",
        "frame_to_data_rgb", "frame_to_data_rgba",
    ),
    "yoru.gui_layout": ("GuiSession",),
    "yoru.libs.gui_error": ("GuiErrorMixin",),
    "yoru.libs.user_paths": (
        "get_yoru_home", "get_log_dir", "get_log_file", "setup_logging",
        "log_exception", "log_message",
    ),
}


@pytest.mark.parametrize("module", sorted(PUBLIC_NAMES))
def test_public_names_exist(module):
    mod = importlib.import_module(module)
    missing = [name for name in PUBLIC_NAMES[module] if not hasattr(mod, name)]
    assert not missing, f"{module} no longer exports {missing}"


def _params(func):
    return list(inspect.signature(func).parameters)


def test_get_detector_signature():
    from yoru.libs.plugins import get_detector

    assert _params(get_detector)[:4] == [
        "backend", "model_path", "conf_thresh", "iou_thresh",
    ]


def test_list_detector_backends_starts_with_auto():
    from yoru.libs.plugins import list_detector_backends

    names = list_detector_backends()
    assert names[0] == "auto"
    assert names[1:] == sorted(names[1:])


def test_detection_columns_prefix_is_stable():
    """Trigger plugins and the tracker's CSV both index rows by position."""
    from yoru.libs.detector_base import DETECTION_COLUMNS

    assert DETECTION_COLUMNS == (
        "x1", "y1", "x2", "y2",
        "confidence", "class", "class_name", "total_time",
        "cx", "cy", "w", "h", "angle",
    )


def test_obb_of_fills_in_an_upright_box():
    from yoru.libs.detector_base import obb_of

    assert obb_of({"x1": 0, "y1": 0, "x2": 4, "y2": 2}) == (2.0, 1.0, 4.0, 2.0, 0.0)


def test_open_camera_signature():
    from yoru.libs.camera import open_camera

    params = inspect.signature(open_camera).parameters
    assert list(params)[:4] == ["src", "width", "height", "fps"]
    assert params["settings_dialog"].kind is inspect.Parameter.KEYWORD_ONLY


def test_gui_session_methods():
    from yoru.gui_layout import GuiSession

    for name in ("begin", "window_kwargs", "add_layout_menu", "finish",
                 "content_region"):
        assert callable(getattr(GuiSession, name, None)), name


def test_yolov5_entry_point_signatures():
    """What YORU-Explain calls to get the model and settings the YOLOv5 plugin uses."""
    from yoru.libs.device import torch_device
    from yoru.libs.plugins.yolov5_detector import (
        V1_NMS_CONF,
        load_yolov5_model,
        rgb_input_requested,
    )

    assert _params(load_yolov5_model)[:2] == ["model_path", "device"]
    assert _params(rgb_input_requested)[:1] == ["value"]
    assert _params(torch_device)[:1] == ["preference"]
    assert V1_NMS_CONF == 0.25


SISTER_PACKAGES = ("yoru_tracker", "yoru_explain")


def test_yoru_never_imports_a_sister_application(repo_root: Path):
    """YORU must run without its sister applications installed: no import, anywhere."""
    offenders = []
    for path in sorted((repo_root / "yoru").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] in SISTER_PACKAGES for n in names):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
    assert not offenders, "YORU imports a sister application:\n" + "\n".join(offenders)
