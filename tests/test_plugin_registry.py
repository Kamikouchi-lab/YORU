"""Tests for the detection/training backend registry.

The registry replaced the old yolo_wrapper (whose tests were deleted with it),
and had no coverage of its own.
"""

import pytest

from yoru.libs import plugins
from yoru.libs.detector_base import DetectorBase


@pytest.fixture
def fake_detector():
    """Register a throwaway backend and remove it again afterwards."""
    seen = {}

    @plugins.register_detector("_test_fake")
    class _Fake(DetectorBase):
        def load(self, model_path, **kwargs):
            seen["model_path"] = model_path
            seen.update(kwargs)

        @property
        def names(self):
            return {}

        def detect(self, image):
            return []

    yield seen
    plugins._DETECTOR_REGISTRY.pop("_test_fake", None)


def test_shared_thresholds_are_forwarded(fake_detector):
    plugins.get_detector("_test_fake", "model.pt")
    assert fake_detector["conf_thresh"] == plugins.DEFAULT_CONF_THRESH
    assert fake_detector["iou_thresh"] == plugins.DEFAULT_IOU_THRESH


def test_explicit_thresholds_override_the_defaults(fake_detector):
    plugins.get_detector("_test_fake", "model.pt", conf_thresh=0.6, iou_thresh=0.2)
    assert fake_detector["conf_thresh"] == 0.6
    assert fake_detector["iou_thresh"] == 0.2


def test_unknown_backend_reports_what_is_available():
    with pytest.raises(ValueError) as exc:
        plugins.get_detector("no_such_backend", "model.pt")
    assert "no_such_backend" in str(exc.value)


@pytest.mark.parametrize(
    "config_value,expected",
    [
        ("yolov8", "ultralytics"),
        ("yolo11", "ultralytics"),
        # v1 configs still say yolov5; it must not become an unknown backend.
        ("yolov5", "ultralytics"),
        ("fasterrcnn", "torchvision"),
        ("maskrcnn", "torchvision"),
        ("ssd", "torchvision"),
        ("onnx", "onnx"),
    ],
)
def test_backend_aliases(config_value, expected):
    assert plugins._normalize_backend(config_value) == expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("best.onnx", "onnx"),
        ("rtdetr-l.pt", "rtdetr"),
        ("fasterrcnn_resnet50_best.pt", "torchvision"),
        ("maskrcnn_resnet50_best.pt", "torchvision"),
        ("ssd_vgg16_best.pt", "torchvision"),
        ("yolo11s.pt", "ultralytics"),
        ("yolov8m.pt", "ultralytics"),
    ],
)
def test_auto_detect_backend_from_filename(filename, expected):
    assert plugins._auto_detect_backend(filename) == expected


def test_auto_detect_falls_back_for_unknown_files(tmp_path):
    """An unreadable/ambiguous file must not raise, just fall back."""
    p = tmp_path / "best.pt"
    p.write_bytes(b"not a torch checkpoint")
    assert plugins._sniff_checkpoint(str(p)) is None
    assert plugins._auto_detect_backend(str(p)) == "ultralytics"


def test_sniff_checkpoint_handles_a_missing_file():
    assert plugins._sniff_checkpoint("does_not_exist.pt") is None


@pytest.mark.parametrize(
    "m_dict,expected",
    [
        ({"model_family": "YOLO", "weight": "yolo11s.pt"}, "ultralytics"),
        ({"model_family": "RT-DETR", "weight": "rtdetr-l.pt"}, "ultralytics"),
        ({"model_family": "Faster R-CNN"}, "torchvision"),
        ({"model_family": "Mask R-CNN"}, "torchvision"),
        ({"model_family": "SSD"}, "torchvision"),
    ],
)
def test_detect_trainer_backend(m_dict, expected):
    assert plugins.detect_trainer_backend(m_dict) == expected
