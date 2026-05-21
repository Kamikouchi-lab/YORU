"""Unit tests for yoru/libs/yolo_wrapper.py.

Covers:
  - detect_model_type(): filename-based inference for all supported types
  - detect_model_type(): checkpoint inspection fallback for ambiguous names
  - load_yolo_model(): factory routing to the correct wrapper class
  - TorchvisionResult: result format conversion (xyxy / xywhn)
  - UltralyticsResult: result format conversion with zero detections
"""

from __future__ import annotations

import sys
import types
import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch, DEFAULT

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


def _import_wrapper():
    """Import yolo_wrapper with the repo root on sys.path."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("yoru.libs.yolo_wrapper")


# ---------------------------------------------------------------------------
# detect_model_type – filename inference
# ---------------------------------------------------------------------------

class TestDetectModelTypeFromFilename:
    """detect_model_type should infer the type from the filename alone."""

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _import_wrapper()

    @pytest.mark.parametrize("filename,expected", [
        # RT-DETR variants
        ("rtdetr_l.pt",          "rtdetr"),
        ("rt-detr_best.pt",      "rtdetr"),
        ("my_rtdetr_model.pt",   "rtdetr"),
        # Faster R-CNN variants
        ("fasterrcnn_model.pt",  "fasterrcnn"),
        ("faster_rcnn_best.pt",  "fasterrcnn"),
        ("faster-rcnn.pt",       "fasterrcnn"),
        # Mask R-CNN variants
        ("maskrcnn_epoch10.pt",  "maskrcnn"),
        ("mask_rcnn.pt",         "maskrcnn"),
        ("mask-rcnn_best.pt",    "maskrcnn"),
        # SSD variants
        ("ssd_model.pt",         "ssd"),
        ("ssd300_vgg16.pt",      "ssd"),
        ("model_ssd.pt",         "ssd"),
        # YOLO11 variants
        ("yolo11n.pt",           "yolo11"),
        ("yolov11_custom.pt",    "yolo11"),
        # YOLOv8 variants
        ("yolov8n.pt",           "yolov8"),
        ("yolo8_custom.pt",      "yolov8"),
        # YOLOv5 (explicit in name → falls through to yolov5 default)
        ("yolov5s.pt",           "yolov5"),
    ])
    def test_filename_inference(self, filename, expected, tmp_path):
        # detect_model_type needs the file to exist only for the fallback path;
        # for unambiguous names we never reach torch.load.
        dummy = tmp_path / filename
        dummy.write_bytes(b"")
        result = self.mod.detect_model_type(str(dummy))
        assert result == expected, f"filename={filename!r}: expected {expected!r}, got {result!r}"


# ---------------------------------------------------------------------------
# detect_model_type – checkpoint inspection fallback
# ---------------------------------------------------------------------------

class TestDetectModelTypeCheckpointFallback:
    """For ambiguous filenames (e.g. 'best.pt'), detect_model_type inspects
    the checkpoint dictionary for a 'model_type' key or ultralytics class."""

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _import_wrapper()

    def test_torchvision_fasterrcnn_checkpoint(self, tmp_path):
        dummy = tmp_path / "best.pt"
        dummy.write_bytes(b"")
        ckpt = {"model_type": "fasterrcnn", "num_classes": 2, "names": {0: "cat"}}
        with patch("torch.load", return_value=ckpt):
            assert self.mod.detect_model_type(str(dummy)) == "fasterrcnn"

    def test_torchvision_maskrcnn_checkpoint(self, tmp_path):
        dummy = tmp_path / "best.pt"
        dummy.write_bytes(b"")
        ckpt = {"model_type": "maskrcnn", "num_classes": 3, "names": {0: "dog"}}
        with patch("torch.load", return_value=ckpt):
            assert self.mod.detect_model_type(str(dummy)) == "maskrcnn"

    def test_torchvision_ssd_checkpoint(self, tmp_path):
        dummy = tmp_path / "weights.pt"
        dummy.write_bytes(b"")
        ckpt = {"model_type": "ssd"}
        with patch("torch.load", return_value=ckpt):
            assert self.mod.detect_model_type(str(dummy)) == "ssd"

    def test_ultralytics_ema_checkpoint(self, tmp_path):
        dummy = tmp_path / "best.pt"
        dummy.write_bytes(b"")
        # Create a real class whose __module__ contains "ultralytics"
        UltralyticsDetectionModel = type(
            "DetectionModel",
            (),
            {"__module__": "ultralytics.nn.tasks"},
        )
        fake_model = UltralyticsDetectionModel()
        ckpt = {"ema": fake_model}
        with patch("torch.load", return_value=ckpt):
            assert self.mod.detect_model_type(str(dummy)) == "yolov8"

    def test_fallback_to_yolov5_on_load_error(self, tmp_path):
        dummy = tmp_path / "ambiguous.pt"
        dummy.write_bytes(b"")
        with patch("torch.load", side_effect=RuntimeError("corrupted")):
            assert self.mod.detect_model_type(str(dummy)) == "yolov5"

    def test_fallback_to_yolov5_for_plain_dict_no_model_type(self, tmp_path):
        dummy = tmp_path / "best.pt"
        dummy.write_bytes(b"")
        # A dict without 'model_type' and without 'ema'/'model' keys
        with patch("torch.load", return_value={"epoch": 5, "optimizer": {}}):
            assert self.mod.detect_model_type(str(dummy)) == "yolov5"


# ---------------------------------------------------------------------------
# load_yolo_model – factory routing
# ---------------------------------------------------------------------------

class TestLoadYoloModelFactory:
    """load_yolo_model should instantiate the correct wrapper class."""

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _import_wrapper()

    def _patch_wrappers(self):
        """Return a context-manager that patches all four wrapper constructors.

        patch.multiple with DEFAULT creates new MagicMocks and returns them
        in the context-manager's dict, keyed by attribute name.
        """
        return patch.multiple(
            self.mod,
            YOLOv5Wrapper=DEFAULT,
            UltralyticsWrapper=DEFAULT,
            RTDETRWrapper=DEFAULT,
            TorchvisionWrapper=DEFAULT,
        )

    @pytest.mark.parametrize("model_type,wrapper_attr", [
        ("yolov5",    "YOLOv5Wrapper"),
        ("yolov8",    "UltralyticsWrapper"),
        ("yolo11",    "UltralyticsWrapper"),
        ("rtdetr",    "RTDETRWrapper"),
        ("fasterrcnn","TorchvisionWrapper"),
        ("maskrcnn",  "TorchvisionWrapper"),
        ("ssd",       "TorchvisionWrapper"),
    ])
    def test_explicit_model_type_routing(self, model_type, wrapper_attr, tmp_path):
        dummy = tmp_path / "model.pt"
        dummy.write_bytes(b"")
        with self._patch_wrappers() as mocks:
            self.mod.load_yolo_model(str(dummy), model_type=model_type)
            mocks[wrapper_attr].assert_called_once_with(str(dummy))

    def test_auto_mode_routes_via_filename(self, tmp_path):
        dummy = tmp_path / "rtdetr_best.pt"
        dummy.write_bytes(b"")
        with self._patch_wrappers() as mocks:
            self.mod.load_yolo_model(str(dummy), model_type="auto")
            mocks["RTDETRWrapper"].assert_called_once_with(str(dummy))

    def test_auto_mode_defaults_to_yolov5_for_ambiguous(self, tmp_path):
        dummy = tmp_path / "best.pt"
        dummy.write_bytes(b"")
        with patch("torch.load", side_effect=RuntimeError("no model")):
            with self._patch_wrappers() as mocks:
                self.mod.load_yolo_model(str(dummy), model_type="auto")
                mocks["YOLOv5Wrapper"].assert_called_once_with(str(dummy))


# ---------------------------------------------------------------------------
# TorchvisionResult – result format
# ---------------------------------------------------------------------------

class TestTorchvisionResult:
    """TorchvisionResult should convert torchvision output to the unified format."""

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _import_wrapper()

    def _make_output(self, n=3, img_h=480, img_w=640):
        boxes  = torch.tensor([[10, 20, 50, 60],
                                [100, 150, 200, 250],
                                [300, 100, 400, 200]], dtype=torch.float32)[:n]
        scores = torch.tensor([0.9, 0.8, 0.7])[:n]
        labels = torch.tensor([1, 2, 1])[:n]   # 1-indexed
        return {"boxes": boxes, "scores": scores, "labels": labels}

    def test_xyxy_shape_and_columns(self):
        out = self._make_output(n=3)
        result = self.mod.TorchvisionResult(out, img_shape=(480, 640, 3), conf_thresh=0.5)
        xyxy = result.xyxy[0]
        assert xyxy.shape == (3, 6), f"expected (3,6), got {xyxy.shape}"
        # conf and cls must be the last two columns
        assert (xyxy[:, 4] > 0).all(), "confidence should be > 0"
        assert set(xyxy[:, 5].long().tolist()).issubset({0, 1}), "class ids should be 0-indexed"

    def test_xywhn_normalised(self):
        out = self._make_output(n=2)
        result = self.mod.TorchvisionResult(out, img_shape=(480, 640, 3), conf_thresh=0.5)
        xywhn = result.xywhn[0]
        assert xywhn.shape == (2, 6)
        # Normalised coords must be in (0, 1]
        assert (xywhn[:, :4] > 0).all()
        assert (xywhn[:, :4] <= 1).all(), "normalised coords must be <= 1"

    def test_confidence_threshold_filtering(self):
        out = self._make_output(n=3)
        # Only scores 0.9, 0.8 are above 0.85
        result = self.mod.TorchvisionResult(out, img_shape=(480, 640, 3), conf_thresh=0.85)
        xyxy = result.xyxy[0]
        assert xyxy.shape[0] == 1, f"expected 1 detection above 0.85, got {xyxy.shape[0]}"

    def test_empty_output(self):
        out = {"boxes": torch.zeros((0, 4)), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.long)}
        result = self.mod.TorchvisionResult(out, img_shape=(480, 640, 3), conf_thresh=0.5)
        assert result.xyxy[0].shape == (0, 6)
        assert result.xywhn[0].shape == (0, 6)


# ---------------------------------------------------------------------------
# UltralyticsResult – zero detections edge case
# ---------------------------------------------------------------------------

class TestUltralyticsResultEmpty:
    """UltralyticsResult should return (0,6) tensors when there are no detections."""

    @pytest.fixture(autouse=True)
    def _mod(self):
        self.mod = _import_wrapper()

    def test_empty_boxes(self):
        fake_boxes = MagicMock()
        fake_boxes.__len__ = MagicMock(return_value=0)
        fake_result = MagicMock()
        fake_result.boxes = fake_boxes

        result = self.mod.UltralyticsResult(fake_result)
        assert result.xyxy[0].shape == (0, 6)
        assert result.xywhn[0].shape == (0, 6)

    def test_none_boxes(self):
        fake_result = MagicMock()
        fake_result.boxes = None

        result = self.mod.UltralyticsResult(fake_result)
        assert result.xyxy[0].shape == (0, 6)
        assert result.xywhn[0].shape == (0, 6)
