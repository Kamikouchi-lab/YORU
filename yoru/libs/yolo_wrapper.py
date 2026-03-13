"""YOLO model wrapper providing a unified interface for YOLOv5, YOLOv8, and YOLO11.

Usage:
    from yoru.libs.yolo_wrapper import load_yolo_model

    model = load_yolo_model("path/to/weights.pt")
    # or explicitly:
    model = load_yolo_model("path/to/weights.pt", model_type="yolov8")

    results = model(image)          # inference
    results.xyxy[0]                 # [N x 6] tensor: x1,y1,x2,y2,conf,cls
    results.xywhn[0]                # [N x 6] tensor: x,y,w,h,conf,cls (normalized)
    model.names                     # dict {class_id: class_name}
"""

import os

import torch


def detect_model_type(model_path: str) -> str:
    """Infer model type from the model file name.

    Returns 'yolov5', 'yolov8', or 'yolo11'.
    Falls back to 'yolov5' when the name is ambiguous (e.g. custom-named files).
    """
    name = os.path.basename(model_path).lower()
    if "yolo11" in name or "yolov11" in name:
        return "yolo11"
    if "yolov8" in name or "yolo8" in name:
        return "yolov8"
    return "yolov5"


def load_yolo_model(model_path: str, model_type: str = "auto"):
    """Factory function – returns the appropriate wrapper for the given weights.

    Args:
        model_path: Path to the model weights (.pt file).
        model_type: One of 'yolov5', 'yolov8', 'yolo11', or 'auto'.
                    When 'auto', the type is inferred from the file name.

    Returns:
        YOLOv5Wrapper  – for YOLOv5 models
        UltralyticsWrapper – for YOLOv8 / YOLO11 models
    """
    if model_type == "auto":
        model_type = detect_model_type(model_path)

    if model_type == "yolov5":
        return YOLOv5Wrapper(model_path)
    else:
        return UltralyticsWrapper(model_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ResultProxy:
    """Minimal list-like wrapper so that result.xyxy[0] works."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._tensor


class UltralyticsResult:
    """Adapts an ultralytics ``Results`` object to the YOLOv5 result API.

    Provides ``.xyxy`` and ``.xywhn`` attributes whose ``[0]`` element is a
    ``[N x 6]`` tensor of ``[x1, y1, x2, y2, conf, cls]`` or
    ``[x, y, w, h, conf, cls]`` (normalised), matching YOLOv5's output format.
    """

    def __init__(self, result):
        self._result = result
        self._xyxy_cache = None
        self._xywhn_cache = None

    def _build_6col(self, coords: torch.Tensor) -> torch.Tensor:
        """Concatenate [N,4] coords with [N,1] conf and [N,1] cls → [N,6]."""
        boxes = self._result.boxes
        conf = boxes.conf.unsqueeze(1)
        cls = boxes.cls.unsqueeze(1)
        return torch.cat([coords, conf, cls], dim=1)

    @property
    def xyxy(self) -> _ResultProxy:
        if self._xyxy_cache is None:
            boxes = self._result.boxes
            if boxes is None or len(boxes) == 0:
                tensor = torch.zeros((0, 6))
            else:
                tensor = self._build_6col(boxes.xyxy)
            self._xyxy_cache = _ResultProxy(tensor)
        return self._xyxy_cache

    @property
    def xywhn(self) -> _ResultProxy:
        if self._xywhn_cache is None:
            boxes = self._result.boxes
            if boxes is None or len(boxes) == 0:
                tensor = torch.zeros((0, 6))
            else:
                tensor = self._build_6col(boxes.xywhn)
            self._xywhn_cache = _ResultProxy(tensor)
        return self._xywhn_cache


# ---------------------------------------------------------------------------
# Public wrappers
# ---------------------------------------------------------------------------

class YOLOv5Wrapper:
    """Wraps a YOLOv5 model loaded via ``torch.hub``."""

    def __init__(self, model_path: str):
        self._model = torch.hub.load(
            "./libs/yolov5", "custom", path=model_path, source="local"
        )

    @property
    def names(self) -> dict:
        return (
            self._model.module.names
            if hasattr(self._model, "module")
            else self._model.names
        )

    def __call__(self, image):
        return self._model(image)


class UltralyticsWrapper:
    """Wraps a YOLOv8 or YOLO11 model from the ``ultralytics`` package."""

    def __init__(self, model_path: str):
        from ultralytics import YOLO
        self._model = YOLO(model_path)

    @property
    def names(self) -> dict:
        return self._model.names

    def __call__(self, image):
        results = self._model(image, verbose=False)
        return UltralyticsResult(results[0])
