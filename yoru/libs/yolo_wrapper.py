"""Model wrapper providing a unified interface for all supported detection models.

Supported models:
  - YOLOv5              (via torch.hub)
  - YOLOv8, YOLO11      (via ultralytics.YOLO)
  - RT-DETR             (via ultralytics.RTDETR)
  - Faster R-CNN        (via torchvision, trained with train_torchvision.py)
  - Mask R-CNN          (via torchvision, trained with train_torchvision.py)
  - SSD                 (via torchvision, trained with train_torchvision.py)

Usage:
    from yoru.libs.yolo_wrapper import load_yolo_model

    model = load_yolo_model("path/to/weights.pt")
    # or explicitly:
    model = load_yolo_model("path/to/weights.pt", model_type="fasterrcnn")

    results = model(image)          # inference (image: numpy HxWxC BGR)
    results.xyxy[0]                 # [N x 6] tensor: x1,y1,x2,y2,conf,cls
    results.xywhn[0]                # [N x 6] tensor: x,y,w,h,conf,cls (normalized)
    model.names                     # dict {class_id: class_name}
"""

import os

import torch


def detect_model_type(model_path: str) -> str:
    """Infer model type from the model file name, with checkpoint inspection fallback.

    Returns one of: 'yolov5', 'yolov8', 'yolo11', 'rtdetr',
                    'fasterrcnn', 'maskrcnn', 'ssd'.
    Falls back to 'yolov5' for ambiguous names.
    """
    name = os.path.basename(model_path).lower()
    if "rtdetr" in name or "rt-detr" in name:
        return "rtdetr"
    if "fasterrcnn" in name or "faster_rcnn" in name or "faster-rcnn" in name:
        return "fasterrcnn"
    if "maskrcnn" in name or "mask_rcnn" in name or "mask-rcnn" in name:
        return "maskrcnn"
    if name.startswith("ssd") or "_ssd" in name:
        return "ssd"
    if "yolo11" in name or "yolov11" in name:
        return "yolo11"
    if "yolov8" in name or "yolo8" in name:
        return "yolov8"

    # For ambiguous names (e.g. "best.pt"), inspect the checkpoint to detect ultralytics models.
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model_obj = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else None
        if model_obj is not None:
            module = type(model_obj).__module__ or ""
            if "ultralytics" in module:
                return "yolov8"  # UltralyticsWrapper handles both YOLOv8 and YOLO11
    except Exception:
        pass

    return "yolov5"


def load_yolo_model(model_path: str, model_type: str = "auto"):
    """Factory function – returns the appropriate wrapper for the given weights.

    Args:
        model_path: Path to the model weights (.pt file).
        model_type: One of 'yolov5', 'yolov8', 'yolo11', 'rtdetr',
                    'fasterrcnn', 'maskrcnn', 'ssd', or 'auto'.
                    When 'auto', the type is inferred from the file name.

    Returns:
        YOLOv5Wrapper       – for YOLOv5 models
        UltralyticsWrapper  – for YOLOv8 / YOLO11 models
        RTDETRWrapper       – for RT-DETR models
        TorchvisionWrapper  – for Faster R-CNN / Mask R-CNN / SSD models
    """
    if model_type == "auto":
        model_type = detect_model_type(model_path)

    if model_type in ("fasterrcnn", "maskrcnn", "ssd"):
        return TorchvisionWrapper(model_path)
    elif model_type == "rtdetr":
        return RTDETRWrapper(model_path)
    elif model_type == "yolov5":
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


class TorchvisionResult:
    """Adapts a torchvision detection output dict to the YOLOv5 result API.

    Converts torchvision's ``{'boxes', 'labels', 'scores'}`` output into
    the unified ``[N x 6]`` format used throughout YORU.
    """

    def __init__(self, output: dict, img_shape: tuple, conf_thresh: float = 0.5):
        boxes  = output["boxes"]   # [N, 4] xyxy pixel coords
        scores = output["scores"]  # [N]
        labels = output["labels"]  # [N] 1-indexed

        mask   = scores > conf_thresh
        boxes  = boxes[mask]
        scores = scores[mask]
        labels = (labels[mask] - 1).float()  # convert to 0-indexed

        h, w = img_shape[:2]

        if boxes.shape[0] > 0:
            xyxy  = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1)], dim=1)
            xc    = (boxes[:, 0] + boxes[:, 2]) / 2 / w
            yc    = (boxes[:, 1] + boxes[:, 3]) / 2 / h
            bw    = (boxes[:, 2] - boxes[:, 0]) / w
            bh    = (boxes[:, 3] - boxes[:, 1]) / h
            xywhn = torch.stack([xc, yc, bw, bh, scores, labels], dim=1)
        else:
            xyxy  = torch.zeros((0, 6))
            xywhn = torch.zeros((0, 6))

        self._xyxy  = _ResultProxy(xyxy)
        self._xywhn = _ResultProxy(xywhn)

    @property
    def xyxy(self) -> _ResultProxy:
        return self._xyxy

    @property
    def xywhn(self) -> _ResultProxy:
        return self._xywhn


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


class RTDETRWrapper:
    """Wraps a RT-DETR model from the ``ultralytics`` package."""

    def __init__(self, model_path: str):
        from ultralytics import RTDETR
        self._model = RTDETR(model_path)

    @property
    def names(self) -> dict:
        return self._model.names

    def __call__(self, image):
        results = self._model(image, verbose=False)
        return UltralyticsResult(results[0])


class TorchvisionWrapper:
    """Wraps a Faster R-CNN, Mask R-CNN, or SSD model.

    Loads checkpoints saved by ``libs/train_torchvision.py``.
    The checkpoint must contain: model_state_dict, num_classes, names, model_type.
    """

    def __init__(self, model_path: str):
        checkpoint       = torch.load(model_path, map_location="cpu")
        self._model_type = checkpoint["model_type"]
        self._num_classes = checkpoint["num_classes"]
        self._names      = checkpoint["names"]
        self._device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model      = self._build_model()
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

    @property
    def names(self) -> dict:
        return self._names

    def _build_model(self):
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn,
            maskrcnn_resnet50_fpn,
            ssd300_vgg16,
        )
        from torchvision.models.detection._utils import retrieve_out_channels
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        from torchvision.models.detection.ssd import SSDHead

        n = self._num_classes + 1  # +1 for background

        if self._model_type == "fasterrcnn":
            model = fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)

        elif self._model_type == "maskrcnn":
            model = maskrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)
            in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
            model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, n)

        elif self._model_type == "ssd":
            model = ssd300_vgg16(weights=None)
            in_channels = retrieve_out_channels(model.backbone, (300, 300))
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head = SSDHead(in_channels, num_anchors, n)

        else:
            raise ValueError(f"Unknown model_type in checkpoint: {self._model_type!r}")

        return model

    def __call__(self, image):
        from torchvision.transforms.functional import to_tensor

        # Convert BGR numpy array → RGB tensor [C, H, W] in [0, 1]
        image_rgb = image[:, :, ::-1].copy()
        img_tensor = to_tensor(image_rgb).to(self._device)

        with torch.no_grad():
            outputs = self._model([img_tensor])

        return TorchvisionResult(
            {k: v.cpu() for k, v in outputs[0].items()},
            image.shape,
        )
