# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Ultralytics (YOLOv8 / YOLO11 / RT-DETR) detection plugin.

Requires: ``pip install ultralytics``
"""

import torch

from yoru.libs.detector_base import DetectorBase
from yoru.libs.device import resolve_device
from yoru.libs.plugins import (
    DEFAULT_CONF_THRESH,
    DEFAULT_IOU_THRESH,
    register_detector,
)


class _UltralyticsDetectorBase(DetectorBase):
    """Shared implementation for all ultralytics-based detectors."""

    _model_cls_name: str = "YOLO"  # overridden in subclasses

    def load(self, model_path: str, **kwargs) -> None:
        import ultralytics

        cls = getattr(ultralytics, self._model_cls_name)
        self._model = cls(model_path)
        self._names: dict = dict(self._model.names)
        self._conf_thresh = float(kwargs.get("conf_thresh", DEFAULT_CONF_THRESH))
        self._iou_thresh = float(kwargs.get("iou_thresh", DEFAULT_IOU_THRESH))
        # ultralytics' own selection falls back CUDA -> CPU and never reaches
        # MPS, so Apple Silicon needs the device named on every predict call.
        self._device = resolve_device(kwargs.get("device", "auto"))

    @property
    def names(self) -> dict:
        return self._names

    def detect(self, image) -> list:
        results = self._model(
            image,
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            device=self._device,
            verbose=False,
        )
        result = results[0]

        # An OBB model puts its predictions in .obb and leaves .boxes empty, so
        # which attribute is populated is itself the reliable test for the
        # task -- more so than the weight's file name, which a user can rename.
        obb = getattr(result, "obb", None)
        if obb is not None and len(obb) > 0:
            return self._obb_detections(obb)

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu()
        conf = boxes.conf.cpu()
        cls = boxes.cls.cpu()

        detections = []
        for i in range(len(boxes)):
            cid = int(cls[i])
            # No oriented keys here on purpose: an upright box's are derivable
            # from x1..y2, and yoru.libs.detector_base.obb_of derives them in
            # one place rather than in each of the four detector plugins.
            detections.append(
                {
                    "x1": float(xyxy[i, 0]),
                    "y1": float(xyxy[i, 1]),
                    "x2": float(xyxy[i, 2]),
                    "y2": float(xyxy[i, 3]),
                    "conf": float(conf[i]),
                    "class_id": cid,
                    "class_name": self._names.get(cid, str(cid)),
                }
            )
        return detections

    def _obb_detections(self, obb) -> list:
        """Rotated predictions, carrying both parameterisations.

        ``x1..y2`` is the upright box around the rotated one, so everything
        written before OBB support -- the trigger plugins, the evaluation IoU --
        keeps working unchanged; ``cx, cy, w, h, angle`` is the box itself, for
        everything that wants the real thing.
        """
        xywhr = obb.xywhr.cpu()
        xyxy = obb.xyxy.cpu()
        conf = obb.conf.cpu()
        cls = obb.cls.cpu()

        detections = []
        for i in range(len(obb)):
            cid = int(cls[i])
            detections.append(
                {
                    "x1": float(xyxy[i, 0]),
                    "y1": float(xyxy[i, 1]),
                    "x2": float(xyxy[i, 2]),
                    "y2": float(xyxy[i, 3]),
                    "cx": float(xywhr[i, 0]),
                    "cy": float(xywhr[i, 1]),
                    "w": float(xywhr[i, 2]),
                    "h": float(xywhr[i, 3]),
                    # Radians, the same convention as yoru.libs.obb.
                    "angle": float(xywhr[i, 4]),
                    "conf": float(conf[i]),
                    "class_id": cid,
                    "class_name": self._names.get(cid, str(cid)),
                }
            )
        return detections


@register_detector("ultralytics")
class UltralyticsDetector(_UltralyticsDetectorBase):
    """YOLOv8 / YOLO11 detector via the ``ultralytics`` package."""

    _model_cls_name = "YOLO"


@register_detector("rtdetr")
class RTDETRDetector(_UltralyticsDetectorBase):
    """RT-DETR detector via the ``ultralytics`` package."""

    _model_cls_name = "RTDETR"
