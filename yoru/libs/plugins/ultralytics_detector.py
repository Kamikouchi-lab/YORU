# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Ultralytics (YOLOv8 / YOLO11 / RT-DETR) detection plugin.

Requires: ``pip install ultralytics``
"""

import torch

from yoru.libs.detector_base import DetectorBase
from yoru.libs.plugins import register_detector


class _UltralyticsDetectorBase(DetectorBase):
    """Shared implementation for all ultralytics-based detectors."""

    _model_cls_name: str = "YOLO"  # overridden in subclasses

    def load(self, model_path: str, **kwargs) -> None:
        import ultralytics

        cls = getattr(ultralytics, self._model_cls_name)
        self._model = cls(model_path)
        self._names: dict = dict(self._model.names)

    @property
    def names(self) -> dict:
        return self._names

    def detect(self, image) -> list:
        results = self._model(image, verbose=False)
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu()
        conf = boxes.conf.cpu()
        cls = boxes.cls.cpu()

        detections = []
        for i in range(len(boxes)):
            cid = int(cls[i])
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


@register_detector("ultralytics")
class UltralyticsDetector(_UltralyticsDetectorBase):
    """YOLOv8 / YOLO11 detector via the ``ultralytics`` package."""

    _model_cls_name = "YOLO"


@register_detector("rtdetr")
class RTDETRDetector(_UltralyticsDetectorBase):
    """RT-DETR detector via the ``ultralytics`` package."""

    _model_cls_name = "RTDETR"
