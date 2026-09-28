# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YOLOv5 detection plugin, run by the bundled ultralytics/yolov5 code.

Serves models trained by YORU v1 -- and any checkpoint of the ultralytics/yolov5
repository -- the way YORU v1 ran them, so that a v1 model gives the same boxes
here that it gave there:

* The model is YOLOv5's ``AutoShape(DetectMultiBackend(weights, fuse=True))``,
  which is what v1's ``torch.hub.load(yolov5_dir, "custom", path=...)``
  returned.  It is built directly rather than through ``torch.hub``, whose
  hubconf would also run yolov5's ``select_device()``: that sets
  ``CUDA_VISIBLE_DEVICES`` for the whole process -- and every process it
  starts -- as soon as a device is named.
* The frame is handed over unchanged.  v1 gave AutoShape the BGR frames OpenCV
  delivers, and AutoShape takes a numpy array as it is, so v1's models have
  always been run on BGR frames -- although yolov5 trains on RGB.  Converting
  to RGB would change their detections, so it is opt-in: set
  ``YORU_YOLOV5_RGB=1`` (or pass ``rgb_input=True`` to ``get_detector``).
  Which order a model was run with is written to the YORU log.
* Everything else is the bundled AutoShape's, as in v1: 640 px letterbox, at
  most 1000 boxes, and CUDA autocast (fp16) around inference -- a YORU v1
  change to ``models/common.py`` that this copy keeps.  The conf / IoU
  thresholds are the shared 0.25 / 0.45, which are AutoShape's defaults too.

This is not ultralytics' ``yolov5*u`` family, which is a different network
(anchor-free, YOLOv8 head) and cannot load these checkpoints.

Requires the bundled ``yoru/libs/yolov5`` and the packages in its
requirements.txt, all of which are YORU dependencies already.
"""

import logging
import os
from pathlib import Path

from yoru.libs.detector_base import DetectorBase
from yoru.libs.device import torch_device
from yoru.libs.plugins import (
    DEFAULT_CONF_THRESH,
    DEFAULT_IOU_THRESH,
    _sniff_checkpoint,
    register_detector,
)
from yoru.libs.user_paths import log_message
from yoru.libs.yolov5_support import yolov5_importable

#: Environment variable that switches YOLOv5 models to RGB frames.  An
#: environment variable rather than a per-GUI setting, like YORU_DEVICE, so
#: that realtime detection, analysis, evaluation and auto-labelling all run a
#: model the same way.
RGB_ENV_VAR = "YORU_YOLOV5_RGB"

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("", "0", "false", "no", "off")


def rgb_input_requested(value=None) -> bool:
    """Whether YOLOv5 models get RGB frames rather than v1's BGR ones.

    *value* (``get_detector(..., rgb_input=...)``) wins when given; otherwise
    ``$YORU_YOLOV5_RGB`` decides.  Anything unrecognised keeps v1's BGR, with
    a warning, since a typo must not silently change every detection.
    """
    if value is None:
        value = os.environ.get(RGB_ENV_VAR, "")
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text not in _FALSE:
        log_message(
            f"{RGB_ENV_VAR}={value!r} is not understood (use 1 or 0); "
            "YOLOv5 models get BGR frames, as in YORU v1",
            logging.WARNING,
        )
    return False


def load_yolov5_model(model_path, device):
    """YOLOv5's AutoShape model for *model_path*, on *device*.

    Mirrors the path yolov5's hubconf ``custom()`` takes for a detection
    checkpoint.  Classification and segmentation checkpoints are refused:
    AutoShape cannot run them, and YORU only detects.
    """
    path = Path(model_path)
    if path.suffix == "" and not path.is_dir():
        path = path.with_suffix(".pt")

    # An explicit "yolov5" model type on, say, a YOLO11 checkpoint would
    # otherwise load and then decode its outputs as garbage boxes.
    kind = _sniff_checkpoint(str(path))
    if kind not in (None, "yolov5"):
        raise ValueError(
            f"{path.name} is not a YOLOv5 checkpoint (it looks like a {kind} "
            "one). Set the model type to 'auto' to load it with the right "
            "backend."
        )

    with yolov5_importable():
        from models.common import AutoShape, DetectMultiBackend
        from models.yolo import ClassificationModel, SegmentationModel

        backend = DetectMultiBackend(str(path), device=device, fuse=True)

    if backend.pt and isinstance(backend.model, (ClassificationModel, SegmentationModel)):
        raise ValueError(
            f"{path.name} is a YOLOv5 {type(backend.model).__name__}; YORU "
            "runs YOLOv5 detection models only."
        )
    return AutoShape(backend).to(device)


@register_detector("yolov5")
class YOLOv5Detector(DetectorBase):
    """YOLOv5 detector via the bundled ultralytics/yolov5 code."""

    def load(self, model_path: str, **kwargs) -> None:
        # YORU_DEVICE and the device selectors are honoured as for the other
        # backends; "auto" is CUDA:0 when there is one, as v1 used.
        self._device = torch_device(kwargs.get("device", "auto"))
        self._model = load_yolov5_model(model_path, self._device)
        self._model.conf = float(kwargs.get("conf_thresh", DEFAULT_CONF_THRESH))
        self._model.iou = float(kwargs.get("iou_thresh", DEFAULT_IOU_THRESH))

        names = self._model.names
        if isinstance(names, (list, tuple)):
            # Checkpoints from older yolov5 releases store a plain list.
            names = dict(enumerate(names))
        self._names: dict = {int(k): str(v) for k, v in names.items()}

        self._rgb = rgb_input_requested(kwargs.get("rgb_input"))
        order = "RGB frames" if self._rgb else "BGR frames, as in YORU v1"
        # Recorded, so that results can be traced back to how they were made.
        print(f"[yoru] YOLOv5 model {Path(model_path).name}: {order}")
        log_message(f"YOLOv5 model {model_path}: {order}")

    @property
    def names(self) -> dict:
        return self._names

    def detect(self, image) -> list:
        # Unchanged BGR, as v1 passed it, unless RGB was asked for (see the
        # module docstring).  A grey frame has no order to change.
        if self._rgb and image.ndim == 3 and image.shape[2] >= 3:
            image = image[..., 2::-1]  # BGR(A) -> RGB; AutoShape makes it contiguous
        results = self._model(image)
        pred = results.xyxy[0].cpu()

        detections = []
        for x1, y1, x2, y2, conf, cls in pred.tolist():
            cid = int(cls)
            # Upright boxes only: YOLOv5 has no rotated-box head, and
            # yoru.libs.detector_base.obb_of derives the oriented columns.
            detections.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "conf": float(conf),
                    "class_id": cid,
                    "class_name": self._names.get(cid, str(cid)),
                }
            )
        return detections
