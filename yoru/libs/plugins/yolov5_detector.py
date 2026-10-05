# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YOLOv5 detection plugin, backed by the vendored copy in ``yoru/libs/yolov5``.

This is upstream YOLOv5, not the YOLOv5u models that ship with ultralytics:
an anchor-based head whose raw output is ``[cx, cy, w, h, obj, cls...]``.  It
is what YORU v1 trained with, and a v1 checkpoint can be served by nothing
else -- ultralytics rejects those files outright, because the classes they
pickle live in YOLOv5's own ``models`` package rather than in ``ultralytics``.

Oriented boxes are out of scope here: YOLOv5 has no rotated-box head, so an
OBB project cannot offer this backend (see ``yoru/libs/init_train.py``).
"""

import numpy as np
import torch

from yoru.libs.detector_base import DetectorBase
from yoru.libs.plugins import (
    DEFAULT_CONF_THRESH,
    DEFAULT_IOU_THRESH,
    register_detector,
)
from yoru.libs.yolov5 import ensure_importable

#: Upper bound on detections kept per frame, matching YOLOv5's own default.
_MAX_DET = 1000


@register_detector("yolov5")
class YOLOv5Detector(DetectorBase):
    """Detection backend for upstream (non-"u") YOLOv5 checkpoints."""

    def load(self, model_path: str, **kwargs) -> None:
        # Binds the top-level 'models' and 'utils' the checkpoint's pickle
        # names.  Without it torch.load raises ModuleNotFoundError('models').
        ensure_importable()
        # attempt_load, not DetectMultiBackend: the latter reaches for
        # upstream's top-level export.py to sniff the file's format, and this
        # backend only ever serves PyTorch weights -- an exported model goes to
        # the 'onnx' backend instead.  attempt_load also downloads a bare
        # "yolov5s.pt" from upstream's releases, converts the half-precision
        # checkpoint to fp32, fuses and sets eval mode.
        from models.experimental import attempt_load

        self._conf_thresh = float(kwargs.get("conf_thresh", DEFAULT_CONF_THRESH))
        self._iou_thresh = float(kwargs.get("iou_thresh", DEFAULT_IOU_THRESH))
        self._imgsz = int(kwargs.get("imgsz", 640))

        device = kwargs.get("device")
        self._device = torch.device(
            device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        # Reading a v1 checkpoint means unpickling it, which runs code from the
        # file (attempt_load passes weights_only=False for exactly that
        # reason).  Only load weights you trained or otherwise trust.
        self._model = attempt_load(model_path, device=self._device, inplace=True, fuse=True)
        self._stride = max(int(self._model.stride.max()), 32)
        self._names = dict(self._model.names)

        # One warm-up pass: the first inference on CUDA otherwise pays for
        # cuDNN autotuning in the middle of a live recording.
        with torch.no_grad():
            self._model(
                torch.zeros(1, 3, self._imgsz, self._imgsz, device=self._device)
            )

    @property
    def names(self) -> dict:
        return self._names

    def detect(self, image) -> list:
        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

        # auto=False: pad to a fixed square, so every frame reaches the model
        # at the same shape and the warm-up above stays valid.
        padded = letterbox(image, self._imgsz, stride=self._stride, auto=False)[0]
        # BGR HWC -> RGB CHW, contiguous because the [::-1] view is not.
        tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[::-1])
        tensor = torch.from_numpy(tensor).to(self._device).float()[None] / 255.0

        with torch.no_grad():
            # The model returns (inference_out, training_out) in eval mode.
            pred = self._model(tensor)[0]

        det = non_max_suppression(
            pred,
            self._conf_thresh,
            self._iou_thresh,
            classes=None,
            agnostic=False,
            max_det=_MAX_DET,
        )[0]
        if not len(det):
            return []

        # Undo the letterbox: back to the coordinates of the frame handed in.
        det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], image.shape).round()
        det = det.cpu()

        detections = []
        for x1, y1, x2, y2, conf, cls in det.tolist():
            cid = int(cls)
            # No oriented keys: YOLOv5 predicts upright boxes only, and
            # yoru.libs.detector_base.obb_of derives the rest from x1..y2.
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
