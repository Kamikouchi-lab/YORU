"""ONNX Runtime detection plugin.

SPDX-License-Identifier: MIT

This file uses ONNX Runtime (MIT license).
"""

import ast
import logging

import cv2
import numpy as np

from yoru.libs.detector_base import DetectorBase
from yoru.libs.plugins import (
    DEFAULT_CONF_THRESH,
    DEFAULT_IOU_THRESH,
    register_detector,
)

logger = logging.getLogger(__name__)


@register_detector("onnx")
class ONNXDetector(DetectorBase):
    """Detection backend using ONNX Runtime.

    Supports ONNX models exported from YOLOv5 and YOLOv8/YOLO11.
    """

    def load(self, model_path: str, **kwargs) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            model_path,
            providers=ort.get_available_providers(),
        )
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self._input_shape = inp.shape  # e.g. [1, 3, 640, 640]
        self._conf_thresh = float(kwargs.get("conf_thresh", DEFAULT_CONF_THRESH))
        self._iou_thresh = float(kwargs.get("iou_thresh", DEFAULT_IOU_THRESH))

        # Try to read class names from model metadata
        self._names: dict = kwargs.get("names", {})
        if not self._names:
            meta = self._session.get_modelmeta().custom_metadata_map
            if "names" in meta:
                try:
                    self._names = ast.literal_eval(meta["names"])
                except Exception:
                    pass

    @property
    def names(self) -> dict:
        return self._names

    def detect(self, image) -> list:
        h0, w0 = image.shape[:2]
        inp_h, inp_w = int(self._input_shape[2]), int(self._input_shape[3])

        # Letterbox rather than a plain resize: the model was trained on
        # aspect-preserving padded images, so stretching the frame costs
        # accuracy (and shifts every box).
        img, ratio, (pad_w, pad_h) = self._letterbox(image, (inp_h, inp_w))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(
            (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        )

        pred = self._session.run(None, {self._input_name: blob})[0]
        return self._postprocess(pred, w0, h0, ratio, pad_w, pad_h, inp_h, inp_w)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _letterbox(image, new_shape, color=(114, 114, 114)):
        """Resize *image* preserving aspect ratio and pad it to *new_shape*.

        Returns ``(padded_image, ratio, (pad_w, pad_h))`` where the pads are the
        border applied to one side, so a detection can be mapped back with
        ``(coord - pad) / ratio``.
        """
        h0, w0 = image.shape[:2]
        new_h, new_w = new_shape
        ratio = min(new_h / h0, new_w / w0)
        unpad_w, unpad_h = int(round(w0 * ratio)), int(round(h0 * ratio))

        if (w0, h0) != (unpad_w, unpad_h):
            image = cv2.resize(
                image, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR
            )

        pad_w = (new_w - unpad_w) / 2
        pad_h = (new_h - unpad_h) / 2
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return image, ratio, (pad_w, pad_h)

    def _has_objectness(self, n_rows: int, num_cols: int, inp_h: int, inp_w: int) -> bool:
        """Whether the raw output carries a YOLOv5-style objectness column.

        YOLOv5:    ``[cx, cy, w, h, obj, cls0, ...]``  -> ``5 + nc`` columns
        YOLOv8/11: ``[cx, cy, w, h, cls0, ...]``       -> ``4 + nc`` columns

        When the export embeds class names the column count settles it. Without
        them, fall back on the row count: YOLOv5 emits three anchor boxes per
        feature-map cell where YOLOv8/11 emit one.
        """
        if self._names:
            nc = len(self._names)
            if num_cols == nc + 5:
                return True
            if num_cols == nc + 4:
                return False
            logger.warning(
                "ONNX output has %d columns, matching neither %d (YOLOv8/11) nor "
                "%d (YOLOv5) for %d classes; assuming the YOLOv8/11 layout.",
                num_cols, nc + 4, nc + 5, nc,
            )
            return False

        cells = sum((inp_h // s) * (inp_w // s) for s in (8, 16, 32))
        return n_rows > cells * 1.5

    def _postprocess(
        self, pred, orig_w, orig_h, ratio, pad_w, pad_h, inp_h, inp_w
    ) -> list:
        pred = np.squeeze(pred, axis=0)  # [N, C] or [C, N]

        # YOLOv8 exports: shape [num_features, num_boxes] -> transpose
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        n_rows, num_cols = pred.shape
        if n_rows == 0 or num_cols < 5:
            return []

        if self._has_objectness(n_rows, num_cols, inp_h, inp_w):
            cls_scores = pred[:, 5:] * pred[:, 4:5]
        else:
            cls_scores = pred[:, 4:]
        if cls_scores.shape[1] == 0:
            return []

        class_ids = cls_scores.argmax(axis=1)
        confidences = cls_scores[np.arange(len(cls_scores)), class_ids]

        # Confidence filter
        mask = confidences > self._conf_thresh
        if not mask.any():
            return []

        pred = pred[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # cx,cy,w,h in the letterboxed frame -> x1,y1,x2,y2 in the original image
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = (cx - bw / 2 - pad_w) / ratio
        y1 = (cy - bh / 2 - pad_h) / ratio
        x2 = (cx + bw / 2 - pad_w) / ratio
        y2 = (cy + bh / 2 - pad_h) / ratio
        x1 = np.clip(x1, 0, orig_w)
        x2 = np.clip(x2, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        y2 = np.clip(y2, 0, orig_h)

        # Per-class NMS. cv2.dnn.NMSBoxes is class-agnostic, so shifting each
        # class into its own coordinate band stops a box of one class from
        # suppressing an overlapping box of a different class.
        offset = class_ids.astype(np.float32) * (max(orig_w, orig_h) + 1.0)
        boxes_nms = np.stack(
            [x1 + offset, y1 + offset, x2 - x1, y2 - y1], axis=1
        ).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_nms, confidences.tolist(), self._conf_thresh, self._iou_thresh
        )
        if len(indices) == 0:
            return []
        indices = np.array(indices).flatten()

        detections = []
        for i in indices:
            cid = int(class_ids[i])
            detections.append(
                {
                    "x1": float(x1[i]),
                    "y1": float(y1[i]),
                    "x2": float(x2[i]),
                    "y2": float(y2[i]),
                    "conf": float(confidences[i]),
                    "class_id": cid,
                    "class_name": self._names.get(cid, str(cid)),
                }
            )
        return detections
