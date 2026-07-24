"""ONNX Runtime detection plugin.

SPDX-License-Identifier: MIT

This file uses ONNX Runtime (MIT license).
"""

import ast

import cv2
import numpy as np

from yoru.libs.detector_base import DetectorBase
from yoru.libs.plugins import register_detector


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
        self._conf_thresh = kwargs.get("conf_thresh", 0.25)
        self._iou_thresh = kwargs.get("iou_thresh", 0.45)

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

        # Pre-process: resize, BGR -> RGB, normalise, HWC -> CHW, add batch
        img = cv2.resize(image, (inp_w, inp_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]

        outputs = self._session.run(None, {self._input_name: blob})
        pred = outputs[0]
        return self._postprocess(pred, w0, h0, inp_w, inp_h)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _postprocess(self, pred, orig_w, orig_h, inp_w, inp_h) -> list:
        pred = np.squeeze(pred, axis=0)  # [N, C] or [C, N]

        # YOLOv8 exports: shape [num_features, num_boxes] -> transpose
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        num_cols = pred.shape[1]
        nc = len(self._names) if self._names else max(num_cols - 5, num_cols - 4, 0)

        # Determine format:
        #   YOLOv5: [cx, cy, w, h, obj_conf, cls0, cls1, ...]  (5 + nc)
        #   YOLOv8: [cx, cy, w, h, cls0, cls1, ...]            (4 + nc)
        has_obj = (num_cols - 5 == nc) if nc > 0 else (num_cols == 85)

        if has_obj:
            obj_conf = pred[:, 4:5]
            cls_scores = pred[:, 5:] * obj_conf
        else:
            cls_scores = pred[:, 4:]

        class_ids = cls_scores.argmax(axis=1)
        confidences = cls_scores[np.arange(len(cls_scores)), class_ids]

        # Confidence filter
        mask = confidences > self._conf_thresh
        if not mask.any():
            return []

        pred = pred[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Convert cx,cy,w,h -> x1,y1,x2,y2 in input-image scale
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Scale to original image dimensions
        scale_x, scale_y = orig_w / inp_w, orig_h / inp_h
        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        # NMS (cv2.dnn expects [x, y, w, h])
        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, confidences.tolist(), self._conf_thresh, self._iou_thresh
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
