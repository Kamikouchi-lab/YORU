# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Torchvision detection plugin (Faster R-CNN / Mask R-CNN / SSD).

Loads checkpoints saved by ``yoru/libs/train_torchvision.py``.
"""

import torch
from torchvision.transforms.functional import to_tensor

from yoru.libs.detector_base import DetectorBase
from yoru.libs.plugins import DEFAULT_CONF_THRESH, register_detector


@register_detector("torchvision")
class TorchvisionDetector(DetectorBase):
    """Detection backend for torchvision Faster R-CNN / Mask R-CNN / SSD."""

    def load(self, model_path: str, **kwargs) -> None:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self._model_type = checkpoint["model_type"]
        self._num_classes = checkpoint["num_classes"]
        self._names: dict = checkpoint["names"]
        self._conf_thresh = float(kwargs.get("conf_thresh", DEFAULT_CONF_THRESH))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = self._build_model()
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

    @property
    def names(self) -> dict:
        return self._names

    def detect(self, image) -> list:
        image_rgb = image[:, :, ::-1].copy()
        img_tensor = to_tensor(image_rgb).to(self._device)

        with torch.no_grad():
            outputs = self._model([img_tensor])

        output = {k: v.cpu() for k, v in outputs[0].items()}
        boxes = output["boxes"]
        scores = output["scores"]
        labels = output["labels"]

        mask = scores > self._conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        detections = []
        for i in range(len(boxes)):
            cid = int(labels[i]) - 1  # torchvision uses 1-indexed labels
            detections.append(
                {
                    "x1": float(boxes[i, 0]),
                    "y1": float(boxes[i, 1]),
                    "x2": float(boxes[i, 2]),
                    "y2": float(boxes[i, 3]),
                    "conf": float(scores[i]),
                    "class_id": cid,
                    "class_name": self._names.get(cid, str(cid)),
                }
            )
        return detections

    # ------------------------------------------------------------------
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
            model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, 256, n
            )
        elif self._model_type == "ssd":
            model = ssd300_vgg16(weights=None)
            in_channels = retrieve_out_channels(model.backbone, (300, 300))
            num_anchors = model.anchor_generator.num_anchors_per_location()
            model.head = SSDHead(in_channels, num_anchors, n)
        else:
            raise ValueError(
                f"Unknown model_type in checkpoint: {self._model_type!r}"
            )

        return model
