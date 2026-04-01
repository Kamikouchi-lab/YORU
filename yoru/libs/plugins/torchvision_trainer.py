# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Torchvision (Faster R-CNN / Mask R-CNN / SSD) training plugin."""

import subprocess

from yoru.libs.plugins import register_trainer
from yoru.libs.trainer_base import TrainerBase

_FAMILY_TO_MODEL = {
    "Faster R-CNN": "fasterrcnn",
    "Mask R-CNN": "maskrcnn",
    "SSD": "ssd",
}


@register_trainer("torchvision")
class TorchvisionTrainer(TrainerBase):
    """Launch Faster R-CNN / Mask R-CNN / SSD training as a subprocess."""

    def train(self, config: dict) -> subprocess.Popen:
        model_type = _FAMILY_TO_MODEL.get(
            config.get("model_family", "Faster R-CNN"), "fasterrcnn"
        )

        cmd = [
            "python",
            "./yoru/libs/train_torchvision.py",
            "--model",
            model_type,
            "--data",
            str(config["data_yaml"]),
            "--epochs",
            str(config["epochs"]),
            "--batch",
            str(config["batch_size"]),
            "--project",
            str(config["project_dir"]),
        ]

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
