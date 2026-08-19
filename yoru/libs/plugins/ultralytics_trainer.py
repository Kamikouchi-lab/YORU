# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Ultralytics (YOLOv8 / YOLO11 / RT-DETR) training plugin.

Requires: ``pip install ultralytics``
"""

import subprocess
import sys
from pathlib import Path

from yoru.libs.plugins import register_trainer
from yoru.libs.trainer_base import TrainerBase

# Resolve relative to this package, not the current working directory.
_TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "train_ultralytics.py"


@register_trainer("ultralytics")
class UltralyticsTrainer(TrainerBase):
    """Launch YOLOv8 / YOLO11 / RT-DETR training as a subprocess."""

    def train(self, config: dict) -> subprocess.Popen:
        cmd = [
            sys.executable,
            str(_TRAIN_SCRIPT),
            "--weights",
            str(config["weights"]),
            "--data",
            str(config["data_yaml"]),
            "--epochs",
            str(config["epochs"]),
            "--imgsz",
            str(config["img_size"]),
            "--batch",
            str(config["batch_size"]),
            "--project",
            str(config["project_dir"]),
        ]
        if config.get("device"):
            cmd += ["--device", str(config["device"])]

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
