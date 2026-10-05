# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YOLOv5 training plugin, run by the bundled ultralytics/yolov5 code.

Requires the bundled ``yoru/libs/yolov5``; see ``libs/train_yolov5.py``.
"""

import subprocess
import sys
from pathlib import Path

from yoru.libs.plugins import register_trainer
from yoru.libs.trainer_base import TrainerBase

# Resolve relative to this package, not the current working directory.
_TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "train_yolov5.py"


@register_trainer("yolov5")
class YOLOv5Trainer(TrainerBase):
    """Launch YOLOv5 training as a subprocess, as YORU v1 did."""

    # yolov5's train.py prints the first of 300 epochs as "0/299".
    epoch_base = 0

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
        if config.get("stop_file"):
            # Lets the GUI end the run cleanly at an epoch boundary;
            # see libs/train_stop.py.
            cmd += ["--stop-file", str(config["stop_file"])]

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
