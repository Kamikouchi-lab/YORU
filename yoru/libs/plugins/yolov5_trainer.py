# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YOLOv5 training plugin, backed by the vendored copy in ``yoru/libs/yolov5``.

Upstream YOLOv5, not the YOLOv5u models ultralytics ships: the anchor-based
model YORU v1 trained with.  Oriented boxes are not available here -- YOLOv5
has no rotated-box head.
"""

import subprocess
import sys
from pathlib import Path

from yoru.libs.plugins import register_trainer
from yoru.libs.trainer_base import TrainerBase

# Resolve relative to this package, not the current working directory.
_TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "train_yolov5.py"

# Upstream's train.py re-expresses its own directory relative to the working
# directory, which raises on Windows when the two are on different drives.
# Running from the repository root keeps that relative path computable
# wherever the user's project directory happens to live.
_REPO_ROOT = Path(__file__).resolve().parents[3]


@register_trainer("yolov5")
class YOLOv5Trainer(TrainerBase):
    """Launch YOLOv5 training as a subprocess."""

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
            cwd=str(_REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
