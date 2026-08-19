# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Torchvision (Faster R-CNN / Mask R-CNN / SSD) training plugin."""

import subprocess
import sys
from pathlib import Path

from yoru.libs.plugins import register_trainer
from yoru.libs.trainer_base import TrainerBase

# Resolve relative to this package, not the current working directory.
_TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "train_torchvision.py"

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
            sys.executable,
            str(_TRAIN_SCRIPT),
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

        if config.get("stop_file"):
            # Lets the GUI end the run cleanly at an epoch boundary;
            # see libs/train_stop.py.
            cmd += ["--stop-file", str(config["stop_file"])]

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
