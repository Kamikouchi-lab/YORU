# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Access to the bundled YOLOv5 code in ``yoru/libs/yolov5``.

YORU v1 trained and ran its models with this copy of ultralytics/yolov5 (the
one v1.1.2 shipped), and a checkpoint written by it can only be read back with
the same code: the pickle inside names its classes ``models.yolo.DetectionModel``,
``models.common.Conv``, ... -- top-level packages of the yolov5 repository,
not of the ``ultralytics`` package.  Ultralytics refuses such a file ("NOT
forwards compatible"), and its ``yolov5*u`` models are a different network,
so neither is a substitute for this copy.

This module is the one place that knows where the copy lives and how to make
its ``models`` / ``utils`` packages importable, for the detector plugin
(``plugins/yolov5_detector.py``) and the training script (``train_yolov5.py``)
alike.  It imports nothing heavy, so it is safe to import anywhere.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

__all__ = [
    "YOLOV5_DIR",
    "require_bundled_yolov5",
    "yolov5_importable",
]

#: The bundled ultralytics/yolov5 repository.
YOLOV5_DIR = Path(__file__).resolve().parent / "yolov5"

#: Top-level packages of the yolov5 repository.  Its checkpoints, and its own
#: code, import them by these bare names.
_TOP_LEVEL_PACKAGES = ("models", "utils")


def require_bundled_yolov5() -> Path:
    """The bundled repository's directory; raises if it is not there."""
    if not (YOLOV5_DIR / "models" / "yolo.py").is_file():
        raise FileNotFoundError(
            f"The bundled YOLOv5 code is missing from {YOLOV5_DIR}. "
            "YOLOv5 models (including every model trained with YORU v1) are "
            "loaded and trained with it; restore yoru/libs/yolov5 from the "
            "YORU repository."
        )
    return YOLOV5_DIR


def _is_inside(location, root: Path) -> bool:
    try:
        path = Path(location).resolve()
    except (OSError, TypeError, ValueError):
        return False
    return path == root or root in path.parents


def _foreign_top_level_modules() -> list:
    """``models`` / ``utils`` modules already imported from somewhere else.

    Python caches a module by name, so once some other ``utils`` has been
    imported, yolov5's ``from utils.general import ...`` would silently get
    that one instead.  Better to say so than to fail somewhere inside yolov5.
    """
    foreign = []
    for name in _TOP_LEVEL_PACKAGES:
        module = sys.modules.get(name)
        if module is None:
            continue
        locations = list(getattr(module, "__path__", None) or [])
        if not locations and getattr(module, "__file__", None):
            locations = [module.__file__]
        if not any(_is_inside(loc, YOLOV5_DIR) for loc in locations):
            foreign.append(f"'{name}' from {locations[0] if locations else 'an unknown location'}")
    return foreign


@contextlib.contextmanager
def yolov5_importable():
    """Put the bundled repository on ``sys.path`` for the duration of the block.

    What ``torch.hub.load(yolov5_dir, ..., source="local")`` did for YORU v1:
    the directory goes first on the path while the model is imported and
    unpickled, and comes off again afterwards.  The modules it brought in stay
    cached in ``sys.modules`` -- the model needs them to run -- so later
    imports of them do not need the path any more.
    """
    root = require_bundled_yolov5()
    foreign = _foreign_top_level_modules()
    if foreign:
        raise ImportError(
            "Cannot load the bundled YOLOv5 code: this process has already "
            f"imported {', '.join(foreign)}, and YOLOv5 needs those names for "
            "its own packages. Load the YOLOv5 model in a fresh process."
        )
    entry = str(root)
    sys.path.insert(0, entry)
    try:
        yield root
    finally:
        try:
            sys.path.remove(entry)
        except ValueError:
            pass
