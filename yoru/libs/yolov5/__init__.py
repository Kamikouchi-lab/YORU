# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Vendored YOLOv5 (https://github.com/ultralytics/yolov5), AGPL-3.0.

Everything below this directory is upstream YOLOv5, carried here rather than
installed, for two reasons: a checkpoint trained with YORU v1 can only be
unpickled by the exact model classes that wrote it, and pinning those classes
to a pip release would make an old project's weights depend on what happens to
be installed.  See THIRD_PARTY_LICENSES.md.

Upstream imports itself as top-level ``models`` and ``utils``, and a legacy
checkpoint pickles class paths in that same namespace (``models.yolo
.DetectionModel``, ``models.common.Conv``, ...).  Unpickling therefore needs
those two names importable, which is why :func:`ensure_importable` exists.

v1 arranged that by handing the directory to ``torch.hub.load(..., source=
"local")``, which *prepends* it to ``sys.path`` -- shadowing every later
``import utils`` in the process, YORU's own and its dependencies' alike.  Here
the two packages are instead loaded straight from this directory and bound to
the names the pickle asks for, so nothing that already resolves starts
resolving somewhere else.

Upstream's own modules still append this directory to ``sys.path`` when they
are imported (``models/common.py`` and friends do it in their header, which is
how their ``import val`` resolves).  An append cannot shadow anything, and by
then ``models`` and ``utils`` are bound to this copy regardless of what else
the path picks up -- which is the guarantee that matters for unpickling.
"""

import importlib.util
import sys
import threading
from pathlib import Path

__all__ = ["ensure_importable", "YOLOV5_DIR"]

#: Root of the vendored checkout; also where ``models/*.yaml`` live.
YOLOV5_DIR = Path(__file__).resolve().parent

# "utils" first: models/*.py import from it while they execute.
_TOP_LEVEL_PACKAGES = ("utils", "models")

# Marks a sys.modules entry as one of ours, so a second call is a no-op and a
# name taken by somebody else is not silently overwritten.
_MARKER = "__yoru_vendored_yolov5__"

_lock = threading.Lock()


def _bind(name: str) -> None:
    """Bind ``yoru/libs/yolov5/<name>/`` to the top-level module *name*."""
    existing = sys.modules.get(name)
    if existing is not None:
        if getattr(existing, _MARKER, False):
            return  # already ours
        raise RuntimeError(
            f"cannot load YOLOv5: the name {name!r} is already taken by "
            f"{getattr(existing, '__file__', existing)!r}. A legacy YOLOv5 "
            f"checkpoint can only be unpickled while 'models' and 'utils' "
            f"refer to the vendored copy, so YORU will not overwrite them."
        )

    package = YOLOV5_DIR / name
    spec = importlib.util.spec_from_file_location(
        name, package / "__init__.py", submodule_search_locations=[str(package)]
    )
    if spec is None or spec.loader is None:  # pragma: no cover - broken checkout
        raise ImportError(f"vendored YOLOv5 is incomplete: {package} not importable")

    module = importlib.util.module_from_spec(spec)
    setattr(module, _MARKER, True)
    # Registered before it is executed: the package body imports its own
    # submodules by the top-level name, and they resolve through this entry.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise


def ensure_importable() -> Path:
    """Make ``models`` and ``utils`` importable, and return :data:`YOLOV5_DIR`.

    Idempotent and thread-safe.  Call it before importing anything from the
    vendored tree and before ``torch.load`` on a legacy checkpoint; callers
    that only use YOLOv8/YOLO11 never call it, so the two names stay free.
    """
    with _lock:
        for name in _TOP_LEVEL_PACKAGES:
            _bind(name)
    return YOLOV5_DIR
