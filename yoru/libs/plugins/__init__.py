# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Plugin registry for detection and training backends.

This module is part of YORU core and is NOT subject to any plugin's license.
"""

import importlib
import logging
import os
import re

from yoru.libs.detector_base import DetectorBase
from yoru.libs.trainer_base import TrainerBase

logger = logging.getLogger(__name__)

# Shared detection thresholds.  Every backend is given these explicitly so that
# the same model/video yields the same detections regardless of which plugin
# happens to serve it (previously each plugin applied its own default).
DEFAULT_CONF_THRESH = 0.25
DEFAULT_IOU_THRESH = 0.45

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_DETECTOR_REGISTRY: dict[str, type[DetectorBase]] = {}
_TRAINER_REGISTRY: dict[str, type[TrainerBase]] = {}
_plugins_loaded = False
_PLUGIN_IMPORT_ERRORS: dict[str, str] = {}


def register_detector(name: str):
    """Class decorator – register a DetectorBase subclass under *name*."""

    def _dec(cls):
        _DETECTOR_REGISTRY[name] = cls
        return cls

    return _dec


def register_trainer(name: str):
    """Class decorator – register a TrainerBase subclass under *name*."""

    def _dec(cls):
        _TRAINER_REGISTRY[name] = cls
        return cls

    return _dec


# ---------------------------------------------------------------------------
# Lazy plugin loading
# ---------------------------------------------------------------------------

_PLUGIN_MODULES = [
    "yoru.libs.plugins.onnx_detector",
    "yoru.libs.plugins.ultralytics_detector",
    "yoru.libs.plugins.torchvision_detector",
    "yoru.libs.plugins.yolov5_detector",
    "yoru.libs.plugins.ultralytics_trainer",
    "yoru.libs.plugins.torchvision_trainer",
    "yoru.libs.plugins.yolov5_trainer",
]


def _ensure_plugins_loaded():
    """Import every plugin module once so that @register_* decorators fire."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for mod in _PLUGIN_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as e:
            # Not only ImportError: a missing CUDA/onnxruntime DLL raises OSError.
            # Record the reason so get_detector() can explain *why* a backend is
            # missing instead of just reporting an unknown backend name.
            _PLUGIN_IMPORT_ERRORS[mod] = f"{type(e).__name__}: {e}"
            logger.warning("Backend plugin %s unavailable - %s: %s", mod, type(e).__name__, e)


# ---------------------------------------------------------------------------
# Backend name helpers
# ---------------------------------------------------------------------------

_BACKEND_ALIASES: dict[str, str] = {
    # "yolov5" is a backend of its own (plugins/yolov5_detector.py), not an
    # alias: ultralytics cannot read a YOLOv5 checkpoint.
    "yolov8": "ultralytics",
    "yolo11": "ultralytics",
    "fasterrcnn": "torchvision",
    "maskrcnn": "torchvision",
    "ssd": "torchvision",
}


def _normalize_backend(name: str) -> str:
    return _BACKEND_ALIASES.get(name, name)


#: ultralytics' YOLOv5u weights: a YOLOv5 backbone under YOLOv8's anchor-free
#: head, published as "yolov5su.pt", "yolov5x6u.pt" and so on.  The "u" sits
#: right after the size letter, which is all that separates them from the
#: upstream names -- and they are a different network, in ultralytics' format,
#: that the bundled YOLOv5 code cannot read.
_YOLOV5U_RE = re.compile(r"yolov?5[nsmlx]6?u")


def _auto_detect_backend(model_path: str) -> str:
    """Infer the detector backend from the model file."""
    basename = os.path.basename(model_path).lower()
    if basename.endswith(".onnx"):
        return "onnx"

    # A YOLOv5 checkpoint is recognised by its contents before anything is
    # read into its name: YORU v1 users named their models freely, and one
    # called e.g. "yolov8_compare.pt" must still reach the only backend that
    # can load it.  No other backend's checkpoint carries these markers.
    if _sniff_checkpoint(model_path) == "yolov5":
        return "yolov5"

    # Filename-based heuristics
    if "rtdetr" in basename or "rt-detr" in basename:
        return "rtdetr"
    if "fasterrcnn" in basename or "faster_rcnn" in basename or "faster-rcnn" in basename:
        return "torchvision"
    if "maskrcnn" in basename or "mask_rcnn" in basename or "mask-rcnn" in basename:
        return "torchvision"
    if basename.startswith("ssd") or "_ssd" in basename:
        return "torchvision"
    if "yolo11" in basename or "yolov11" in basename:
        return "ultralytics"
    if "yolov8" in basename or "yolo8" in basename:
        return "ultralytics"

    # For ambiguous names (e.g. "best.pt"), inspect the checkpoint contents.
    sniffed = _sniff_checkpoint(model_path)
    if sniffed is not None:
        return sniffed

    # A YOLOv5 name whose file cannot be read (typically an official weight
    # such as "yolov5s.pt" that is not downloaded yet): the bundled YOLOv5
    # fetches the real YOLOv5 release, where ultralytics would silently swap
    # in its different "yolov5su.pt".
    if "yolov5" in basename:
        # ...unless the name asks for YOLOv5u, which is ultralytics'.
        return "ultralytics" if _YOLOV5U_RE.search(basename) else "yolov5"

    return "ultralytics"


#: How a YOLOv5 checkpoint's pickle names the yolov5 repository's modules.
#: ``torch.save`` pickles with protocol 2, whose GLOBAL opcode writes
#: ``c<module>\n<name>\n``; protocol 4+ pushes the module name as a
#: SHORT_BINUNICODE string (``\x8c``, length byte, bytes).  Both spell the bare
#: top-level ``models.`` package, which no other backend's checkpoint has:
#: ultralytics' own ``ultralytics.models.yolo`` is never preceded by either.
_YOLOV5_PICKLE_MARKERS = (
    b"cmodels.yolo\n",
    b"cmodels.common\n",
    b"\x8c\x0bmodels.yolo",
    b"\x8c\x0dmodels.common",
)

#: Bytes read from the head of a pre-zip (torch < 1.6) checkpoint.  Its model
#: pickle, class names included, comes before the tensor data.
_LEGACY_HEAD_BYTES = 4 << 20


def _is_yolov5_pickle(blob: bytes) -> bool:
    return any(marker in blob for marker in _YOLOV5_PICKLE_MARKERS)


def _legacy_checkpoint_head(model_path: str):
    """The first bytes of a pre-zip checkpoint, or None if unreadable."""
    try:
        with open(model_path, "rb") as f:
            return f.read(_LEGACY_HEAD_BYTES)
    except OSError:
        return None


def _sniff_checkpoint(model_path: str):
    """Identify the backend from a ``.pt`` file without unpickling it.

    A torch ``.pt`` file is a ZIP archive whose ``data.pkl`` entry names the
    classes the checkpoint will construct.  Reading those names as plain bytes
    is enough to tell the backends apart, and avoids ``torch.load(...,
    weights_only=False)``, which would execute arbitrary code from the file and
    pull the entire model (hundreds of MB for e.g. rtdetr-x) into memory just to
    read one field.

    Returns the backend name, or ``None`` when the file cannot be identified.
    """
    import zipfile

    try:
        with zipfile.ZipFile(model_path) as zf:
            entry = next(
                (n for n in zf.namelist() if n.endswith("data.pkl")), None
            )
            if entry is None:
                return None
            blob = zf.read(entry)
    except (OSError, zipfile.BadZipFile):
        # Not a zip-format checkpoint (torch < 1.6) or unreadable.  YOLOv5
        # predates the zip format, so its oldest checkpoints are still worth
        # recognising; nothing else is identified from these.
        head = _legacy_checkpoint_head(model_path)
        if head is not None and _is_yolov5_pickle(head):
            return "yolov5"
        return None

    # First: a YOLOv5 checkpoint can mention "ultralytics" too (the git remote
    # of a clone of ultralytics/yolov5 is saved with every run).
    if _is_yolov5_pickle(blob):
        return "yolov5"
    # Checkpoints written by yoru/libs/train_torchvision.py carry a
    # "model_type" field holding one of these names.
    if b"model_type" in blob:
        for marker in (b"fasterrcnn", b"maskrcnn", b"ssd"):
            if marker in blob:
                return "torchvision"
    if b"ultralytics" in blob:
        return "ultralytics"
    if b"torchvision" in blob:
        return "torchvision"
    return None


# ---------------------------------------------------------------------------
# Public API – detectors
# ---------------------------------------------------------------------------


def get_detector(
    backend: str,
    model_path: str,
    conf_thresh: float = DEFAULT_CONF_THRESH,
    iou_thresh: float = DEFAULT_IOU_THRESH,
    **kwargs,
) -> DetectorBase:
    """Instantiate, load, and return a detector plugin.

    Args:
        backend: One of ``'ultralytics'``, ``'rtdetr'``,
                 ``'torchvision'``, ``'onnx'``, or ``'auto'``.
        model_path: Path to model weights.
        conf_thresh: Confidence threshold applied by every backend.
        iou_thresh: NMS IoU threshold applied by every backend.
        **kwargs: Forwarded to ``DetectorBase.load()``.
    """
    _ensure_plugins_loaded()

    if backend == "auto":
        backend = _auto_detect_backend(model_path)
    else:
        backend = _normalize_backend(backend)

    if backend not in _DETECTOR_REGISTRY:
        msg = (
            f"Unknown detector backend: {backend!r}. "
            f"Available: {sorted(_DETECTOR_REGISTRY)}"
        )
        if _PLUGIN_IMPORT_ERRORS:
            details = "; ".join(
                f"{m} ({why})" for m, why in sorted(_PLUGIN_IMPORT_ERRORS.items())
            )
            msg += f". Some backends failed to load: {details}"
        raise ValueError(msg)

    detector = _DETECTOR_REGISTRY[backend]()
    detector.load(
        model_path, conf_thresh=conf_thresh, iou_thresh=iou_thresh, **kwargs
    )
    return detector


# ---------------------------------------------------------------------------
# Public API – trainers
# ---------------------------------------------------------------------------


def get_trainer(backend: str) -> TrainerBase:
    """Return a trainer plugin instance for *backend*."""
    _ensure_plugins_loaded()
    backend = _normalize_backend(backend)

    if backend not in _TRAINER_REGISTRY:
        raise ValueError(
            f"Unknown trainer backend: {backend!r}. "
            f"Available: {sorted(_TRAINER_REGISTRY)}"
        )
    return _TRAINER_REGISTRY[backend]()


def detect_trainer_backend(m_dict: dict) -> str:
    """Determine the correct trainer backend from the GUI's *m_dict*."""
    family = m_dict.get("model_family", "YOLO")

    if family in ("Faster R-CNN", "Mask R-CNN", "SSD"):
        return "torchvision"
    if family == "RT-DETR":
        return "ultralytics"

    weight = os.path.basename(str(m_dict.get("weight", ""))).lower()
    if weight.startswith("yolov5"):
        # Trained with the bundled YOLOv5 (yoru/libs/yolov5), as in YORU v1 --
        # unless the name is one of ultralytics' YOLOv5u models.
        return "ultralytics" if _YOLOV5U_RE.match(weight) else "yolov5"
    if any(tag in weight for tag in ("yolov8", "yolo8", "yolo11", "yolov11")):
        return "ultralytics"

    return "ultralytics"
