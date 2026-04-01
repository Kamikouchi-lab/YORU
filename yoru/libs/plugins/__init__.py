# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Plugin registry for detection and training backends.

This module is part of YORU core and is NOT subject to any plugin's license.
"""

import importlib
import os

from yoru.libs.detector_base import DetectorBase
from yoru.libs.trainer_base import TrainerBase

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

_DETECTOR_REGISTRY: dict[str, type[DetectorBase]] = {}
_TRAINER_REGISTRY: dict[str, type[TrainerBase]] = {}
_plugins_loaded = False


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
    "yoru.libs.plugins.ultralytics_trainer",
    "yoru.libs.plugins.torchvision_trainer",
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
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Backend name helpers
# ---------------------------------------------------------------------------

_BACKEND_ALIASES: dict[str, str] = {
    "yolov5": "ultralytics",
    "yolov8": "ultralytics",
    "yolo11": "ultralytics",
    "fasterrcnn": "torchvision",
    "maskrcnn": "torchvision",
    "ssd": "torchvision",
}


def _normalize_backend(name: str) -> str:
    return _BACKEND_ALIASES.get(name, name)


def _auto_detect_backend(model_path: str) -> str:
    """Infer the detector backend from the model file."""
    basename = os.path.basename(model_path).lower()
    if basename.endswith(".onnx"):
        return "onnx"

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
    try:
        import torch

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "model_type" in ckpt:
                return _normalize_backend(ckpt["model_type"])
            model_obj = ckpt.get("ema") or ckpt.get("model")
            if model_obj is not None:
                module = type(model_obj).__module__ or ""
                if "ultralytics" in module:
                    return "ultralytics"
    except Exception:
        pass

    return "ultralytics"


# ---------------------------------------------------------------------------
# Public API – detectors
# ---------------------------------------------------------------------------


def get_detector(backend: str, model_path: str, **kwargs) -> DetectorBase:
    """Instantiate, load, and return a detector plugin.

    Args:
        backend: One of ``'ultralytics'``, ``'rtdetr'``,
                 ``'torchvision'``, ``'onnx'``, or ``'auto'``.
        model_path: Path to model weights.
        **kwargs: Forwarded to ``DetectorBase.load()``.
    """
    _ensure_plugins_loaded()

    if backend == "auto":
        backend = _auto_detect_backend(model_path)
    else:
        backend = _normalize_backend(backend)

    if backend not in _DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown detector backend: {backend!r}. "
            f"Available: {sorted(_DETECTOR_REGISTRY)}"
        )

    detector = _DETECTOR_REGISTRY[backend]()
    detector.load(model_path, **kwargs)
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

    weight = m_dict.get("weight", "").lower()
    if any(tag in weight for tag in ("yolov8", "yolo8", "yolo11", "yolov11")):
        return "ultralytics"

    return "ultralytics"
