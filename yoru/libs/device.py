"""Compute-device selection for YORU.

Centralises the choice between CUDA, Apple MPS and CPU so that every training
and inference entry point agrees, and so that a user can override it.

Resolution order for ``preference="auto"``:

1. the ``YORU_DEVICE`` environment variable, when set,
2. CUDA, when ``torch.cuda.is_available()``,
3. Apple MPS, when ``torch.backends.mps.is_available()``,
4. CPU.

Note on MPS: ultralytics never auto-selects it (``select_device("")`` falls
back CUDA -> CPU), so Apple Silicon users silently trained on CPU unless the
device was named explicitly. This module makes ``"mps"`` explicit for them.

Like ``user_paths``, this module has no GUI/OpenCV dependency, so it is safe to
import early and to unit-test headlessly.
"""

from __future__ import annotations

import logging
import os

from yoru.libs.user_paths import log_message

#: Values accepted by :func:`resolve_device` in addition to CUDA indices.
KNOWN_DEVICES = ("auto", "cuda", "mps", "cpu")

_ENV_VAR = "YORU_DEVICE"


def cuda_available() -> bool:
    """``True`` when a usable CUDA device is present."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def mps_available() -> bool:
    """``True`` when Apple Metal Performance Shaders are usable."""
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def resolve_device(preference: str = "auto") -> str:
    """Resolve *preference* to a device string usable by torch and ultralytics.

    Args:
        preference: ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``, or a CUDA
            index such as ``"0"`` or ``"0,1"``. ``None`` and ``""`` mean
            ``"auto"``.

    Returns:
        ``"cuda"``, ``"mps"``, ``"cpu"``, or the requested CUDA index. An
        unavailable request degrades to the next best device and is logged, so
        a saved condition file naming ``cuda`` still runs on a laptop.
    """
    pref = str(preference or "auto").strip().lower()

    if pref == "auto":
        env = os.environ.get(_ENV_VAR, "").strip().lower()
        if env:
            # An explicit environment override is still validated below.
            return resolve_device(env)
        if cuda_available():
            return "cuda"
        if mps_available():
            return "mps"
        return "cpu"

    if pref == "cpu":
        return "cpu"

    if pref == "mps":
        if mps_available():
            return "mps"
        log_message("device 'mps' requested but unavailable; using cpu", logging.WARNING)
        return "cpu"

    if pref == "cuda" or pref.replace(",", "").isdigit():
        if cuda_available():
            return "cuda" if pref == "cuda" else pref
        fallback = "mps" if mps_available() else "cpu"
        log_message(
            f"device {pref!r} requested but CUDA is unavailable; using {fallback}",
            logging.WARNING,
        )
        return fallback

    log_message(f"unknown device {preference!r}; falling back to auto", logging.WARNING)
    return resolve_device("auto")


def torch_device(preference: str = "auto"):
    """Same resolution as :func:`resolve_device`, as a ``torch.device``."""
    import torch

    resolved = resolve_device(preference)
    # torch.device wants an index-free name or 'cuda:N'; ultralytics wants '0'.
    if resolved not in ("cuda", "mps", "cpu"):
        resolved = f"cuda:{resolved.split(',')[0]}"
    return torch.device(resolved)


def describe(preference: str = "auto") -> str:
    """Human-readable one-liner for logs and GUI status text."""
    resolved = resolve_device(preference)
    if resolved == "cpu":
        return "CPU"
    if resolved == "mps":
        return "Apple MPS"
    try:
        import torch

        index = 0 if resolved == "cuda" else int(resolved.split(",")[0])
        return f"CUDA:{index} ({torch.cuda.get_device_name(index)})"
    except Exception:
        return f"CUDA ({resolved})"
