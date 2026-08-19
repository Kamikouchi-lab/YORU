# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Predict the GPU memory a training run will need, before it starts.

Training dies with a CUDA out-of-memory error minutes -- sometimes hours --
after the user presses "Train Model", and the only way to find a batch size
that fits has been to try one and wait.  This module estimates the peak VRAM
from the settings that are already on screen (model, image size, batch) plus
what can be read cheaply off the dataset, and compares it with the memory the
card actually has free.

The estimate is a calibrated approximation, not a simulation::

    peak = (weights_and_optimizer + activations + assigner_workspace) * frag
           + cuda_context

* ``weights_and_optimizer`` follows from the parameter count and how many
  copies of each parameter the optimizer keeps (see ``_Profile.bytes_per_param``).
* ``activations`` is the dominant term.  It is linear in batch size and
  quadratic in image size, so one measured GB-per-image constant per model
  (``_Profile.act_gb``, at 640 px) pins the whole curve.
* ``assigner_workspace`` covers the ``(batch, max_labels, anchors)`` tensors
  YOLO's task-aligned assigner builds; it is negligible for the usual handful
  of animals per frame and dominant for crowded frames, which is exactly the
  case a user cannot predict by intuition.
* ``frag`` is the headroom the caching allocator takes for fragmentation, and
  ``cuda_context`` is the ~0.8 GB the CUDA runtime holds outside the PyTorch
  allocator -- invisible in the ``GPU_mem`` column ultralytics prints, but
  very much taken on the card.

``act_gb`` is the one number that has to be measured rather than derived.  The
values below were calibrated against ultralytics ``GPU_mem`` readings at
imgsz 640 with AMP enabled; treat the result as +/-30% and recalibrate a row
whenever a real run disagrees with it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = [
    "GPUInfo",
    "DatasetStats",
    "VramEstimate",
    "VramVerdict",
    "get_gpu_info",
    "read_dataset_stats",
    "estimate_training_vram",
    "largest_batch_that_fits",
    "check_training_vram",
]

# Memory the CUDA runtime, cuDNN and cuBLAS hold outside the PyTorch caching
# allocator.  Never appears in ultralytics' GPU_mem column (that is
# torch.cuda.memory_reserved) but is charged to the card all the same.
CUDA_CONTEXT_GB = 0.8

# The caching allocator's reserved pool creeps above the live-tensor peak as
# variable-shaped batches fragment it -- the effect that makes GPU_mem climb
# epoch over epoch.  ~10% covers where it settles.
FRAGMENTATION = 1.10

# Fraction of free VRAM above which a run is reported as "tight" rather than
# "ok".  Leaves room for the estimate being on the optimistic side.
TIGHT_FRACTION = 0.85

# YOLO's task-aligned assigner holds roughly this many (batch, labels, anchors)
# fp32 tensors live at once (overlaps, alignment metrics, masks, gathers).
_ASSIGNER_TENSORS = 8

# Anchor points a 640 px YOLO head produces (80^2 + 40^2 + 20^2).
_ANCHORS_640 = 8400

# Label files read when sampling a dataset for its instance counts.  Reading
# every file in a 100k-image project would stall the GUI for no extra accuracy.
_LABEL_SAMPLE = 400

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class _Profile:
    """Per-model constants the estimate is built from.

    Attributes:
        params_m (float): trainable parameters, in millions.
        act_gb (float): activation memory per image at 640 px, in GB.
        bytes_per_param (int): bytes held per parameter across weights,
            gradients, optimizer state and the EMA copy.  20 for the
            ultralytics path (fp32 weights + grads + two AdamW moments + EMA),
            12 for the torchvision path (fp32 weights + grads + SGD momentum,
            no EMA).
        anchors (int): assigner grid points at 640 px, or 0 for models whose
            loss is not anchor-based (RT-DETR matches a fixed query set, the
            torchvision detectors size their own proposals).
        scales_with_imgsz (bool): whether the GUI's Image Size drives the
            activation size.  False for the torchvision models, which resize
            internally via GeneralizedRCNNTransform and ignore the setting.
    """

    params_m: float
    act_gb: float
    bytes_per_param: int
    anchors: int
    scales_with_imgsz: bool = True


# Calibrated at imgsz 640, AMP on, against ultralytics GPU_mem readings.
_PROFILES: dict[str, _Profile] = {
    # --- YOLOv8 -----------------------------------------------------------
    "yolov8n": _Profile(3.2, 0.15, 20, _ANCHORS_640),
    "yolov8s": _Profile(11.2, 0.27, 20, _ANCHORS_640),
    "yolov8m": _Profile(25.9, 0.47, 20, _ANCHORS_640),
    "yolov8l": _Profile(43.7, 0.63, 20, _ANCHORS_640),
    "yolov8x": _Profile(68.2, 0.95, 20, _ANCHORS_640),
    # --- YOLO11 -----------------------------------------------------------
    "yolo11n": _Profile(2.6, 0.14, 20, _ANCHORS_640),
    "yolo11s": _Profile(9.4, 0.25, 20, _ANCHORS_640),
    "yolo11m": _Profile(20.1, 0.45, 20, _ANCHORS_640),
    "yolo11l": _Profile(25.3, 0.55, 20, _ANCHORS_640),
    "yolo11x": _Profile(56.9, 0.92, 20, _ANCHORS_640),
    # --- RT-DETR ----------------------------------------------------------
    # Far heavier per image than a YOLO of the same parameter count: the
    # decoder carries ~300 queries plus up to ~200 denoising queries, and the
    # loss is computed again at every one of the 6 decoder layers.
    "rtdetr-l": _Profile(32.0, 0.97, 20, 0),
    "rtdetr-x": _Profile(67.0, 1.50, 20, 0),
    # --- torchvision ------------------------------------------------------
    # fp32 throughout (train_torchvision.py enables no AMP), so the per-image
    # cost is high relative to the parameter count.
    "fasterrcnn": _Profile(41.8, 0.75, 12, 0, scales_with_imgsz=False),
    "maskrcnn": _Profile(44.4, 0.95, 12, 0, scales_with_imgsz=False),
    "ssd": _Profile(35.6, 0.40, 12, 0, scales_with_imgsz=False),
}

# Longest keys first so "yolov8n" is not shadowed by a shorter prefix.
_PROFILE_KEYS = sorted(_PROFILES, key=len, reverse=True)


def profile_for(weight: str) -> _Profile | None:
    """Look up the profile for a weight file name.

    Args:
        weight (str): weight file name, e.g. ``"rtdetr-l.pt"``.

    Returns:
        _Profile | None: the matching profile, or None for a name no row
        covers (a user-supplied checkpoint, say), in which case no estimate
        can be given.
    """
    stem = Path(str(weight)).stem.lower()
    for key in _PROFILE_KEYS:
        if stem.startswith(key):
            return _PROFILES[key]
    return None


# ---------------------------------------------------------------------------
# GPU discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPUInfo:
    """A CUDA device and how much of its memory is spoken for.

    Attributes:
        name (str): device name as the driver reports it.
        total_gb (float): total VRAM.
        used_gb (float): VRAM already in use by every process on the card.
        source (str): where the numbers came from, ``"nvidia-smi"`` or
            ``"torch"``.
    """

    name: str
    total_gb: float
    used_gb: float
    source: str

    @property
    def free_gb(self) -> float:
        return max(0.0, self.total_gb - self.used_gb)


def _nvidia_smi_path() -> str | None:
    """Locate nvidia-smi, including the spots it hides in on Windows."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in (
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def _no_window_flags() -> int:
    """CREATE_NO_WINDOW, so the GUI does not blink a console on every poll."""
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def get_gpu_info(index: int = 0) -> GPUInfo | None:
    """Report the training GPU without initialising CUDA in this process.

    nvidia-smi is tried first on purpose: importing torch costs seconds, and
    ``torch.cuda.get_device_properties`` creates a CUDA context worth a few
    hundred MB in the *GUI* process -- memory that then is not there for the
    training subprocess this function exists to protect.

    Args:
        index (int, optional): CUDA device index. Defaults to 0.

    Returns:
        GPUInfo | None: the device, or None when there is no usable NVIDIA GPU.
    """
    smi = _nvidia_smi_path()
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=6, check=True,
                creationflags=_no_window_flags(),
            ).stdout
            rows = [r for r in out.splitlines() if r.strip()]
            if index < len(rows):
                name, total_mib, used_mib = (
                    c.strip() for c in rows[index].split(",")[:3]
                )
                return GPUInfo(
                    name=name,
                    total_gb=float(total_mib) / 1024.0,
                    used_gb=float(used_mib) / 1024.0,
                    source="nvidia-smi",
                )
        except Exception:
            pass  # fall through to torch

    # torch is imported lazily: it may be absent, and it is slow when present.
    try:
        import torch

        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            return None
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 2**30
        try:
            free_b, _ = torch.cuda.mem_get_info(index)
            used_gb = total_gb - free_b / 2**30
        except Exception:
            used_gb = 0.0
        return GPUInfo(props.name, total_gb, used_gb, "torch")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetStats:
    """What the dataset contributes to the memory estimate.

    Attributes:
        n_train (int): training images found.
        n_val (int): validation images found.
        n_classes (int): classes declared in the dataset YAML.
        max_instances (int): most labelled objects on any sampled image; this
            is what sizes the assigner and matching workspaces.
        sampled_labels (int): label files actually read.
    """

    n_train: int = 0
    n_val: int = 0
    n_classes: int = 0
    max_instances: int = 0
    sampled_labels: int = 0


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1 for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )


def _split_dirs(root: Path, entry) -> tuple[Path, Path]:
    """Resolve one YAML split entry to its ``(images, labels)`` directories.

    ``create_yaml_train`` writes ``train: <project>/train/``, i.e. the split
    root rather than the image directory, but a YAML written by hand may point
    straight at ``.../train/images``.  Both are accepted.
    """
    path = Path(str(entry))
    if not path.is_absolute():
        path = root / path
    if path.name == "images":
        return path, path.parent / "labels"
    return path / "images", path / "labels"


def read_dataset_stats(data_yaml) -> DatasetStats:
    """Read image counts and label density out of a dataset YAML.

    Never raises: a missing or malformed YAML simply yields zeroed stats, and
    the estimate falls back to model-only terms.

    Args:
        data_yaml (str | PathLike | None): path to the dataset YAML.

    Returns:
        DatasetStats: what could be determined.
    """
    if not data_yaml:
        return DatasetStats()
    yaml_path = Path(str(data_yaml))
    if not yaml_path.is_file():
        return DatasetStats()
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return DatasetStats()
    if not isinstance(data, dict):
        return DatasetStats()

    root = Path(str(data.get("path") or yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    train_images, train_labels = _split_dirs(root, data.get("train", "train"))
    val_images, _ = _split_dirs(root, data.get("val", "val"))

    n_classes = data.get("nc")
    if not isinstance(n_classes, int):
        names = data.get("names")
        n_classes = len(names) if isinstance(names, (list, dict)) else 0

    max_instances = 0
    sampled = 0
    if train_labels.is_dir():
        for label in sorted(train_labels.glob("*.txt")):
            if label.name == "classes.txt":
                continue
            try:
                lines = label.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
            except OSError:
                continue
            max_instances = max(
                max_instances, sum(1 for ln in lines if ln.strip())
            )
            sampled += 1
            if sampled >= _LABEL_SAMPLE:
                break

    return DatasetStats(
        n_train=_count_images(train_images),
        n_val=_count_images(val_images),
        n_classes=int(n_classes or 0),
        max_instances=max_instances,
        sampled_labels=sampled,
    )


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VramEstimate:
    """Peak VRAM a run is predicted to hold, broken into its terms.

    Attributes:
        total_gb (float): what the card is expected to be asked for.
        model_gb (float): weights, gradients, optimizer state, EMA.
        activation_gb (float): forward activations kept for the backward pass.
        assigner_gb (float): label-assignment workspace.
        overhead_gb (float): CUDA context plus the fragmentation allowance.
        notes (tuple[str, ...]): caveats worth showing next to the number.
    """

    total_gb: float
    model_gb: float
    activation_gb: float
    assigner_gb: float
    overhead_gb: float
    notes: tuple = ()


def estimate_training_vram(
    weight: str,
    imgsz: int = 640,
    batch: int = 16,
    stats: DatasetStats | None = None,
) -> VramEstimate | None:
    """Estimate peak VRAM for one training configuration.

    Args:
        weight (str): weight file name, e.g. ``"yolo11s.pt"``.
        imgsz (int, optional): training image size. Defaults to 640.
        batch (int, optional): images per batch. Defaults to 16.
        stats (DatasetStats, optional): dataset stats from
            :func:`read_dataset_stats`.  Without them the assigner term is
            dropped, which under-estimates crowded datasets.

    Returns:
        VramEstimate | None: the estimate, or None if *weight* matches no
        known model.
    """
    prof = profile_for(weight)
    if prof is None:
        return None

    batch = max(1, int(batch))
    imgsz = max(32, int(imgsz))
    notes = []

    if prof.scales_with_imgsz:
        scale = (imgsz / 640.0) ** 2
    else:
        scale = 1.0
        notes.append(
            "torchvision models resize internally (min 800 px); the Image Size "
            "setting does not change their memory use."
        )

    model_gb = prof.params_m * 1e6 * prof.bytes_per_param / 2**30
    activation_gb = prof.act_gb * batch * scale

    assigner_gb = 0.0
    if prof.anchors and stats and stats.max_instances:
        anchors = prof.anchors * scale
        assigner_gb = (
            _ASSIGNER_TENSORS * batch * stats.max_instances * anchors * 4 / 2**30
        )
        if assigner_gb > 0.5:
            notes.append(
                f"{stats.max_instances} objects on the busiest training image "
                f"add ~{assigner_gb:.1f} GB of label-assignment workspace."
            )

    live = model_gb + activation_gb + assigner_gb
    overhead_gb = live * (FRAGMENTATION - 1.0) + CUDA_CONTEXT_GB
    return VramEstimate(
        total_gb=live + overhead_gb,
        model_gb=model_gb,
        activation_gb=activation_gb,
        assigner_gb=assigner_gb,
        overhead_gb=overhead_gb,
        notes=tuple(notes),
    )


def largest_batch_that_fits(
    weight: str,
    imgsz: int,
    budget_gb: float,
    stats: DatasetStats | None = None,
    max_batch: int = 512,
) -> int:
    """Largest batch size whose estimate stays inside *budget_gb*.

    Args:
        weight (str): weight file name.
        imgsz (int): training image size.
        budget_gb (float): memory the run may use.
        stats (DatasetStats, optional): dataset stats.
        max_batch (int, optional): search ceiling. Defaults to 512.

    Returns:
        int: the largest batch that fits, or 0 if even a batch of 1 does not.
    """
    best = 0
    for batch in range(1, int(max_batch) + 1):
        est = estimate_training_vram(weight, imgsz, batch, stats)
        if est is None or est.total_gb > budget_gb:
            break
        best = batch
    return best


@dataclass(frozen=True)
class VramVerdict:
    """The estimate turned into something the GUI can colour and print.

    Attributes:
        level (str): ``"ok"``, ``"tight"``, ``"over"`` or ``"unknown"``.
        headline (str): one-line summary, e.g. ``"14.2 GB / 11.0 GB free"``.
        detail (str): the reason and what to do about it.
        estimate (VramEstimate | None): the underlying estimate.
        gpu (GPUInfo | None): the device it was compared against.
        suggested_batch (int | None): a batch that would fit, when the current
            one does not.
    """

    level: str
    headline: str
    detail: str
    estimate: VramEstimate | None = None
    gpu: GPUInfo | None = None
    suggested_batch: int | None = None

    @property
    def is_warning(self) -> bool:
        return self.level in ("tight", "over")


def check_training_vram(
    weight: str,
    imgsz: int,
    batch: int,
    stats: DatasetStats | None = None,
    gpu: GPUInfo | None = None,
) -> VramVerdict:
    """Estimate the run's memory and judge it against the available GPU.

    Args:
        weight (str): weight file name.
        imgsz (int): training image size.
        batch (int): images per batch.
        stats (DatasetStats, optional): dataset stats.
        gpu (GPUInfo, optional): the device; queried via :func:`get_gpu_info`
            when omitted.  Pass a cached value to keep GUI callbacks cheap.

    Returns:
        VramVerdict: level, headline and advice.
    """
    est = estimate_training_vram(weight, imgsz, batch, stats)
    if est is None:
        return VramVerdict(
            "unknown",
            "not estimated",
            f"No memory profile for '{weight}'.",
        )

    if gpu is None:
        gpu = get_gpu_info()
    if gpu is None:
        return VramVerdict(
            "unknown",
            f"~{est.total_gb:.1f} GB needed",
            "No NVIDIA GPU detected, so there is nothing to compare against. "
            "Training on CPU will use system RAM instead, and will be far "
            "slower.",
            estimate=est,
        )

    free = gpu.free_gb
    headline = f"~{est.total_gb:.1f} GB needed / {free:.1f} GB free ({gpu.name})"
    breakdown = (
        f"model {est.model_gb:.1f} + activations {est.activation_gb:.1f}"
        + (f" + assigner {est.assigner_gb:.1f}" if est.assigner_gb >= 0.05 else "")
        + f" + overhead {est.overhead_gb:.1f} GB."
    )
    notes = (" " + " ".join(est.notes)) if est.notes else ""

    if est.total_gb > free:
        fit = largest_batch_that_fits(weight, imgsz, free, stats)
        if fit >= 1:
            advice = f"Try Batch {fit} or smaller, or reduce Image Size."
        else:
            advice = (
                "Even a batch of 1 does not fit; choose a smaller model or a "
                "smaller Image Size."
            )
        used_note = ""
        if gpu.used_gb > 0.5:
            used_note = (
                f" {gpu.used_gb:.1f} GB of this card is already in use by other "
                "processes."
            )
        return VramVerdict(
            "over",
            headline,
            f"Likely to run out of GPU memory. {advice} {breakdown}"
            f"{used_note}{notes}",
            estimate=est,
            gpu=gpu,
            suggested_batch=fit or None,
        )

    if est.total_gb > free * TIGHT_FRACTION:
        return VramVerdict(
            "tight",
            headline,
            "This should fit, but with little headroom -- the estimate is only "
            "accurate to about +/-30%, and reserved memory creeps up over the "
            f"first epochs. {breakdown}{notes}",
            estimate=est,
            gpu=gpu,
            suggested_batch=largest_batch_that_fits(
                weight, imgsz, free * TIGHT_FRACTION, stats
            ) or None,
        )

    return VramVerdict("ok", headline, breakdown + notes, estimate=est, gpu=gpu)
