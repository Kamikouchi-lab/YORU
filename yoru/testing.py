"""Programmatic entry points used by the smoke tests in ``tests/``.

Both entries drive YORU's own code paths -- ``create_yaml_train`` for the
dataset YAML, ``libs/train_ultralytics.py`` for training and
``load_yolo_model`` for detection -- on a deliberately tiny configuration, so
a full train/detect round trip still fits in CI on CPU. Nothing is written
outside ``out_dir`` and the YORU user directory.
"""

import csv
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

from yoru.libs.create_yaml_train import create_project
from yoru.libs.device import resolve_device
from yoru.libs.user_paths import get_yoru_home, log_exception, log_message
from yoru.libs.yolo_wrapper import load_yolo_model

# Smoke-sized training settings: an 80/20 split of a few dozen images at a
# small image size, so the real training loop, dataloader and validation pass
# all run without turning CI into a GPU job.
SMOKE_IMG_SIZE = 320
SMOKE_BATCH = 4
SMOKE_TRAIN_IMAGES = 64
SMOKE_VAL_IMAGES = 16

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
_TRAIN_SCRIPT = Path(__file__).resolve().parent / "libs" / "train_ultralytics.py"


def _list_images(directory: Path) -> list:
    """Return the image files directly inside *directory*, sorted by name."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


def _pick(images: list, count: int) -> list:
    """Take *count* images spread evenly over *images*, always the same ones."""
    if len(images) <= count:
        return list(images)
    return images[:: len(images) // count][:count]


def _label_path(image: Path) -> Path:
    """Return the YOLO label file of *image*: beside it, or under ``labels/``."""
    beside = image.with_suffix(".txt")
    if beside.exists():
        return beside
    return image.parent.parent / "labels" / (image.stem + ".txt")


def _stage_split(images: list, dest_dir: Path) -> int:
    """Place *images* and their label files side by side in *dest_dir*.

    Returns the number of image/label pairs staged. Symlinks are used so that
    a large dataset is not copied; hosts that refuse symlinks fall back to a
    plain copy.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged = 0
    for image in images:
        label = _label_path(image)
        if not label.exists():
            continue
        for src in (image, label):
            dst = dest_dir / src.name
            if dst.exists():
                continue
            try:
                os.symlink(src, dst)
            except OSError:
                # Windows refuses symlinks unless developer mode is enabled.
                shutil.copyfile(src, dst)
        staged += 1
    return staged


def _stage_dataset(data_dir: Path, dataset_dir: Path) -> tuple:
    """Build a small train/val dataset under *dataset_dir* from *data_dir*.

    *data_dir* may already be split into ``train/`` and ``val/`` (the layout
    the training GUI asks for), or hold images and YOLO label files side by
    side in a single directory, in which case the split is made here.
    """
    train_src = _list_images(data_dir / "train")
    val_src = _list_images(data_dir / "val")

    if train_src:
        train_src = _pick(train_src, SMOKE_TRAIN_IMAGES)
        val_src = _pick(val_src, SMOKE_VAL_IMAGES)
    else:
        images = _list_images(data_dir)
        if not images:
            raise FileNotFoundError(f"no images found in {data_dir}")
        selected = _pick(images, SMOKE_TRAIN_IMAGES + SMOKE_VAL_IMAGES)
        train_src = selected[:SMOKE_TRAIN_IMAGES]
        val_src = selected[SMOKE_TRAIN_IMAGES:]

    if not val_src:
        # Training still needs something to validate against.
        val_src = train_src[:1]

    train_count = _stage_split(train_src, dataset_dir / "train")
    val_count = _stage_split(val_src, dataset_dir / "val")
    if not train_count or not val_count:
        raise FileNotFoundError(f"no image/label pairs found in {data_dir}")
    return train_count, val_count


def _classes_file(data_dir: Path, dataset_dir: Path) -> str:
    """Return the ``classes.txt`` for *data_dir*, deriving one when there is none."""
    for candidate in (data_dir / "classes.txt", data_dir.parent / "classes.txt"):
        if candidate.exists():
            return str(candidate)

    class_ids = set()
    for label in sorted((dataset_dir / "train").glob("*.txt")):
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_ids.add(int(line.split()[0]))
    class_num = max(class_ids) + 1 if class_ids else 1

    derived = dataset_dir / "classes.txt"
    derived.write_text("\n".join(f"class{i}" for i in range(class_num)), encoding="utf-8")
    log_message(f"no classes.txt near {data_dir}; derived {class_num} class name(s)")
    return str(derived)


def _pretrained_weights() -> Path:
    """Return an absolute path to the pretrained model the smoke runs start from.

    ultralytics downloads a bare file name into the current directory, so the
    model is cached in the YORU user directory (``YORU_SMOKE_MODEL`` overrides
    it) and always handed over as an absolute path.
    """
    name = os.environ.get("YORU_SMOKE_MODEL", "yolov8n.pt")
    if os.path.isabs(name):
        return Path(name)

    weights_dir = get_yoru_home() / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights = weights_dir / name
    if not weights.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(str(weights))
    if not weights.exists():
        raise FileNotFoundError(f"could not obtain pretrained weights: {weights}")
    return weights


def run_inference(images_dir: str, weights_path: str, out_dir: str) -> None:
    """
    Minimal test entry: run inference on images_dir using weights_path,
    and write results into out_dir. Raise on error; return None on success.

    The weights are loaded through load_yolo_model(), so the smoke test goes
    through the same wrapper the detection GUIs use, and every detection is
    written to ``detections.csv`` in out_dir.
    """
    images = _list_images(Path(images_dir))
    if not images:
        raise FileNotFoundError(f"no images found in {images_dir}")
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = load_yolo_model(str(weights_path))
    names = model.names
    result_path = out_path / "detections.csv"
    detections = 0

    with open(result_path, "w", newline="", encoding="utf-8") as f:
        result_writer = csv.writer(f)
        result_writer.writerow(
            ["image", "x1", "y1", "x2", "y2", "confidence", "class", "class_name"]
        )
        for image_path in images:
            # The detection GUIs hand the wrapper BGR frames, as cv2 reads them.
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"could not read image: {image_path}")
            results = model(frame)
            for x1, y1, x2, y2, conf, cls in results.xyxy[0].tolist():
                result_writer.writerow(
                    [image_path.name, x1, y1, x2, y2, conf, int(cls), names[int(cls)]]
                )
                detections += 1

    log_message(
        f"smoke inference: {len(images)} image(s), {detections} detection(s) "
        f"-> {result_path}"
    )


def run_training(data_dir: str, out_dir: str, epochs: int = 1, device: str = "cpu") -> None:
    """
    Minimal training entry for smoke tests. Should run quickly and write some artifact into out_dir.

    Follows the training GUI: write the dataset YAML with create_yaml_train,
    then launch libs/train_ultralytics.py in a subprocess. Only a few dozen
    images from *data_dir* are used, at a small image size, so the run stays
    short while still going through the real training loop. *device* is
    resolved with yoru.libs.device, so "auto" trains on MPS on Apple Silicon.
    Raise on error; return None on success.
    """
    # Absolute from here on: the staged symlinks point at the source images, and
    # the child process runs in out_dir, so neither may depend on the caller's cwd.
    data_path = Path(data_dir).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    dataset_dir = out_path / "dataset"
    runs_dir = out_path / "runs"
    resolved = resolve_device(device)
    train_count, val_count = _stage_dataset(data_path, dataset_dir)
    weights = _pretrained_weights()

    m_dict = {
        "project_dir": str(dataset_dir),
        "classes_path": _classes_file(data_path, dataset_dir),
        "weight": str(weights),
        "yaml_path": str(dataset_dir / "config.yaml"),
        "img": SMOKE_IMG_SIZE,
        "batch": SMOKE_BATCH,
        "epoch": int(epochs),
    }
    cr_project = create_project(m_dict)
    cr_project.create_yaml()
    cr_project.add_class_info()
    cr_project.add_training_info()

    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--weights",
        str(weights),
        "--data",
        m_dict["yaml_path"],
        "--epochs",
        str(m_dict["epoch"]),
        "--imgsz",
        str(SMOKE_IMG_SIZE),
        "--batch",
        str(SMOKE_BATCH),
        "--project",
        str(runs_dir),
        "--device",
        resolved,
    ]

    log_message(
        f"smoke training on {resolved}: {train_count} train / {val_count} val image(s), "
        f"{m_dict['epoch']} epoch(s)"
    )
    try:
        # Run from out_dir so that ultralytics cannot drop a download into the
        # working tree.
        result = subprocess.run(
            cmd,
            cwd=str(out_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as e:
        log_exception("Failed to start smoke training", e)
        raise

    if result.returncode != 0:
        log_message(f"smoke training failed:\n{result.stdout}", logging.ERROR)
        raise RuntimeError(
            f"training failed with exit code {result.returncode}:\n{result.stdout}"
        )

    produced = sorted(runs_dir.glob("**/*.pt"))
    if not produced:
        log_message(f"smoke training wrote no weights into {runs_dir}", logging.ERROR)
        raise RuntimeError(f"no trained weights were written into {runs_dir}")

    log_message(f"smoke training finished: {produced[0]}")
