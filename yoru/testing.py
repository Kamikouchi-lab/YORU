# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Entry points used by the smoke tests in ``tests/``.

These used to be stubs that only created a directory (and, for training, wrote a
dummy ``checkpoint.pt``), which made ``tests/test_inference_smoke.py`` and
``tests/test_training_smoke.py`` pass without exercising anything.  They now run
real work through the plugin registry, so a failure in either path is a genuine
signal.
"""

import csv
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def run_inference(images_dir: str, weights_path: str, out_dir: str) -> None:
    """Detect on every image in *images_dir* and write ``detections.csv``.

    Raises on error; returns None on success.
    """
    import cv2

    from yoru.libs.plugins import get_detector

    images = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No images found in {images_dir}")

    detector = get_detector("auto", str(weights_path))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "detections.csv"

    read = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["file_name", "x1", "y1", "x2", "y2",
             "confidence", "class", "class_name"]
        )
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                raise ValueError(f"Could not read image: {img_path}")
            read += 1
            for d in detector.detect(frame):
                writer.writerow(
                    [img_path.name, d["x1"], d["y1"], d["x2"], d["y2"],
                     d["conf"], d["class_id"], d["class_name"]]
                )

    if read == 0:
        raise ValueError(f"No readable images in {images_dir}")


def run_training(
    data_dir: str, out_dir: str, epochs: int = 1, device: str = "cpu"
) -> None:
    """Run a short training job into *out_dir*.

    *data_dir* may be the dataset YAML itself or a directory containing
    ``config.yaml``.  Raises if the training subprocess fails.
    """
    from yoru.libs.plugins import get_trainer
    from yoru.libs.train_progress import ProgressPrinter

    data = Path(data_dir)
    if data.is_dir():
        data_yaml = data / "config.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(f"No config.yaml in {data_dir}")
    else:
        data_yaml = data
        if not data_yaml.is_file():
            raise FileNotFoundError(f"Dataset YAML not found: {data_dir}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainer = get_trainer("ultralytics")
    proc = trainer.train(
        {
            "img_size": 320,
            "batch_size": 1,
            "epochs": int(epochs),
            "data_yaml": str(data_yaml),
            "weights": "yolo11n.pt",
            "project_dir": str(out),
            "model_family": "YOLO",
            "device": device,
        }
    )
    if proc.stdout is not None:
        # Collapse the per-batch progress redraws; see train_progress.py.
        printer = ProgressPrinter()
        for raw_line in proc.stdout:
            printer.write(printer.clean(raw_line))
        printer.close()
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"Training subprocess exited with code {returncode}"
        )
