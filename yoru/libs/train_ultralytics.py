# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Training script for YOLOv8 / YOLO11 via the ultralytics package.

Called by train_GUI.py via subprocess:
    python ./yoru/libs/train_ultralytics.py \
        --weights yolov8s.pt \
        --data    path/to/config.yaml \
        --epochs  300 \
        --imgsz   640 \
        --batch   16 \
        --project path/to/project_dir \
        --name    exp_yolov8s

--stop-file names a file that ends training cleanly after the epoch in
progress as soon as it appears; the training GUI's "Stop after this epoch"
button writes it.  See libs/train_stop.py.
"""

import argparse
import sys
import warnings
from pathlib import Path

# This file is run as a script, not imported (see plugins/ultralytics_trainer.py),
# so only its own directory is on sys.path.  The conda install in docs/install.md
# runs YORU from a source checkout without pip-installing it, where that leaves
# the package itself unimportable; put the repository root back on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yoru.libs.train_stop import clear_stop, stop_requested  # noqa: E402

warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")


def run_name(weights: str) -> str:
    """Name of the results folder for this run: ``exp_<model>``.

    Ultralytics would otherwise default to ``train``, which is also where YORU
    keeps the training images (``<project>/train/``), so every run landed in a
    fresh ``train2``, ``train3``, ... beside the dataset.  Naming the run after
    the model keeps different models apart; ultralytics appends 2, 3, ... by
    itself when the same model is trained again.
    """
    stem = Path(weights).stem.strip() or "model"
    return "exp_" + "_".join(stem.split())


def install_stop_callback(model, stop_file) -> None:
    """End training after the current epoch once *stop_file* appears.

    ``on_train_epoch_end`` fires before ultralytics decides whether to
    validate and save, and its epoch loop breaks as soon as ``trainer.stop``
    is set.  Setting the flag from here therefore gets the epoch that just
    finished validated, written to ``last.pt`` / ``best.pt`` and followed by
    the usual final evaluation: the same ending as a run that reaches its last
    epoch, only earlier.  Killing the process instead would lose all three.
    """
    if not stop_file:
        return
    stop_path = Path(stop_file)

    def _stop_if_requested(trainer):
        if not stop_requested(stop_path):
            return
        # Take the request: a file left behind would stop the next run after
        # a single epoch.
        clear_stop(stop_path)
        print(
            f"[yoru] Stop requested: ending after epoch {trainer.epoch + 1}. "
            "Validation and the final weights are still written."
        )
        trainer.stop = True

    model.add_callback("on_train_epoch_end", _stop_if_requested)


def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 / YOLO11 model using the ultralytics package."
    )
    parser.add_argument("--weights", required=True, help="Pretrained weights (e.g. yolov8s.pt)")
    parser.add_argument("--data",    required=True, help="Path to dataset YAML file")
    parser.add_argument("--epochs",  type=int, default=300, help="Number of training epochs")
    parser.add_argument("--imgsz",   type=int, default=640, help="Input image size")
    parser.add_argument("--batch",   type=int, default=16,  help="Batch size")
    parser.add_argument("--project", default=".",           help="Project output directory")
    parser.add_argument("--name",    default=None,
                        help="Results folder under --project "
                             "(default: exp_<model>, e.g. exp_yolo11s)")
    parser.add_argument("--stop-file", default=None,
                        help="Path of the stop-request file: training ends "
                             "cleanly after the epoch during which this file "
                             "appears (default: no cooperative stop)")
    parser.add_argument("--device",  default=None,
                        help="Training device, e.g. '0', '0,1' or 'cpu' "
                             "(default: chosen by ultralytics)")
    args = parser.parse_args()

    try:
        if "rtdetr" in args.weights.lower():
            from ultralytics import RTDETR
            model = RTDETR(args.weights)
        else:
            from ultralytics import YOLO
            model = YOLO(args.weights)
        install_stop_callback(model, args.stop_file)
        train_kwargs = dict(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name or run_name(args.weights),
        )
        if args.device is not None:
            train_kwargs["device"] = args.device
        model.train(**train_kwargs)
    except FileNotFoundError as e:
        print(f"[yoru] Model weights not found: {e}")
        raise SystemExit(1)
    except Exception as e:
        # Print the traceback: a one-line message is not enough to diagnose or
        # report a training failure.
        import traceback

        print(f"[yoru] Training failed: {e}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
