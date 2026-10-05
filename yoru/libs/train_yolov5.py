# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Training script for YOLOv5, run with the bundled ultralytics/yolov5 code.

Called by plugins/yolov5_trainer.py via subprocess:
    python ./yoru/libs/train_yolov5.py \
        --weights yolov5s.pt \
        --data    path/to/config.yaml \
        --epochs  300 \
        --imgsz   640 \
        --batch   16 \
        --project path/to/project_dir \
        --name    exp_yolov5s \
        --device  auto

This runs yolov5's own ``train.py`` -- the script YORU v1 launched -- with the
arguments v1 gave it (``--imgsz --batch-size --epochs --data --weights
--project``, and ``--device`` only when CUDA's default is not what is meant),
so a model trained here is the model v1 would have trained.  Everything else is
left at yolov5's defaults, as in v1.  Two things are added around it, neither
of which touches the training itself:

* the run folder is ``exp_<model>`` (``exp_yolov5s``, ``exp_yolov5s2``, ...),
  as for every other YORU v2 trainer, instead of yolov5's bare ``exp``;
* ``--stop-file`` names a file that ends training cleanly after the epoch in
  progress as soon as it appears; the training GUI's "Stop after this epoch"
  button writes it.  See libs/train_stop.py.

The weights are YOLOv5's own: an official name such as ``yolov5s.pt`` that is
not on disk is fetched from the ultralytics/yolov5 releases by yolov5 itself,
never replaced by ultralytics' different ``yolov5su.pt``.
"""

import argparse
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

# This file is run as a script, not imported (see plugins/yolov5_trainer.py),
# so only its own directory is on sys.path.  The conda install in docs/install.md
# runs YORU from a source checkout without pip-installing it, where that leaves
# the package itself unimportable; put the repository root back on the path.
# (DataLoader workers re-run this module's top level with the path already set.)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yoru.libs.device import describe, resolve_device  # noqa: E402
from yoru.libs.train_stop import clear_stop, stop_requested  # noqa: E402
from yoru.libs.yolov5_support import require_bundled_yolov5  # noqa: E402


def run_name(weights: str) -> str:
    """Name of the results folder for this run: ``exp_<model>``.

    The same rule as libs/train_ultralytics.py, so a YOLOv5 run and a YOLO11
    run in one project sit side by side as ``exp_yolov5s`` and ``exp_yolo11s``.
    yolov5 appends 2, 3, ... by itself when the same model is trained again.
    """
    stem = Path(weights).stem.strip() or "model"
    return "exp_" + "_".join(stem.split())


def yolov5_device(preference: str) -> str:
    """The ``--device`` value for yolov5's train.py, or "" to leave it out.

    yolov5's select_device() knows neither "auto" nor a bare "cuda", and naming
    a CUDA device there pins CUDA_VISIBLE_DEVICES.  Its default ("") already
    picks the first CUDA device, so it is only named when something else is
    meant -- exactly what YORU v1 passed.
    """
    device = resolve_device(preference)
    return "" if device == "cuda" else device


def yolov5_argv(args) -> list:
    """yolov5 train.py's command line for these arguments."""
    argv = [
        "--imgsz", str(args.imgsz),
        "--batch-size", str(args.batch),
        "--epochs", str(args.epochs),
        "--data", str(args.data),
        "--weights", str(args.weights),
        "--project", str(args.project),
        "--name", args.name or run_name(args.weights),
    ]
    device = yolov5_device(args.device)
    if device:
        argv += ["--device", device]
    return argv


def install_stop_request(train_module, stop_file) -> None:
    """End training after the current epoch once *stop_file* appears.

    yolov5 asks its EarlyStopping object whether to stop once an epoch has been
    validated, and before that epoch's checkpoint is saved.  Told yes, it saves
    ``last.pt`` / ``best.pt`` as usual, leaves the epoch loop, and still runs
    its final validation of ``best.pt``: the same ending a completed run gets,
    only earlier.  Answering yes from here is therefore all a clean stop takes.
    The class is replaced in train.py's namespace, where ``train()`` looks it
    up, so the bundled yolov5 code itself stays unmodified.
    """
    if not stop_file:
        return
    stop_path = Path(stop_file)
    base = train_module.EarlyStopping

    class StopOnRequest(base):
        def __call__(self, epoch, fitness):
            stop = super().__call__(epoch=epoch, fitness=fitness)
            if stop_requested(stop_path):
                # Take the request: a file left behind would stop the next run
                # after a single epoch.
                clear_stop(stop_path)
                # Flushed: yolov5 logs to stderr, so a buffered stdout line
                # would reach the GUI only after the final validation.
                print(
                    f"[yoru] Stop requested: ending after epoch {epoch + 1}. "
                    "Validation and the final weights are still written.",
                    flush=True,
                )
                return True
            return stop

    train_module.EarlyStopping = StopOnRequest


def _is_directory(entry, directory: Path) -> bool:
    try:
        return Path(entry or ".").resolve() == directory
    except (OSError, ValueError):
        return False


def _use_bundled_yolov5() -> Path:
    """Import yolov5's modules the way running its train.py directly would.

    Its directory goes to the head of sys.path in place of this script's, so
    that ``models``, ``utils`` and ``val`` resolve to yolov5's own and nothing
    in yoru/libs can shadow them.  Done here rather than at import time, so
    that importing this module (the tests do) leaves sys.path alone.
    DataLoader workers inherit the path from this process.
    """
    root = require_bundled_yolov5()
    sys.path[:] = [str(root)] + [p for p in sys.path if not _is_directory(p, _HERE)]
    return root


def _cwd_on_the_drive_of(root: Path, args) -> None:
    """Move to *root*'s drive if the working directory is on another one.

    yolov5's train.py and val.py store their own directory relative to the
    working directory when imported, and on Windows ``os.path.relpath``
    raises across drives -- training from a shell open on D: would fail before
    it started.  The paths handed over are made absolute first, so they still
    mean what they meant.
    """
    try:
        os.path.relpath(root, Path.cwd())
        return
    except ValueError:
        pass
    args.data = str(Path(args.data).resolve())
    args.project = str(Path(args.project).resolve())
    if Path(args.weights).exists():
        args.weights = str(Path(args.weights).resolve())
    # A weight still to be downloaded lands here, beside the others YORU
    # fetches when started from its checkout.
    os.chdir(_REPO_ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a YOLOv5 model with the bundled ultralytics/yolov5 code."
    )
    parser.add_argument("--weights", required=True, help="Pretrained weights (e.g. yolov5s.pt)")
    parser.add_argument("--data",    required=True, help="Path to dataset YAML file")
    parser.add_argument("--epochs",  type=int, default=300, help="Number of training epochs")
    parser.add_argument("--imgsz",   type=int, default=640, help="Input image size")
    parser.add_argument("--batch",   type=int, default=16,  help="Batch size")
    parser.add_argument("--project", default=".",           help="Project output directory")
    parser.add_argument("--name",    default=None,
                        help="Results folder under --project "
                             "(default: exp_<model>, e.g. exp_yolov5s)")
    parser.add_argument("--stop-file", default=None,
                        help="Path of the stop-request file: training ends "
                             "cleanly after the epoch during which this file "
                             "appears (default: no cooperative stop)")
    parser.add_argument("--device",  default="auto",
                        help="Device: auto, cuda, mps, cpu, or a CUDA index "
                             "such as '0' / '0,1'")
    args = parser.parse_args(argv)

    print(f"Device: {describe(args.device)}", flush=True)

    try:
        root = _use_bundled_yolov5()
        _cwd_on_the_drive_of(root, args)
        import train as yolov5_train  # yolov5's train.py

        install_stop_request(yolov5_train, args.stop_file)
        # yolov5 parses its options off sys.argv, as when v1 ran it directly.
        sys.argv = [str(root / "train.py"), *yolov5_argv(args)]
        opt = yolov5_train.parse_opt()
        yolov5_train.main(opt)
    except FileNotFoundError as e:
        print(f"[yoru] File not found: {e}")
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        # Print the traceback: a one-line message is not enough to diagnose or
        # report a training failure.
        import traceback

        print(f"[yoru] Training failed: {e}")
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
