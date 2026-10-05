# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Training script for YOLOv5 via the vendored copy in ``yoru/libs/yolov5``.

Called by train_GUI.py via subprocess (see plugins/yolov5_trainer.py):
    python ./yoru/libs/train_yolov5.py \
        --weights yolov5s.pt \
        --data    path/to/config.yaml \
        --epochs  300 \
        --imgsz   640 \
        --batch   16 \
        --project path/to/project_dir \
        --name    exp_yolov5s

The command line mirrors train_ultralytics.py so that the two trainer plugins
stay interchangeable, and the results land in the same place --
``<project>/exp_<model>/weights/best.pt`` -- so evaluation and analysis do not
need to know which backend produced a run.

--stop-file names a file that ends training cleanly after the epoch in
progress as soon as it appears; the training GUI's "Stop after this epoch"
button writes it.  See libs/train_stop.py.

Upstream's ``train.main()`` is deliberately not used: it runs
``check_git_status()``, which fetches from the network inside whatever git
repository it finds itself in, and ``check_requirements()``, which pip-installs
into the live environment.  Neither is acceptable from a GUI button, so this
script does main()'s three useful steps itself -- resolve the run directory,
select the device, call ``train()`` -- and skips the rest.
"""

import argparse
import sys
import warnings
from pathlib import Path

# This file is run as a script, not imported (see plugins/yolov5_trainer.py),
# so only its own directory is on sys.path.  The conda install in docs/install.md
# runs YORU from a source checkout without pip-installing it, where that leaves
# the package itself unimportable; put the repository root back on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yoru.libs.yolov5 import YOLOV5_DIR, ensure_importable  # noqa: E402

warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

#: Upstream's own default hyperparameters for training from a COCO checkpoint.
_HYP = YOLOV5_DIR / "data" / "hyps" / "hyp.scratch-low.yaml"


def run_name(weights: str) -> str:
    """Name of the results folder for this run: ``exp_<model>``.

    The same rule as train_ultralytics.run_name, for the same reason: YOLOv5
    would otherwise default to ``exp``, and ``<project>/train/`` already holds
    the dataset.  Upstream appends 2, 3, ... by itself when a model is trained
    again.
    """
    stem = Path(weights).stem.strip() or "model"
    return "exp_" + "_".join(stem.split())


def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv5 model using the vendored upstream copy."
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
    parser.add_argument("--device",  default=None,
                        help="Training device, e.g. '0', '0,1' or 'cpu' "
                             "(default: chosen automatically)")
    args = parser.parse_args()

    # Bind the top-level 'models'/'utils' that the vendored tree imports itself
    # as, before anything from it is imported.
    v5_dir = ensure_importable()
    # train.py and the "import val" inside it are modules, not packages, so
    # they are reached the way upstream reaches them.  Appended, not inserted:
    # nothing that already resolves may start resolving in here.  This process
    # exists only to train, so the extra names it exposes reach nothing else.
    if str(v5_dir) not in sys.path:
        sys.path.append(str(v5_dir))

    if args.stop_file:
        # train.py reads this; a command-line flag would mean carrying a patch
        # through upstream's argparse as well as its epoch loop.
        import os

        os.environ["YORU_STOP_FILE"] = str(args.stop_file)

    try:
        # Importing train.py appends the vendored directory to sys.path, which
        # is how its own "import val" resolves.  Harmless here: this process
        # exists only to train, and the two names that could shadow anything
        # are already bound by ensure_importable() above.
        import train as yolov5_train
        from utils.callbacks import Callbacks
        from utils.general import increment_path
        from utils.torch_utils import select_device

        # Upstream's parser supplies the ~40 defaults train() reads off opt.
        # Its own argv must not reach it, hence the blanking: several of our
        # flags share a name with upstream's but not their meaning.
        argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            opt = yolov5_train.parse_opt(known=True)
        finally:
            sys.argv = argv

        opt.weights = str(args.weights)
        opt.cfg = ""  # train from the pretrained weights, not from scratch
        opt.data = str(Path(args.data).resolve())
        opt.hyp = str(_HYP)
        opt.epochs = int(args.epochs)
        opt.imgsz = int(args.imgsz)
        opt.batch_size = int(args.batch)
        opt.project = str(Path(args.project).resolve())
        opt.name = args.name or run_name(args.weights)
        opt.device = "" if args.device is None else str(args.device)
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))

        device = select_device(opt.device, batch_size=opt.batch_size)
        yolov5_train.train(opt.hyp, opt, device, Callbacks())
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
