"""Training script for YOLOv8 / YOLO11 via the ultralytics package.

Called by train_GUI.py via subprocess:
    python ./yoru/libs/train_ultralytics.py \
        --weights yolov8s.pt \
        --data    path/to/config.yaml \
        --epochs  300 \
        --imgsz   640 \
        --batch   16 \
        --project path/to/project_dir \
        --device  auto
"""

import argparse
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    # train_GUI.py launches this file by path, so sys.path[0] is yoru/libs and
    # the repo root has to be added before the yoru package can be imported.
    _ROOT = str(Path(__file__).resolve().parents[2])
    if _ROOT not in sys.path:
        sys.path.append(_ROOT)

from yoru.libs.device import describe, resolve_device

warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")


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
    parser.add_argument("--device",  default="auto",        help="Device: auto, cuda, mps, cpu")
    args = parser.parse_args()

    # ultralytics never auto-selects MPS, so the device has to be named here.
    device = resolve_device(args.device)
    print(f"Device: {describe(device)}")

    if "rtdetr" in args.weights.lower():
        from ultralytics import RTDETR
        model = RTDETR(args.weights)
    else:
        from ultralytics import YOLO
        model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        device=device,
    )


if __name__ == "__main__":
    main()
