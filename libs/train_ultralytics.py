"""Training script for YOLOv8 / YOLO11 via the ultralytics package.

Called by train_GUI.py via subprocess:
    python ./libs/train_ultralytics.py \
        --weights yolov8s.pt \
        --data    path/to/config.yaml \
        --epochs  300 \
        --imgsz   640 \
        --batch   16 \
        --project path/to/project_dir
"""

import argparse
import warnings

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
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
