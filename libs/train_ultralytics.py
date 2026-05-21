import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8/YOLO11 with Ultralytics")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="initial weights")
    parser.add_argument("--data", type=str, required=True, help="path to dataset YAML")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="runs/train", help="output project dir")
    parser.add_argument("--name", type=str, default="exp", help="run name")
    parser.add_argument("--patience", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
