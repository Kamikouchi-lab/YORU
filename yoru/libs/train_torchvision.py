"""Training script for Faster R-CNN, Mask R-CNN, and SSD using YOLO-format labels.

Called by train_GUI.py via subprocess:
    python ./yoru/libs/train_torchvision.py \\
        --model   fasterrcnn \\
        --data    path/to/config.yaml \\
        --epochs  50 \\
        --batch   4 \\
        --project path/to/project_dir

Supported models: fasterrcnn, maskrcnn, ssd

Label format (YOLO-style .txt):
    <class_id> <x_center> <y_center> <width> <height>  (all normalized to [0, 1])

Saved checkpoint format (loadable by TorchvisionWrapper):
    {
        "epoch": int,
        "model_state_dict": ...,
        "num_classes": int,
        "names": {0: "cat", 1: "dog", ...},
        "model_type": "fasterrcnn" | "maskrcnn" | "ssd",
    }
Output files: <project>/<name>/<model_type>_best.pt  and  <model_type>_last.pt
"""

import argparse
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor


class YOLOFormatDataset(Dataset):
    """Loads images and YOLO-format .txt labels for torchvision detection models."""

    def __init__(self, img_dir: Path, label_dir: Path, is_mask_rcnn: bool = False):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.is_mask_rcnn = is_mask_rcnn
        self.img_files = sorted(
            f for f in self.img_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        img_tensor = to_tensor(image)  # [C, H, W] in [0, 1], no manual resize

        boxes, labels = [], []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = max(0.0, (xc - bw / 2) * w)
                    y1 = max(0.0, (yc - bh / 2) * h)
                    x2 = min(float(w), (xc + bw / 2) * w)
                    y2 = min(float(h), (yc + bh / 2) * h)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls + 1)  # 1-indexed (0 = background)

        if boxes:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes_t, "labels": labels_t}

        if self.is_mask_rcnn:
            masks = []
            for box in boxes_t:
                mask = torch.zeros((h, w), dtype=torch.uint8)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                mask[y1:y2, x1:x2] = 1
                masks.append(mask)
            target["masks"] = (
                torch.stack(masks) if masks
                else torch.zeros((0, h, w), dtype=torch.uint8)
            )

        return img_tensor, target


def build_model(model_type: str, num_classes: int):
    """Build a torchvision detection model with a custom head for num_classes."""
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        maskrcnn_resnet50_fpn,
        ssd300_vgg16,
    )
    from torchvision.models.detection._utils import retrieve_out_channels
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models.detection.ssd import SSDHead

    n = num_classes + 1  # +1 for background class

    if model_type == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)

    elif model_type == "maskrcnn":
        model = maskrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, n)

    elif model_type == "ssd":
        model = ssd300_vgg16(weights="DEFAULT")
        in_channels = retrieve_out_channels(model.backbone, (300, 300))
        num_anchors = model.anchor_generator.num_anchors_per_location()
        model.head = SSDHead(in_channels, num_anchors, n)

    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN / Mask R-CNN / SSD with YOLO-format labels."
    )
    parser.add_argument(
        "--model", required=True, choices=["fasterrcnn", "maskrcnn", "ssd"],
        help="Model architecture"
    )
    parser.add_argument("--data",    required=True, help="Path to config.yaml")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--batch",   type=int, default=4)
    parser.add_argument("--project", required=True, help="Output directory")
    parser.add_argument("--name",    default="train")
    args = parser.parse_args()

    with open(args.data) as f:
        config = yaml.safe_load(f)

    data_root   = Path(args.data).parent
    num_classes = config["nc"]
    names       = {i: config["names"][i] for i in range(len(config["names"]))}

    is_mask = args.model == "maskrcnn"
    train_ds = YOLOFormatDataset(
        data_root / "train/images", data_root / "train/labels", is_mask
    )
    val_ds = YOLOFormatDataset(
        data_root / "val/images", data_root / "val/labels", is_mask
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} images  Val: {len(val_ds)} images")

    model = build_model(args.model, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=5e-4
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.epochs // 3), gamma=0.1
    )

    output_dir = Path(args.project) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Step [{i+1}/{len(train_loader)}] "
                    f"Loss: {losses.item():.4f}"
                )

        lr_scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch [{epoch+1}/{args.epochs}] Avg Loss: {avg_loss:.4f}")

        checkpoint = {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "num_classes":      num_classes,
            "names":            names,
            "model_type":       args.model,
        }

        torch.save(checkpoint, output_dir / f"{args.model}_last.pt")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, output_dir / f"{args.model}_best.pt")
            print(f"  -> Best model saved (loss={best_loss:.4f})")

    print(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
