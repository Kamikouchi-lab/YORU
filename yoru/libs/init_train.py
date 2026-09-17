# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import os

import yaml


class loadingParam:
    def __init__(self):
        print("Training GUI initiation")


#: Families that can be trained on oriented boxes.  Only ultralytics' YOLO
#: detect models have an OBB variant; RT-DETR and the torchvision detectors
#: have no rotated-box head at all, so an OBB project cannot offer them.
OBB_CAPABLE_FAMILIES = ("YOLO",)

# Per-family option definitions
MODEL_FAMILY_CONFIG = {
    "YOLO": {
        # YOLOv5 is intentionally absent: the bundled yolov5 backend was removed
        # in v2.0 and ultralytics cannot train a legacy yolov5 checkpoint.
        "versions":  ["YOLOv8", "YOLO11"],
        "sizes":     ["n", "s", "m", "l", "x"],
    },
    "RT-DETR": {
        "sizes":     ["l", "x"],
    },
    "Faster R-CNN": {
        "backbones": ["ResNet50-FPN"],
    },
    "Mask R-CNN": {
        "backbones": ["ResNet50-FPN"],
    },
    "SSD": {
        "backbones": ["VGG16"],
    },
}


class init_train:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

        self.m_dict["project_dir"] = "."
        self.m_dict["yaml_path"] = self.m_dict["project_dir"] + "/config.yaml"
        self.m_dict["weight_list"] = [
            # YOLOv8
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            # YOLO11
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            # YOLOv8 / YOLO11, oriented boxes
            "yolov8n-obb.pt", "yolov8s-obb.pt", "yolov8m-obb.pt",
            "yolov8l-obb.pt", "yolov8x-obb.pt",
            "yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt",
            "yolo11l-obb.pt", "yolo11x-obb.pt",
            # RT-DETR
            "rtdetr-l.pt", "rtdetr-x.pt",
        ]
        self.m_dict["weight"] = "yolo11s.pt"

        # Oriented bounding boxes.  Set when the project is created and stored
        # in its config.yaml as ``task: obb``; from there it decides the weight
        # (``yolo11s-obb.pt``), the label format labelImg writes, and how the
        # detector reads its results back.  A project cannot change its mind
        # later without relabelling, which is why this lives with the project
        # rather than with the training run.
        self.m_dict["obb"] = False
        self.m_dict["obb_capable_families"] = list(OBB_CAPABLE_FAMILIES)

        # Model family
        self.m_dict["model_family_list"] = list(MODEL_FAMILY_CONFIG.keys())
        self.m_dict["model_family"]      = "YOLO"

        # YOLO-specific
        self.m_dict["yolo_version_list"] = MODEL_FAMILY_CONFIG["YOLO"]["versions"]
        self.m_dict["yolo_version"]      = "YOLO11"
        self.m_dict["yolo_size_list"]    = MODEL_FAMILY_CONFIG["YOLO"]["sizes"]
        self.m_dict["yolo_size"]         = "s"

        # RT-DETR-specific
        self.m_dict["rtdetr_size_list"]  = MODEL_FAMILY_CONFIG["RT-DETR"]["sizes"]
        self.m_dict["rtdetr_size"]       = "l"

        # Torchvision-specific (initial value = Faster R-CNN backbone options)
        self.m_dict["tv_backbone_list"]  = MODEL_FAMILY_CONFIG["Faster R-CNN"]["backbones"]
        self.m_dict["tv_backbone"]       = "ResNet50-FPN"

        self.m_dict["classes_path"]    = "."
        self.m_dict["all_label_dir"]   = self.m_dict["project_dir"] + "/all_label_images"

        self.m_dict["img"]             = 640
        self.m_dict["batch"]           = 16
        self.m_dict["epoch"]           = 300
        self.m_dict["quit"]            = False
        self.m_dict["back_to_home"]    = False

    def __del__(self):
        print("== Initialization finished ==.")


if __name__ == "__main__":
    mdict0 = init_train()
    print(mdict0.m_dict)
