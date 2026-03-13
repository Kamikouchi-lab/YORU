import os

import yaml


class loadingParam:
    def __init__(self):
        print("Training GUI initiation")


# Per-family option definitions
MODEL_FAMILY_CONFIG = {
    "YOLO": {
        "versions":  ["YOLOv5", "YOLOv8", "YOLO11"],
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
            # YOLOv5
            "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt",
            # YOLOv8
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            # YOLO11
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
            # RT-DETR
            "rtdetr-l.pt", "rtdetr-x.pt",
        ]
        self.m_dict["weight"] = "yolov5s.pt"

        # Model family
        self.m_dict["model_family_list"] = list(MODEL_FAMILY_CONFIG.keys())
        self.m_dict["model_family"]      = "YOLO"

        # YOLO-specific
        self.m_dict["yolo_version_list"] = MODEL_FAMILY_CONFIG["YOLO"]["versions"]
        self.m_dict["yolo_version"]      = "YOLOv5"
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
