import os

import yaml


class loadingParam:
    def __init__(self):
        print("Training GUI initiation")


class init_train:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

        self.m_dict["project_dir"] = "."
        self.m_dict["yaml_path"] = self.m_dict["project_dir"] + "/config.yaml"
        self.m_dict["weight_list"] = [
            # YOLOv5
            "yolov5n.pt",
            "yolov5s.pt",
            "yolov5m.pt",
            "yolov5l.pt",
            "yolov5x.pt",
            # YOLOv8
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
            # YOLO11
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt",
            "yolo11l.pt",
            "yolo11x.pt",
        ]
        self.m_dict["weight"] = "yolov5s.pt"
        self.m_dict["yolo_version_list"] = ["YOLOv5", "YOLOv8", "YOLO11"]
        self.m_dict["yolo_version"] = "YOLOv5"
        self.m_dict["yolo_size_list"] = ["n", "s", "m", "l", "x"]
        self.m_dict["yolo_size"] = "s"
        self.m_dict["classes_path"] = "."
        self.m_dict["all_label_dir"] = self.m_dict["project_dir"] + "/all_label_images"

        self.m_dict["img"] = 640
        self.m_dict["batch"] = 16
        self.m_dict["epoch"] = 300
        self.m_dict["quit"] = False
        self.m_dict["back_to_home"] = False

    def __del__(self):
        print("== Initialization finished ==.")


if __name__ == "__main__":
    mdict0 = init_train()
    print(mdict0.m_dict)
