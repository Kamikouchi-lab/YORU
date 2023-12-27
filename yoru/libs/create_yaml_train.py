import datetime
import os

import yaml


class create_project:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

    def create_yaml(self):
        file_path = self.m_dict["project_dir"] + "/config.yaml"
        with open(file_path, "w") as yf:
            yaml.dump(
                {
                    "path": self.m_dict["project_dir"],
                    "train": self.m_dict["project_dir"] + "/train/",
                    "val": self.m_dict["project_dir"] + "/val/",
                    "yaml_path": file_path,
                    "YOLO_ver": "yolov5",
                    "system_ver": "0.1.0",
                    "create_date": datetime.date.today(),
                },
                yf,
            )
        print("create yaml file")

    def add_class_info(self):
        if "classes.txt" in self.m_dict["classes_path"] and os.path.exists(
            self.m_dict["classes_path"]
        ):
            with open(self.m_dict["classes_path"], "r", encoding="utf-8") as f:
                # ファイルの内容を行ごとに読み込む
                lines = f.readlines()
            # 行のリストから改行文字を削除
            items = [line.strip() for line in lines]

            self.m_dict["class_num"] = len(items)
            self.m_dict["class_list"] = items
            print(self.m_dict["class_list"])

        file_path = self.m_dict["yaml_path"]
        if os.path.exists(file_path):
            print(file_path)
            with open(file_path, "r") as yf:
                exsisting_data = yaml.safe_load(yf)
                if exsisting_data.get("add_class_info_date"):
                    return None
            with open(file_path, "a") as yf:
                yaml.dump(
                    {
                        "nc": self.m_dict["class_num"],
                        "names": self.m_dict["class_list"],
                        "add_class_info_date": datetime.date.today(),
                    },
                    yf,
                )
            print("add class info in yaml file")
        else:
            print("failed....")

    def add_training_info(self):
        file_path = self.m_dict["yaml_path"]

        if os.path.exists(file_path):
            with open(file_path, "r") as yf:
                exsisting_data = yaml.safe_load(yf)
                if exsisting_data.get("training_date"):
                    exsisting_data["image_size"] = self.m_dict["img"]
                    exsisting_data["batch-size"] = self.m_dict["batch"]
                    exsisting_data["epochs"] = self.m_dict["epoch"]
                    exsisting_data["data"] = self.m_dict["yaml_path"]
                    exsisting_data["weights"] = self.m_dict["weight"]
                    exsisting_data["project_dir"] = self.m_dict["project_dir"]
                    exsisting_data["patience"] = False
                    exsisting_data["training_date"] = datetime.date.today()
                    # 変更した内容をYAMLファイルに書き込みます
                    with open(file_path, "w") as yf:
                        yaml.dump(exsisting_data, yf, default_flow_style=False)
                        return None
            with open(file_path, "a") as yf:
                yaml.dump(
                    {
                        "image_size": self.m_dict["img"],
                        "batch-size": self.m_dict["batch"],
                        "epochs": self.m_dict["epoch"],
                        "data": self.m_dict["yaml_path"],
                        "weights": self.m_dict["weight"],
                        "project_dir": self.m_dict["project_dir"],
                        "patience": False,
                        "training_date": datetime.date.today(),
                    },
                    yf,
                )
            print("add class info in yaml file")
        else:
            print("failed....")
