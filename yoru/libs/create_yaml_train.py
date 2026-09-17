# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import datetime
import os

import yaml

#: The value written to, and read from, a project config's ``task`` key.
TASK_DETECT = "detect"
TASK_OBB = "obb"


def task_of(m_dict):
    """``"obb"`` or ``"detect"`` for this GUI state."""
    return TASK_OBB if m_dict.get("obb") else TASK_DETECT


def is_obb_project(config_data):
    """True when a loaded config.yaml describes an oriented-box project.

    Accepts a missing key -- every project created before OBB support is an
    ordinary detection project -- and also a bare ``obb: true``, which is what
    a user editing the file by hand is most likely to write.
    """
    if not config_data:
        return False
    task = str(config_data.get("task", "") or "").strip().lower()
    if task:
        return task == TASK_OBB
    return bool(config_data.get("obb", False))


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
                    "Model": self.m_dict.get("weight", "yolo11s.pt"),
                    # Ultralytics' own word for it, and the one thing in this
                    # file the whole project hangs on: it decides the label
                    # format labelImg writes, the weight the trainer loads and
                    # how the detector reads its results back.  Projects made
                    # before OBB support have no "task" key, and read as
                    # "detect" everywhere it is looked up.
                    "task": task_of(self.m_dict),
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
                existing_data = yaml.safe_load(yf)
                if existing_data and existing_data.get("add_class_info_date"):
                    print("Class info already exists, skipping")
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
                existing_data = yaml.safe_load(yf) or {}

            training_info = {
                "image_size": self.m_dict["img"],
                "batch-size": self.m_dict["batch"],
                "epochs": self.m_dict["epoch"],
                "data": self.m_dict["yaml_path"],
                "weights": self.m_dict["weight"],
                "project_dir": self.m_dict["project_dir"],
                "patience": False,
                "training_date": datetime.date.today(),
            }

            if existing_data.get("training_date"):
                # Update existing training info
                existing_data.update(training_info)
                with open(file_path, "w") as yf:
                    yaml.dump(existing_data, yf, default_flow_style=False)
                return None

            with open(file_path, "a") as yf:
                yaml.dump(training_info, yf)
            print("add class info in yaml file")
        else:
            print("failed....")
