import glob
import itertools
import json
import os
import time
import tkinter as tk
from collections import Counter
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm


class yolo_analysis_image:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        self.yolo_model_path = self.m_dict["model_path"]
        self.datas_path = self.m_dict["datas_dir"]
        print("init")

    def analyze_image(self):
        # Resolve the absolute path of the yolov5 repository (which contains
        # hubconf.py) relative to this file's location, so it does not depend
        # on the current working directory.
        yolov5_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "yolov5"
        )
        yolo_model = torch.hub.load(
            yolov5_dir, "custom", path=self.yolo_model_path, source="local"
        )

        # Get the class names
        class_names = (
            yolo_model.module.names
            if hasattr(yolo_model, "module")
            else yolo_model.names
        )

        search_path = os.path.join(self.datas_path, "*.png")
        img_path_list = glob.glob(search_path)
        image_count = len(img_path_list)

        for img_path in tqdm(img_path_list, desc="Processing images"):
            base_name = os.path.basename(img_path)
            file_name_without_ext = os.path.splitext(base_name)[0]

            frame = cv2.imread(img_path)
            height, width, channels = frame.shape

            yolo_result = yolo_model(frame)

            # Build the output path
            result_txt_path = os.path.join(
                self.datas_path, file_name_without_ext + ".txt"
            )
            result = []
            for *box, conf, cls in yolo_result.xywhn[0]:
                # list in xywh format (center x, center y, width, height)
                class_name = class_names[int(cls.item())]
                # Save the result to the list
                result.append(
                    [
                        int(cls.item()),
                        box[0].item(),
                        box[1].item(),
                        box[2].item(),
                        box[3].item(),
                    ]
                )

            # print(result)
            with open(result_txt_path, "w") as file:
                for sublist in result:
                    file.write(" ".join(map(str, sublist)) + "\n")

        print("Complete!")
