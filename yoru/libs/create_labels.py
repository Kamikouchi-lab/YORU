# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import glob
import logging
import os

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


class yolo_analysis_image:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        self.yolo_model_path = self.m_dict["model_path"]
        self.datas_path = self.m_dict["datas_dir"]
        logger.debug("yolo_analysis_image initialized")

    def analyze_image(self):
        from yoru.libs.plugins import get_detector

        detector = get_detector("auto", self.yolo_model_path)

        # クラス名の取得
        class_names = detector.names

        img_path_list = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"):
            img_path_list.extend(glob.glob(os.path.join(self.datas_path, ext)))
        image_count = len(img_path_list)

        for img_path in tqdm(img_path_list, desc="Processing images"):
            base_name = os.path.basename(img_path)
            file_name_without_ext = os.path.splitext(base_name)[0]

            frame = cv2.imread(img_path)
            if frame is None:
                logger.warning("Failed to read image: %s", img_path)
                continue
            height, width, channels = frame.shape

            detections = detector.detect(frame)

            # 出力パスの作成
            result_txt_path = os.path.join(
                self.datas_path, file_name_without_ext + ".txt"
            )
            result = []
            for d in detections:
                # xywhn形式（中心x, 中心y, 幅, 高さ）に変換（正規化）
                x_center = (d["x1"] + d["x2"]) / 2 / width
                y_center = (d["y1"] + d["y2"]) / 2 / height
                w = (d["x2"] - d["x1"]) / width
                h = (d["y2"] - d["y1"]) / height
                # 結果をリストに保存
                result.append(
                    [
                        d["class_id"],
                        x_center,
                        y_center,
                        w,
                        h,
                    ]
                )

            with open(result_txt_path, "w") as file:
                for sublist in result:
                    file.write(" ".join(map(str, sublist)) + "\n")

        logger.info("Label creation complete!")
