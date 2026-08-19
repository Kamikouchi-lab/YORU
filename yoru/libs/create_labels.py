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

    def _write_classes_txt(self, class_names) -> None:
        """Write labelImg's classes.txt, one class name per line, index order."""
        if not class_names:
            return
        path = os.path.join(self.datas_path, "classes.txt")
        if os.path.exists(path):
            logger.info("classes.txt already exists, leaving it untouched: %s", path)
            return
        ordered = [class_names[k] for k in sorted(class_names)]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(n) for n in ordered) + "\n")
        logger.info("Wrote %s (%d classes)", path, len(ordered))

    def analyze_image(self):
        from yoru.libs.plugins import DEFAULT_CONF_THRESH, get_detector

        try:
            conf_thresh = float(self.m_dict.get("threshold", DEFAULT_CONF_THRESH))
        except (TypeError, ValueError):
            conf_thresh = DEFAULT_CONF_THRESH

        detector = get_detector(
            "auto", self.yolo_model_path, conf_thresh=conf_thresh
        )

        # クラス名の取得
        class_names = detector.names

        # labelImg needs classes.txt beside the labels to open the folder and to
        # map the class indices written below back to names.
        self._write_classes_txt(class_names)

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
                if d["conf"] < conf_thresh:
                    continue
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
