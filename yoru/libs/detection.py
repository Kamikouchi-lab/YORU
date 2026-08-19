# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import logging
import time

import cv2
import numpy as np

from yoru.libs.plugins import get_detector

logger = logging.getLogger(__name__)


class yolo_detection:
    def __init__(self, m_dict=None):
        self.m_dict = m_dict if m_dict is not None else {}
        self.yolo_model_path = self.m_dict["yolo_model"]
        self.names = {}
        self.colormap = {}

    def detect(self, m_dict):
        logger.info("YOLO detection start...")

        while True:
            try:
                if not self.m_dict.get("yolo_process_state", False):
                    if self.m_dict.get("quit", False):
                        break
                    time.sleep(0.01)
                    continue

                self.m_dict = m_dict
                self.yolo_model_path = self.m_dict["yolo_model"]
                logger.info("Model: %s", self.m_dict["yolo_model"])

                # Determine which detector backend to use.
                # "detector_backend" takes priority; fall back to "yolo_model_type".
                backend = self.m_dict.get(
                    "detector_backend",
                    self.m_dict.get("yolo_model_type", "auto"),
                )
                self.detector = get_detector(backend, self.yolo_model_path)

                self.m_dict["class_list"] = self.detector.names
                self.m_dict["class_name_list"] = list(
                    self.m_dict["class_list"].values()
                )
                logger.info("Classes: %s", self.m_dict["class_name_list"])

                while True:
                    image = self.m_dict.get("current_camera_frame")
                    if image is not None and image.size > 0 and self.m_dict["yolo_detection"]:
                        detections = self.detector.detect(image)

                        n = len(detections)
                        yolo_results = np.empty((n, 8), dtype=object)
                        yoru_names_list = []
                        for i, d in enumerate(detections):
                            yolo_results[i] = [
                                d["x1"],
                                d["y1"],
                                d["x2"],
                                d["y2"],
                                d["conf"],
                                d["class_id"],
                                d["class_name"],
                                self.m_dict["total_time"],
                            ]
                            yoru_names_list.append(d["class_name"])

                        self.m_dict["yolo_class_names"] = yoru_names_list
                        self.m_dict["yolo_results"] = yolo_results
                        self.m_dict["now"] = time.perf_counter()

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    elif self.m_dict["quit"]:
                        break
                    elif not self.m_dict["yolo_process_state"]:
                        logger.info("YOLO break")
                        break
            except Exception as e:
                logger.error("Detection error: %s", e)
                time.sleep(0.5)
            if self.m_dict.get("quit", False):
                break


if __name__ == "__main__":
    d = {}
    imgWin = yolo_detection(m_dict=d)
    imgWin.detect(d)
