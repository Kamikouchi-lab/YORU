# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Shared visualization utilities for detection results."""

import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def get_colormap(label_names, colormap_name="gist_rainbow"):
    """Generate a colormap dict mapping label indices to RGB tuples."""
    colormap = {}
    cmap = plt.get_cmap(colormap_name)
    n = len(label_names)
    for i in range(n):
        rgb = [int(d) for d in np.array(cmap(float(i) / n)) * 255][:3]
        colormap[i] = tuple(rgb)
    return colormap


def draw_detections(img, results, colormap, names):
    """Draw bounding boxes and labels on an image.

    Args:
        img: The image (numpy array) to draw on.
        results: Iterable of detection results. Each item should unpack as
            (frame_no, x1, y1, x2, y2, ..., conf, cls, class_name, ...).
        colormap: Dict mapping class_id -> (R, G, B).
        names: Dict mapping class_id -> class_name (used for label text).

    Returns:
        The annotated image.
    """
    for *box, conf, cls, class_name in results:
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(
            img,
            pt1=(int(box[0]), int(box[1])),
            pt2=(int(box[2]), int(box[3])),
            color=colormap.get(int(cls), (255, 255, 255)),
            thickness=4,
            lineType=cv2.LINE_4,
            shift=0,
        )
        cv2.putText(
            img,
            text=label,
            org=(int(box[0]), int(box[1]) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=colormap.get(int(cls), (255, 255, 255)),
            thickness=5,
            lineType=cv2.LINE_4,
        )
    return img


class yolo_drawing:
    def __init__(self, m_dict=None):
        self.m_dict = m_dict if m_dict is not None else {}
        self.names = {}
        self.colormap = {}

    def get_colormap(self, label_names, colormap_name):
        return get_colormap(label_names, colormap_name)

    def drawing(self, img, results):
        for *box, conf, cls, name, _time in results:
            label = f"{self.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(
                img,
                pt1=(int(box[0]), int(box[1])),
                pt2=(int(box[2]), int(box[3])),
                color=self.colormap[int(cls)],
                thickness=4,
                lineType=cv2.LINE_4,
                shift=0,
            )
            cv2.putText(
                img,
                text=label,
                org=(int(box[0]), int(box[1]) - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=self.colormap[int(cls)],
                thickness=5,
                lineType=cv2.LINE_4,
            )
        return img

    def YOLOdraw(self, m_dict):
        logger.info("YOLO detection start...")

        while True:
            if self.m_dict["yolo_process_state"]:
                self.m_dict = m_dict

                while True:
                    self.names = self.m_dict["class_name_list"]
                    self.colormap = self.get_colormap(self.names, "gist_rainbow")

                    image = self.m_dict["current_camera_frame"]
                    results = self.m_dict["yolo_results"]

                    if image.any() and self.m_dict["yolo_detection"]:
                        image_result = self.drawing(image, results)
                        self.m_dict["yolo_detection_frame"] = image_result
                        self.m_dict["now"] = time.perf_counter()

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    elif self.m_dict["quit"]:
                        break
                    elif not self.m_dict["yolo_process_state"]:
                        logger.info("YOLO drawing break")
                        break
            if self.m_dict["quit"]:
                break
