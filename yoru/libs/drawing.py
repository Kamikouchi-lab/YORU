# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Shared visualization utilities for detection results."""

import logging
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from yoru.libs.detector_base import DETECTION_COLUMNS
from yoru.libs.obb import obb_corners, obb_to_aabb

logger = logging.getLogger(__name__)

# Column positions, looked up once from the schema rather than written out as
# literals: the row layout has one definition, in detector_base.
COL_CONF = DETECTION_COLUMNS.index("confidence")
COL_CLASS = DETECTION_COLUMNS.index("class")
COL_CLASS_NAME = DETECTION_COLUMNS.index("class_name")
COLS_OBB = tuple(DETECTION_COLUMNS.index(k) for k in ("cx", "cy", "w", "h", "angle"))


def get_colormap(label_names, colormap_name="gist_rainbow"):
    """Generate a colormap dict mapping label indices to RGB tuples."""
    colormap = {}
    cmap = plt.get_cmap(colormap_name)
    n = len(label_names)
    for i in range(n):
        rgb = [int(d) for d in np.array(cmap(float(i) / n)) * 255][:3]
        colormap[i] = tuple(rgb)
    return colormap


#: Below this many radians a box is drawn as an upright rectangle.  Not an
#: optimisation: cv2.rectangle produces cleaner edges than a four-point
#: polyline, and every detection from a non-OBB model has an angle of exactly
#: zero, so this is the path almost every box takes.
_UPRIGHT_EPS = 1e-4


def draw_box(img, box, color, label=None, thickness=4, font_scale=1.5):
    """Draw one box, rotated or not, with its label above it.

    *box* is ``(cx, cy, w, h, angle)`` -- the same five numbers every part of
    YORU passes a box around as.  An upright box and a rotated one differ only
    in which OpenCV call draws the outline; the label is placed the same way
    for both, at the top-left of the box's upright extent, so a tilted box's
    text does not wander off with the rotation.
    """
    cx, cy, w, h, angle = (float(v) for v in box)
    if abs(angle) < _UPRIGHT_EPS:
        x1, y1, x2, y2 = obb_to_aabb(box)
        cv2.rectangle(
            img,
            pt1=(int(x1), int(y1)),
            pt2=(int(x2), int(y2)),
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_4,
            shift=0,
        )
    else:
        pts = np.array(obb_corners(box), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            img, [pts], isClosed=True, color=color,
            thickness=thickness, lineType=cv2.LINE_AA,
        )
        x1, y1, _x2, _y2 = obb_to_aabb(box)

    if label:
        cv2.putText(
            img,
            text=label,
            org=(int(x1), int(y1) - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=color,
            thickness=5,
            lineType=cv2.LINE_4,
        )
    return img


def draw_detections(img, results, colormap, names=None):
    """Draw a frame's detections.

    *results* are rows in ``yoru.libs.detector_base.DETECTION_COLUMNS`` order,
    which is what ``m_dict["yolo_results"]`` holds.  The oriented columns at
    the end are what make a rotated box draw as one; a model that does not
    predict rotation fills them with the upright box and an angle of zero, so
    there is nothing to branch on here.
    """
    for row in results:
        cls = int(row[COL_CLASS])
        conf = float(row[COL_CONF])
        class_name = row[COL_CLASS_NAME]
        if names is not None:
            class_name = names.get(cls, class_name) if hasattr(names, "get") else class_name
        color = colormap.get(cls, (255, 255, 255))
        box = tuple(float(row[i]) for i in COLS_OBB)
        draw_box(img, box, color, label=f"{class_name} {conf:.2f}")
    return img


class yolo_drawing:
    def __init__(self, m_dict=None):
        self.m_dict = m_dict if m_dict is not None else {}
        self.names = {}
        self.colormap = {}

    def get_colormap(self, label_names, colormap_name):
        return get_colormap(label_names, colormap_name)

    def drawing(self, img, results):
        for row in results:
            cls = int(row[COL_CLASS])
            label = f"{self.names[cls]} {float(row[COL_CONF]):.2f}"
            box = tuple(float(row[i]) for i in COLS_OBB)
            draw_box(img, box, self.colormap[cls], label=label)
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
