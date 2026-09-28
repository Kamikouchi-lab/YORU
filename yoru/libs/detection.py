# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import logging
import time

import numpy as np

from yoru.libs.detector_base import DETECTION_COLUMNS, detection_row
from yoru.libs.plugins import get_detector
from yoru.libs.realtime_state import clear_detection

logger = logging.getLogger(__name__)


class yolo_detection:
    def __init__(self, m_dict=None):
        self.m_dict = m_dict if m_dict is not None else {}
        self.yolo_model_path = self.m_dict["yolo_model"]
        self.names = {}
        self.colormap = {}

    def detect(self, m_dict):
        self.m_dict = m_dict
        detector = None
        last_frame = None
        try:
            while not m_dict.get("quit", False):
                try:
                    if not m_dict.get("yolo_process_state", False):
                        clear_detection(m_dict)
                        detector = None
                        last_frame = None
                        time.sleep(0.01)
                        continue
                    if detector is None:
                        self.yolo_model_path = m_dict["yolo_model"]
                        backend = m_dict.get("detector_backend", m_dict.get("yolo_model_type", "auto"))
                        detector = get_detector(backend, self.yolo_model_path)
                        self.detector = detector
                        m_dict["class_list"] = detector.names
                        m_dict["class_name_list"] = list(detector.names.values())
                    if not m_dict.get("yolo_detection", False) or not m_dict.get("capture_running", False):
                        clear_detection(m_dict)
                        time.sleep(0.01)
                        continue
                    generation = m_dict.get("detection_generation", 0)
                    if (m_dict.get("camera_frame_id"), generation) == last_frame:
                        time.sleep(0.001)
                        continue
                    snapshot = m_dict.get("camera_snapshot")
                    if snapshot is None:
                        time.sleep(0.01)
                        continue
                    frame_id, captured_at, image = snapshot
                    key = (frame_id, generation)
                    if key == last_frame:
                        time.sleep(0.001)
                        continue
                    last_frame = key
                    detections = detector.detect(image)
                    # Do not republish a result after OFF/reload/quit during inference.
                    if (m_dict.get("quit", False) or not m_dict.get("yolo_detection", False)
                            or not m_dict.get("yolo_process_state", False)
                            or not m_dict.get("capture_running", False)
                            or generation != m_dict.get("detection_generation", 0)):
                        clear_detection(m_dict)
                        continue
                    rows = np.asarray(
                        [detection_row(d, captured_at - m_dict["t0"]) for d in detections],
                        dtype=object,
                    ).reshape(-1, len(DETECTION_COLUMNS))
                    published_at = time.perf_counter()
                    m_dict["yolo_results"] = rows
                    m_dict["yolo_class_names"] = [d["class_name"] for d in detections]
                    m_dict["now"] = published_at
                    m_dict["detection_snapshot"] = (frame_id, captured_at, published_at, generation, rows)
                except Exception:
                    clear_detection(m_dict)
                    logger.exception("Detection failed")
                    time.sleep(0.5)
        finally:
            clear_detection(m_dict)


if __name__ == "__main__":
    d = {}
    imgWin = yolo_detection(m_dict=d)
    imgWin.detect(d)
