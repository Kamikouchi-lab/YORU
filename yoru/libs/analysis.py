# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import logging
import os
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import pandas as pd
import torch
from munkres import Munkres

from yoru.libs.detector_base import obb_of
from yoru.libs.drawing import draw_box, get_colormap
from yoru.libs.plugins import DEFAULT_CONF_THRESH, get_detector

#: Columns every analysis table carries for the box's own shape, inserted
#: between the centre and the confidence.  ``angle`` is in radians and is 0 for
#: a model that does not predict rotation, so an ordinary detection run's table
#: gains three columns and loses none, and one reader handles both kinds of
#: output.
OBB_COLUMNS = ["w", "h", "angle"]

logger = logging.getLogger(__name__)


def _conf_thresh(m_dict) -> float:
    """Confidence threshold chosen in the GUI, as a float."""
    try:
        return float(m_dict.get("threshold", DEFAULT_CONF_THRESH))
    except (TypeError, ValueError):
        return DEFAULT_CONF_THRESH


class yolo_analysis:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        self.yolo_model_path = self.m_dict["model_path"]
        self.mov_path_list = self.m_dict["input_path"]
        self.out_path = self.m_dict["output_path"]
        logger.debug("yolo_analysis initialized")

    def cal_id(self, pre_mat, cur_mat):
        pre_mat_calculate = pre_mat.copy()
        cur_mat_calculate = cur_mat.copy()


        actual_cur_num = len(cur_mat_calculate)
        actual_pre_num = len(pre_mat_calculate)

        while len(cur_mat_calculate) > len(pre_mat_calculate):
            pre_mat_calculate.append((-1000, -1000))
        while len(pre_mat_calculate) > len(cur_mat_calculate):
            cur_mat_calculate.append((-1000, -1000))

        if actual_cur_num < 1:
            return None
        pre_mat_calculate = torch.tensor(pre_mat_calculate).type(torch.float64)
        cur_mat_calculate = torch.tensor(cur_mat_calculate).type(torch.float64)

        # print(pre_mat , "pre_mat")
        # print(cur_mat , "cur_mat")
        matrix = torch.cdist(pre_mat_calculate, cur_mat_calculate)
        matrix = matrix.numpy()
        match_mat = Munkres().compute(matrix)

        ret_match_mat = []

        for i, j in match_mat:
            if i >= actual_pre_num:
                i = -1
            if j >= actual_cur_num:
                j = -1
            ret_match_mat.append((i, j))

        return ret_match_mat

    def drawing(self, result, img):
        for (res_frame_no, *res_box, res_x_center, res_y_center,
             res_w, res_h, res_angle, res_conf, res_cls, res_class_name) in result:
            label = f"{res_class_name} {res_conf:.2f}"
            draw_box(
                img,
                (res_x_center, res_y_center, res_w, res_h, res_angle),
                self.colormap[int(res_cls)],
                label=label,
            )
        return img

    def tracking_drawing(self, result, img):
        for (res_frame_no, *res_box, res_x_center, res_y_center,
             res_w, res_h, res_angle, res_conf, res_cls, res_class_name,
             tracking_id) in result:
            label = f"{res_class_name} {res_conf:.2f}"
            label += f" id:{tracking_id}"
            draw_box(
                img,
                (res_x_center, res_y_center, res_w, res_h, res_angle),
                self.colormap[int(res_cls)],
                label=label,
            )
        return img

    def analyze(self):
        """Detect on every configured movie and write one CSV per movie.

        Progress is published through ``m_dict`` so that this method can run on
        a worker thread; the GUI render loop copies it into the widgets.
        """
        self.m_dict["estimate_time"] = "Estimated remaining time: calculating..."
        self.m_dict["no_movies"] = "Leaving movies: calculating..."
        self.m_dict["movie_progress"] = 0.0
        detector = get_detector(
            "auto", self.yolo_model_path, conf_thresh=_conf_thresh(self.m_dict)
        )

        # Get class names
        self.class_names = detector.names

        self.colormap = get_colormap(self.class_names, "gist_rainbow")

        movie_count = len(self.mov_path_list)
        total_movies = movie_count
        self.m_dict["no_movies"] = f"Leaving movies: {int(movie_count)} movies"
        print(f"=== Start movie analysis: {total_movies} movie(s) ===", flush=True)

        for movie_index, self.mov_path in enumerate(self.mov_path_list, start=1):
            df_results = pd.DataFrame()
            result_list = []
            video = cv2.VideoCapture(self.mov_path)
            out = None
            frame_count = 0

            # For tracking
            pre_ids = []
            pre_center_pos = []  # Stores the previous position information
            global_counter = 0

            # Get the file name (without extension)
            base_name = os.path.basename(self.mov_path)
            file_name_without_ext = os.path.splitext(base_name)[0]

            # Join the new file name with the specified output directory
            file_path = os.path.join(self.out_path, file_name_without_ext + ".csv")

            try:
                # Output video settings
                if self.m_dict["create_video"]:
                    out_movie_path = os.path.join(
                        self.out_path, file_name_without_ext + "_render_" + ".mp4"
                    )
                    out = cv2.VideoWriter(
                        out_movie_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        video.get(cv2.CAP_PROP_FPS),
                        (
                            int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        ),
                    )

                # Get the number of frames in the video
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                process_times = []

                result_list = []
                pre_ids = []
                self.m_dict["movie_progress"] = 0.0

                print(
                    f"[{movie_index}/{total_movies}] Analyzing '{base_name}' "
                    f"({total_frames} frames)...",
                    flush=True,
                )
                last_logged_pct = -10  # stdout progress, logged in 10% steps

                while video.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        self.m_dict["estimate_time"] = (
                            "Estimated remaining time: Processing"
                        )
                        break

                    start_time = time.time()

                    if self.m_dict["v_flip"]:
                        frame = cv2.flip(frame, 0)

                    if self.m_dict["h_flip"]:
                        frame = cv2.flip(frame, 1)

                    detections = detector.detect(frame)

                    cur_center_pos = []
                    result = []
                    result_excluded = []
                    exclude_classes = list(self.m_dict.get("tracking_exclude_classes", []))
                    for d in detections:
                        if d["conf"] < self.m_dict["threshold"]:
                            continue
                        # The centre comes from the oriented box, which for a
                        # rectangle is its centroid either way -- the same
                        # number as before for an upright detection, and the
                        # right one for a rotated detection.
                        x_center, y_center, box_w, box_h, box_angle = obb_of(d)

                        entry = [
                            frame_count,
                            d["x1"],
                            d["y1"],
                            d["x2"],
                            d["y2"],
                            x_center,
                            y_center,
                            box_w,
                            box_h,
                            box_angle,
                            d["conf"],
                            d["class_id"],
                            d["class_name"],
                        ]

                        if self.m_dict["tracking_state"] and d["class_id"] in exclude_classes:
                            result_excluded.append(entry)
                        else:
                            result.append(entry)
                            cur_center_pos.append((x_center, y_center))

                    if self.m_dict["tracking_state"]:
                        id_matrix = self.cal_id(pre_center_pos, cur_center_pos)
                        cur_ids = []
                        if id_matrix is not None:
                            id_matrix.sort(
                                key=lambda x: x[1] if x[1] >= 0 else float("inf")
                            )
                            for ids in id_matrix:
                                if ids[0] == -1 or ids[1] == -1:
                                    cur_ids.append(global_counter)
                                    global_counter += 1
                                else:
                                    if 0 <= ids[0] < len(pre_ids):
                                        cur_ids.append(pre_ids[ids[0]])
                                    else:
                                        cur_ids.append(global_counter)
                                        global_counter += 1
                            result = [x + [y] for x, y in zip(result, cur_ids)]

                        pre_ids = cur_ids
                        pre_center_pos = cur_center_pos

                        # Add excluded classes with tracking_id=-1
                        result = result + [x + [-1] for x in result_excluded]

                    if self.m_dict["create_video"]:
                        if self.m_dict["tracking_state"]:
                            frame = self.tracking_drawing(result, frame)
                        else:
                            frame = self.drawing(result, frame)
                        out.write(frame)

                    frame_count += 1
                    result_list = result_list + result

                    progress = frame_count / total_frames if total_frames > 0 else 0.0
                    self.m_dict["movie_progress"] = progress

                    pct = int(progress * 100)
                    if pct // 10 > last_logged_pct // 10:
                        last_logged_pct = pct
                        print(
                            f"[{movie_index}/{total_movies}] {base_name}: "
                            f"{pct}% ({frame_count}/{total_frames} frames)",
                            flush=True,
                        )

                    end_time = time.time()
                    process_time = end_time - start_time
                    process_times.append(process_time)

                    avg_process_time = sum(process_times) / len(process_times)
                    remaining_frames = total_frames - frame_count
                    remaining_time_estimate = avg_process_time * remaining_frames
                    self.m_dict["estimate_time"] = (
                        f"Estimated remaining time: {int(remaining_time_estimate)} seconds"
                    )

                # Convert the list to a dataframe
                if self.m_dict["tracking_state"]:
                    df_results = pd.DataFrame(
                        result_list,
                        columns=[
                            "frame",
                            "x1",
                            "y1",
                            "x2",
                            "y2",
                            "x_center",
                            "y_center",
                            *OBB_COLUMNS,
                            "confidence",
                            "class",
                            "class_name",
                            "tracking_id",
                        ],
                    )
                else:
                    df_results = pd.DataFrame(
                        result_list,
                        columns=[
                            "frame",
                            "x1",
                            "y1",
                            "x2",
                            "y2",
                            "x_center",
                            "y_center",
                            *OBB_COLUMNS,
                            "confidence",
                            "class",
                            "class_name",
                        ],
                    )
                df_results.to_csv(file_path, index=False)
            finally:
                video.release()
                if out is not None:
                    out.release()

            movie_count = movie_count - 1
            self.m_dict["no_movies"] = f"Leaving movies: {int(movie_count)} movies"

        self.m_dict["estimate_time"] = "Estimated remaining time: none"
        self.m_dict["no_movies"] = "Leaving movies: none"
        self.m_dict["movie_progress"] = 1.0
        print("=== Movie analysis complete ===", flush=True)

    def create_video(self, mov_path=None):
        """Render an annotated copy of *mov_path*.

        *mov_path* defaults to the movie analysed most recently, then to the
        first configured input, so the method no longer depends on ``analyze()``
        having populated ``self.mov_path`` as a side effect.
        """
        if mov_path is None:
            mov_path = getattr(self, "mov_path", None)
        if mov_path is None:
            mov_path = self.mov_path_list[0] if len(self.mov_path_list) else None
        if mov_path is None:
            raise ValueError("No input movie selected for create_video()")
        self.mov_path = mov_path

        conf_thresh = _conf_thresh(self.m_dict)
        self.m_dict["cr_estimate_time"] = "Estimated remaining time: calculating..."
        detector = get_detector(
            "auto", self.yolo_model_path, conf_thresh=conf_thresh
        )
        self.class_names = detector.names
        self.colormap = get_colormap(self.class_names, "gist_rainbow")

        base_name = os.path.basename(self.mov_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        out_movie_path = os.path.join(
            self.out_path, file_name_without_ext + "_render_" + ".mp4"
        )

        cap = cv2.VideoCapture(self.mov_path)
        out = cv2.VideoWriter(
            out_movie_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        process_times = []
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    self.m_dict["cr_estimate_time"] = (
                        "Estimated remaining time: Processing"
                    )
                    break

                start_time = time.time()

                if self.m_dict["v_flip"]:
                    frame = cv2.flip(frame, 0)

                if self.m_dict["h_flip"]:
                    frame = cv2.flip(frame, 1)

                detections = detector.detect(frame)

                result = []
                for d in detections:
                    if d["conf"] < conf_thresh:
                        continue
                    x_center, y_center, box_w, box_h, box_angle = obb_of(d)
                    result.append([
                        frame_count,
                        d["x1"], d["y1"], d["x2"], d["y2"],
                        x_center, y_center,
                        box_w, box_h, box_angle,
                        d["conf"], d["class_id"], d["class_name"],
                    ])
                result_frame = self.drawing(result, frame)

                out.write(result_frame)
                frame_count += 1

                end_time = time.time()
                process_time = end_time - start_time
                process_times.append(process_time)

                avg_process_time = sum(process_times) / len(process_times)
                remaining_frames = total_frames - frame_count
                remaining_time_estimate = avg_process_time * remaining_frames
                self.m_dict["cr_estimate_time"] = (
                    f"Estimated remaining time: {int(remaining_time_estimate)} seconds"
                )
        finally:
            cap.release()
            out.release()

        self.m_dict["cr_estimate_time"] = "Estimated remaining time: none"


class yolo_analysis_image:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        self.yolo_model_path = self.m_dict["model_path"]
        self.img_path_list = self.m_dict["input_path_image"]
        self.out_path = self.m_dict["output_path"]
        logger.debug("yolo_analysis_image initialized")

    def drawing(self, img, box, conf, cls):
        """Draw one detection.

        *box* is ``(cx, cy, w, h, angle)`` -- the same five numbers every other
        part of YORU passes a box around as, so a rotated detection draws as a
        rotated box here too.
        """
        label = f"{self.class_names[int(cls)]} {conf:.2f}"
        return draw_box(img, box, self.colormap[int(cls)], label=label)

    def analyze_image(self):
        """Detect on every configured image and write one combined CSV.

        Progress is published through ``m_dict`` so this can run off the GUI thread.
        """
        conf_thresh = _conf_thresh(self.m_dict)
        self.m_dict["analy_state"] = "Analyzing..."
        self.m_dict["image_progress"] = 0.0
        self.m_dict["image_progress_label"] = "0%"

        detector = get_detector(
            "auto", self.yolo_model_path, conf_thresh=conf_thresh
        )

        # Get class names
        self.class_names = detector.names

        self.colormap = get_colormap(self.class_names, "gist_rainbow")

        image_count = len(self.img_path_list)
        print(f"=== Start image analysis: {image_count} image(s) ===", flush=True)
        last_logged_pct = -10  # For progress logging to stdout (output in 10% steps)

        df_results = pd.DataFrame()
        result_list = []
        # Join the new file name with the specified output directory
        file_path = os.path.join(self.out_path, "image_analysis_results" + ".csv")

        for image_index, self.img_path in enumerate(self.img_path_list):
            base_name = os.path.basename(self.img_path)
            file_name_without_ext = os.path.splitext(base_name)[0]

            frame = cv2.imread(self.img_path)
            if frame is None:
                logger.warning("Failed to read image: %s", self.img_path)
                continue
            if self.m_dict["v_flip"]:
                frame = cv2.flip(frame, 0)

            if self.m_dict["h_flip"]:
                frame = cv2.flip(frame, 1)

            detections = detector.detect(frame)

            result_frame = frame
            for d in detections:
                if d["conf"] < conf_thresh:
                    continue
                x_center, y_center, box_w, box_h, box_angle = obb_of(d)

                # Save the results to the list
                result_list.append(
                    [
                        file_name_without_ext,
                        d["x1"],
                        d["y1"],
                        d["x2"],
                        d["y2"],
                        x_center,
                        y_center,
                        box_w,
                        box_h,
                        box_angle,
                        d["conf"],
                        d["class_id"],
                        d["class_name"],
                    ]
                )

                result_frame = self.drawing(
                    frame,
                    (x_center, y_center, box_w, box_h, box_angle),
                    d["conf"],
                    d["class_id"],
                )

            # Write the frame to the output video
            result_file_path = os.path.join(
                self.out_path, file_name_without_ext + "_render.png"
            )
            cv2.imwrite(result_file_path, result_frame)

            progress = (image_index + 1) / image_count if image_count > 0 else 0.0
            self.m_dict["image_progress"] = progress
            self.m_dict["image_progress_label"] = f"{image_index + 1}/{image_count}"

            # Emit progress to stdout in 10% steps
            pct = int(progress * 100)
            if pct // 10 > last_logged_pct // 10:
                last_logged_pct = pct
                print(
                    f"    {pct}% ({image_index + 1}/{image_count}) {base_name}",
                    flush=True,
                )

        # Convert the list to a DataFrame
        df_results = pd.DataFrame(
            result_list,
            columns=[
                "file_name",
                "x1",
                "y1",
                "x2",
                "y2",
                "x_center",
                "y_center",
                *OBB_COLUMNS,
                "confidence",
                "class",
                "class_name",
            ],
        )
        # Output as CSV
        df_results.to_csv(file_path, index=False)
        print(f"=== Image analysis complete -> {file_path} ===", flush=True)

        self.m_dict["analy_state"] = "Done!"
        self.m_dict["image_progress"] = 1.0
        self.m_dict["image_progress_label"] = "Done"


class file_open:
    def __init__(self):
        self.a = 1

    def get_file_path(self):
        root = tk.Tk()
        root.withdraw()  # Do not display the Tk root window

        # Show the file selection dialog
        file_path = filedialog.askopenfilename()

        return file_path

    def get_directory_path(self):
        root = tk.Tk()
        root.withdraw()  # Do not display the Tk root window

        # Show the folder selection dialog
        directory_path = filedialog.askdirectory()

        return directory_path


# Usage example
if __name__ == "__main__":
    fileopen = file_open()
    model_path = fileopen.get_file_path()
    movie_path = fileopen.get_file_path()
    output_path = fileopen.get_directory_path()
    mydict = {
        "model_path": model_path,
        "input_path": movie_path,
        "output_path": output_path,
    }
    analyzer = yolo_analysis(mydict)
    analyzer.analyze()
