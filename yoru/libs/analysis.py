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

from yoru.libs.drawing import get_colormap
from yoru.libs.plugins import DEFAULT_CONF_THRESH, get_detector

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
        for res_frame_no, *res_box, res_x_center, res_y_center, res_conf, res_cls , res_class_name in result:
        
            # print(results)
            label = f"{res_class_name} {res_conf:.2f}"

            cv2.rectangle(
                img,
                pt1=(int(res_box[0]), int(res_box[1])),
                pt2=(int(res_box[2]), int(res_box[3])),
                color=self.colormap[int(res_cls)],
                thickness=4,
                lineType=cv2.LINE_4,
                shift=0,
            )
            cv2.putText(
                img,
                text=label,
                org=(int(res_box[0]), int(res_box[1]) - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=self.colormap[int(res_cls)],
                thickness=5,
                lineType=cv2.LINE_4,
            )
        return img

    def tracking_drawing(self, result, img):
        for res_frame_no, *res_box, res_x_center, res_y_center, res_conf, res_cls, res_class_name, tracking_id in result:
            label = f"{res_class_name} {res_conf:.2f}"
            label += f" id:{tracking_id}"

            cv2.rectangle(
                img,
                pt1=(int(res_box[0]), int(res_box[1])),
                pt2=(int(res_box[2]), int(res_box[3])),
                color=self.colormap[int(res_cls)],
                thickness=4,
                lineType=cv2.LINE_4,
                shift=0,
            )
            cv2.putText(
                img,
                text=label,
                org=(int(res_box[0]), int(res_box[1]) - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=self.colormap[int(res_cls)],
                thickness=5,
                lineType=cv2.LINE_4,
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

        # クラス名の取得
        self.class_names = detector.names

        self.colormap = get_colormap(self.class_names, "gist_rainbow")

        movie_count = len(self.mov_path_list)
        self.m_dict["no_movies"] = f"Leaving movies: {int(movie_count)} movies"

        for self.mov_path in self.mov_path_list:
            df_results = pd.DataFrame()
            result_list = []
            video = cv2.VideoCapture(self.mov_path)
            out = None
            frame_count = 0

            # トラッキング用
            pre_ids = []
            pre_center_pos = []  # 以前の位置情報を入力する
            global_counter = 0

            # ファイル名の取得（拡張子なし）
            base_name = os.path.basename(self.mov_path)
            file_name_without_ext = os.path.splitext(base_name)[0]

            # 指定の出力ディレクトリに新しいファイル名を結合
            file_path = os.path.join(self.out_path, file_name_without_ext + ".csv")

            try:
                # 出力動画の設定
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

                # ビデオのフレーム数を取得
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                process_times = []

                result_list = []
                pre_ids = []
                self.m_dict["movie_progress"] = 0.0

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
                        x_center = (d["x1"] + d["x2"]) / 2
                        y_center = (d["y1"] + d["y2"]) / 2

                        entry = [
                            frame_count,
                            d["x1"],
                            d["y1"],
                            d["x2"],
                            d["y2"],
                            x_center,
                            y_center,
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

                        # 除外クラスはtracking_id=-1として追加
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

                    end_time = time.time()
                    process_time = end_time - start_time
                    process_times.append(process_time)

                    avg_process_time = sum(process_times) / len(process_times)
                    remaining_frames = total_frames - frame_count
                    remaining_time_estimate = avg_process_time * remaining_frames
                    self.m_dict["estimate_time"] = (
                        f"Estimated remaining time: {int(remaining_time_estimate)} seconds"
                    )

                # リストをデータフレームに変換
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
                    x_center = (d["x1"] + d["x2"]) / 2
                    y_center = (d["y1"] + d["y2"]) / 2
                    result.append([
                        frame_count,
                        d["x1"], d["y1"], d["x2"], d["y2"],
                        x_center, y_center,
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
        # print(results)
        label = f"{self.class_names[int(cls)]} {conf:.2f}"
        # label = f"{name} {conf:.2f}

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

        # クラス名の取得
        self.class_names = detector.names

        self.colormap = get_colormap(self.class_names, "gist_rainbow")

        image_count = len(self.img_path_list)

        df_results = pd.DataFrame()
        result_list = []
        # 指定の出力ディレクトリに新しいファイル名を結合
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
                x_center = (d["x1"] + d["x2"]) / 2
                y_center = (d["y1"] + d["y2"]) / 2

                # 結果をリストに保存
                result_list.append(
                    [
                        file_name_without_ext,
                        d["x1"],
                        d["y1"],
                        d["x2"],
                        d["y2"],
                        x_center,
                        y_center,
                        d["conf"],
                        d["class_id"],
                        d["class_name"],
                    ]
                )

                result_frame = self.drawing(
                    frame,
                    [d["x1"], d["y1"], d["x2"], d["y2"]],
                    d["conf"],
                    d["class_id"],
                )

            # フレームを出力動画に書き込む
            result_file_path = os.path.join(
                self.out_path, file_name_without_ext + "_render.png"
            )
            cv2.imwrite(result_file_path, result_frame)

            progress = (image_index + 1) / image_count if image_count > 0 else 0.0
            self.m_dict["image_progress"] = progress
            self.m_dict["image_progress_label"] = f"{image_index + 1}/{image_count}"

        # リストをデータフレームに変換
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
                "confidence",
                "class",
                "class_name",
            ],
        )
        # csvとして出力
        df_results.to_csv(file_path, index=False)

        self.m_dict["analy_state"] = "Done!"
        self.m_dict["image_progress"] = 1.0
        self.m_dict["image_progress_label"] = "Done"


class file_open:
    def __init__(self):
        self.a = 1

    def get_file_path(self):
        root = tk.Tk()
        root.withdraw()  # Tkのルートウィンドウを表示しない

        # ファイル選択ダイアログを表示
        file_path = filedialog.askopenfilename()

        return file_path

    def get_directory_path(self):
        root = tk.Tk()
        root.withdraw()  # Tkのルートウィンドウを表示しない

        # フォルダ選択ダイアログを表示
        directory_path = filedialog.askdirectory()

        return directory_path


# 使用例
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
