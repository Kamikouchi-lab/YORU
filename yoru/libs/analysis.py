import os
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
import torch
from munkres import Munkres


class yolo_analysis:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        self.yolo_model_path = self.m_dict["model_path"]
        self.mov_path_list = self.m_dict["input_path"]
        self.out_path = self.m_dict["output_path"]
        print("init")

    def cal_id(self, pre_mat, cur_mat):
        actual_cur_num = len(cur_mat)
        actual_pre_num = len(pre_mat)

        while len(cur_mat) > len(pre_mat):
            pre_mat.append((-1000, -1000))
        while len(pre_mat) > len(cur_mat):
            cur_mat.append((-1000, -1000))

        if cur_mat is None:
            return None
        pre_mat = torch.tensor(pre_mat).type(torch.float64)
        cur_mat = torch.tensor(cur_mat).type(torch.float64)

        matrix = torch.cdist(pre_mat, cur_mat)
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

    def analyze(self):
        dpg.disable_item("analyze_btn")
        dpg.disable_item("create_movie")

        dpg.set_value("analy_time", "Estimated remaining time: calculating...")
        dpg.set_value("no_mov", "Leaving movies: calculating...")
        yolo_model = torch.hub.load(
            "./libs/yolov5", "custom", path=self.yolo_model_path, source="local"
        )

        # クラス名の取得
        class_names = (
            yolo_model.module.names
            if hasattr(yolo_model, "module")
            else yolo_model.names
        )

        movie_count = len(self.mov_path_list)
        self.m_dict["no_movies"] = f"Leaving movies: {int(movie_count)} movies"
        dpg.set_value("no_mov", self.m_dict["no_movies"])

        for self.mov_path in self.mov_path_list:
            df_results = pd.DataFrame()
            result_list = []
            video = cv2.VideoCapture(self.mov_path)
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

            # 出力動画の設定
            if self.m_dict["create_video"]:
                # 指定の出力ディレクトリに新しいファイル名を結合
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

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    self.m_dict[
                        "estimate_time"
                    ] = f"Estimated remaining time: Processing"
                    dpg.set_value("analy_time", self.m_dict["estimate_time"])
                    break

                start_time = time.time()  # 処理開始時間

                if self.m_dict["v_flip"]:
                    frame = cv2.flip(frame, 0)

                if self.m_dict["h_flip"]:
                    frame = cv2.flip(frame, 1)

                yolo_result = yolo_model(frame)

                if self.m_dict["create_video"]:
                    # 検出結果の描画
                    yolo_result.render()  # render()は検出結果を描画します
                    result_frame = yolo_result.ims[0]
                    # フレームを出力動画に書き込む
                    out.write(result_frame)
                cur_center_pos = []
                result = []
                for *box, conf, cls in yolo_result.xyxy[
                    0
                ]:  # xyxy形式（左上のx、左上のy、右下のx、右下のy、確信度、クラス）のリスト
                    x_center = (box[0].item() + box[2].item()) / 2
                    y_center = (box[1].item() + box[3].item()) / 2
                    class_name = class_names[int(cls.item())]

                    # 結果をリストに保存
                    result.append(
                        [
                            frame_count,
                            box[0].item(),
                            box[1].item(),
                            box[2].item(),
                            box[3].item(),
                            x_center,
                            y_center,
                            conf.item(),
                            cls.item(),
                            class_name,
                        ]
                    )

                    cur_center_pos.append((x_center, y_center))

                if self.m_dict["tracking_state"]:
                    # トラッキングの実装
                    id_matrix = self.cal_id(pre_center_pos, cur_center_pos)
                    cur_ids = []
                    # id_matrixを今のフレームで並び替える
                    id_matrix.sort(key=lambda x: x[1] if x[1] >= 0 else float("inf"))
                    for ids in id_matrix:
                        if ids[0] == -1 or ids[1] == -1:
                            cur_ids.append(global_counter)
                            global_counter += 1
                        else:
                            cur_ids.append(pre_ids[ids[0]])
                    # リストの結合
                    result = [x + [y] for x, y in zip(result, cur_ids)]
                    # print(result)

                    pre_ids = cur_ids
                    pre_center_pos = cur_center_pos

                frame_count += 1
                result_list = result_list + result

                end_time = time.time()  # 処理終了時間
                process_time = end_time - start_time  # このフレームの処理時間
                process_times.append(process_time)  # 処理時間をリストに保存

                # 平均フレーム処理時間
                avg_process_time = sum(process_times) / len(process_times)

                # 残りのフレーム数
                remaining_frames = total_frames - frame_count

                # 残りの処理時間の見積もり
                remaining_time_estimate = avg_process_time * remaining_frames
                self.m_dict[
                    "estimate_time"
                ] = f"Estimated remaining time: {int(remaining_time_estimate)} seconds"
                dpg.set_value("analy_time", self.m_dict["estimate_time"])

            # リストをデータフレームに変換
            # print(result_list)
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
            # csvとして出力
            df_results.to_csv(file_path, index=False)

            video.release()
            if self.m_dict["create_video"]:
                out.release()
            movie_count = movie_count - 1
            self.m_dict["no_movies"] = f"Leaving movies: {int(movie_count)} movies"
            dpg.set_value("no_mov", self.m_dict["no_movies"])

        self.m_dict["estimate_time"] = "Estimated remaining time: none"
        self.m_dict["no_movies"] = "Leaving movies: none"
        dpg.set_value("analy_time", self.m_dict["estimate_time"])
        dpg.set_value("no_mov", self.m_dict["no_movies"])
        dpg.enable_item("analyze_btn")
        dpg.enable_item("create_movie")

    def create_video(self):
        dpg.set_value("cr_analy_time", "Estimated remaining time: calculating...")
        yolo_model = torch.hub.load(
            "./libs/yolov5", "custom", path=self.yolo_model_path, source="local"
        )

        # ファイル名の取得（拡張子なし）
        base_name = os.path.basename(self.mov_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        # 指定の出力ディレクトリに新しいファイル名を結合
        out_movie_path = os.path.join(
            self.out_path, file_name_without_ext + "_render_" + ".mp4"
        )

        # 入力動画の読み込み
        cap = cv2.VideoCapture(self.mov_path)
        # 出力動画の設定
        out = cv2.VideoWriter(
            out_movie_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

        # ビデオのフレーム数を取得
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        process_times = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            start_time = time.time()  # 処理開始時間
            frame = cv2.flip(frame, 0)

            if not ret:
                self.m_dict[
                    "cr_estimate_time"
                ] = f"Estimated remaining time: Processing"
                dpg.set_value("cr_analy_time", self.m_dict["cr_estimate_time"])
                break

            # オブジェクト検出
            result = yolo_model(frame)

            # 検出結果の描画
            result.render()  # render()は検出結果を描画します

            result_frame = result.ims[0]

            # フレームを出力動画に書き込む
            out.write(result_frame)

            frame_count += 1

            end_time = time.time()  # 処理終了時間
            process_time = end_time - start_time  # このフレームの処理時間
            process_times.append(process_time)  # 処理時間をリストに保存

            # 平均フレーム処理時間
            avg_process_time = sum(process_times) / len(process_times)

            # 残りのフレーム数
            remaining_frames = total_frames - frame_count

            # 残りの処理時間の見積もり
            remaining_time_estimate = avg_process_time * remaining_frames
            self.m_dict[
                "cr_estimate_time"
            ] = f"Estimated remaining time: {int(remaining_time_estimate)} seconds"
            dpg.set_value("cr_analy_time", self.m_dict["cr_estimate_time"])

        cap.release()
        out.release()
        self.m_dict["cr_estimate_time"] = "Estimated remaining time: none"
        dpg.set_value("cr_analy_time", self.m_dict["cr_estimate_time"])


class yolo_analysis_image:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        self.yolo_model_path = self.m_dict["model_path"]
        self.img_path_list = self.m_dict["input_path_image"]
        self.out_path = self.m_dict["output_path"]
        print("init")

    def analyze_image(self):
        dpg.disable_item("analyze_img_btn")

        # dpg.set_value("analy_time", "Estimated remaining time: calculating...")
        # dpg.set_value("no_mov", "Leaving movies: calculating...")
        yolo_model = torch.hub.load(
            "./libs/yolov5", "custom", path=self.yolo_model_path, source="local"
        )

        # クラス名の取得
        class_names = (
            yolo_model.module.names
            if hasattr(yolo_model, "module")
            else yolo_model.names
        )

        image_count = len(self.img_path_list)

        df_results = pd.DataFrame()
        result_list = []
        # 指定の出力ディレクトリに新しいファイル名を結合
        file_path = os.path.join(self.out_path, "image_analysis_results" + ".csv")

        for self.img_path in self.img_path_list:
            base_name = os.path.basename(self.img_path)
            file_name_without_ext = os.path.splitext(base_name)[0]

            frame = cv2.imread(self.img_path)
            if self.m_dict["v_flip"]:
                frame = cv2.flip(frame, 0)

            if self.m_dict["h_flip"]:
                frame = cv2.flip(frame, 1)

            yolo_result = yolo_model(frame)
            yolo_result.render()
            result_frame = yolo_result.ims[0]
            # フレームを出力動画に書き込む
            result_file_path = os.path.join(
                self.out_path, file_name_without_ext + "_render.png"
            )
            cv2.imwrite(result_file_path, result_frame)

            for *box, conf, cls in yolo_result.xyxy[
                0
            ]:  # xyxy形式（左上のx、左上のy、右下のx、右下のy、確信度、クラス）のリスト
                x_center = (box[0].item() + box[2].item()) / 2
                y_center = (box[1].item() + box[3].item()) / 2
                class_name = class_names[int(cls.item())]

                # 結果をリストに保存
                result_list.append(
                    [
                        file_name_without_ext,
                        box[0].item(),
                        box[1].item(),
                        box[2].item(),
                        box[3].item(),
                        x_center,
                        y_center,
                        conf.item(),
                        cls.item(),
                        class_name,
                    ]
                )

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

        # dpg.set_value("no_mov", self.m_dict["no_movies"])

        # dpg.set_value("analy_time", self.m_dict["estimate_time"])
        # dpg.set_value("no_mov", self.m_dict["no_movies"])
        dpg.enable_item("analyze_img_btn")


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
