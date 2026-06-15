import os
import time
import tkinter as tk
from tkinter import filedialog

import cv2
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from munkres import Munkres

from yoru.libs.yolo_wrapper import load_yolo_model


class yolo_analysis:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        self.yolo_model_path = self.m_dict["model_path"]
        self.mov_path_list = self.m_dict["input_path"]
        self.out_path = self.m_dict["output_path"]
        print("init")

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

    def get_colormap(self, label_names, colormap_name):
        colormap = {}
        cmap = plt.get_cmap(colormap_name)
        label_ids = list(range(len(label_names)))
        for i in range(len(label_ids)):
            rgb = [int(d) for d in np.array(cmap(float(i) / len(label_ids))) * 255][:3]
            colormap[label_ids[i]] = tuple(rgb)

        return colormap

    def drawing(self, result, img):
        for res_frame_no, *res_box, res_x_center, res_y_center, res_conf, res_cls , res_class_name in result:
        
            # print(results)
            label = f"{res_class_name} {res_conf:.2f}"
            # label = f"{name} {conf:.2f}

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
        # print(label)
        # self.m_dict["yolo_detection_farme"] = img
        # cv2.imshow('prj_view2', img)
        return img

    def tracking_drawing(self, result, img):
        for res_frame_no, *res_box, res_x_center, res_y_center, res_conf, res_cls , res_class_name, tracking_id in result:
        
            # print(results)
            label = f"{res_class_name} {res_conf:.2f}"
            label += f" id:{tracking_id}"
            # label = f"{name} {conf:.2f}

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
        # print(label)
        # self.m_dict["yolo_detection_farme"] = img
        # cv2.imshow('prj_view2', img)
        return img

    def analyze(self):
        dpg.disable_item("analyze_btn")
        dpg.disable_item("create_movie")

        dpg.set_value("analy_time", "Estimated remaining time: calculating...")
        dpg.set_value("no_mov", "Leaving movies: calculating...")
        yolo_model = load_yolo_model(self.yolo_model_path)

        # Get class names
        self.class_names = yolo_model.names

        self.colormap = self.get_colormap(self.class_names, "gist_rainbow")

        movie_count = len(self.mov_path_list)
        total_movies = movie_count
        self.m_dict["no_movies"] = f"Leaving movies: {int(movie_count)} movies"
        dpg.set_value("no_mov", self.m_dict["no_movies"])
        print(f"=== Start movie analysis: {total_movies} movie(s) ===", flush=True)

        for movie_index, self.mov_path in enumerate(self.mov_path_list, start=1):
            df_results = pd.DataFrame()
            result_list = []
            video = cv2.VideoCapture(self.mov_path)
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

            # Output video settings
            if self.m_dict["create_video"]:
                # Join the new file name with the specified output directory
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
            dpg.set_value("movie_progress_bar", 0.0)
            dpg.configure_item("movie_progress_bar", overlay="0%")

            print(
                f"[{movie_index}/{total_movies}] Analyzing '{base_name}' "
                f"({total_frames} frames)...",
                flush=True,
            )
            last_logged_pct = -10  # For progress logging to stdout (output in 10% steps)

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    self.m_dict["estimate_time"] = (
                        f"Estimated remaining time: Processing"
                    )
                    dpg.set_value("analy_time", self.m_dict["estimate_time"])
                    break

                start_time = time.time()  # Processing start time

                if self.m_dict["v_flip"]:
                    frame = cv2.flip(frame, 0)

                if self.m_dict["h_flip"]:
                    frame = cv2.flip(frame, 1)

                yolo_result = yolo_model(frame)

                # yolo_result.render()  # render() draws the detection results
                # result_frame = yolo_result.ims[0]
                # Write the frame to the output video

                cur_center_pos = []
                result = []
                result_excluded = []  # Detection results for classes excluded from tracking
                exclude_classes = list(self.m_dict.get("tracking_exclude_classes", []))
                for *box, conf, cls in yolo_result.xyxy[0]:  # List in xyxy format (top-left x, top-left y, bottom-right x, bottom-right y, confidence, class)
                    if conf.item() <  self.m_dict["threshold"]:
                        break
                    x_center = (box[0].item() + box[2].item()) / 2
                    y_center = (box[1].item() + box[3].item()) / 2
                    class_name = self.class_names[int(cls.item())]

                    entry = [
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

                    # If tracking is ON and the class is excluded, add to a separate list
                    if self.m_dict["tracking_state"] and int(cls.item()) in exclude_classes:
                        result_excluded.append(entry)
                    else:
                        result.append(entry)
                        cur_center_pos.append((x_center, y_center))

                if self.m_dict["tracking_state"]:
                    # Tracking implementation
                    id_matrix = self.cal_id(pre_center_pos, cur_center_pos)
                    cur_ids = []
                    # if id_matrix is None:
                    # print(cur_center_pos)
                    if id_matrix is not None:
                        # Sort id_matrix by the current frame
                        id_matrix.sort(
                            key=lambda x: x[1] if x[1] >= 0 else float("inf")
                        )
                        for ids in id_matrix:
                            if ids[0] == -1 or ids[1] == -1:
                                cur_ids.append(global_counter)
                                global_counter += 1
                                # print(str(global_counter))
                            else:
                                if 0 <= ids[0] < len(pre_ids):
                                    cur_ids.append(pre_ids[ids[0]])
                                else:
                                    # Handling for out-of-range cases (e.g., assign a new ID)
                                    cur_ids.append(global_counter)
                                    global_counter += 1
                                    # print("b")
                        # Combine the lists
                        result = [x + [y] for x, y in zip(result, cur_ids)]
                        # print(result)

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
                dpg.set_value("movie_progress_bar", progress)
                dpg.configure_item("movie_progress_bar", overlay=f"{int(progress * 100)}%")

                # Emit progress to stdout in 10% steps
                pct = int(progress * 100)
                if pct // 10 > last_logged_pct // 10:
                    last_logged_pct = pct
                    print(
                        f"    [{movie_index}/{total_movies}] {base_name}: "
                        f"{pct}% ({frame_count}/{total_frames} frames)",
                        flush=True,
                    )

                end_time = time.time()  # Processing end time
                process_time = end_time - start_time  # Processing time for this frame
                process_times.append(process_time)  # Save the processing time to the list

                # Average frame processing time
                avg_process_time = sum(process_times) / len(process_times)

                # Number of remaining frames
                remaining_frames = total_frames - frame_count

                # Estimate of the remaining processing time
                remaining_time_estimate = avg_process_time * remaining_frames
                self.m_dict["estimate_time"] = (
                    f"Estimated remaining time: {int(remaining_time_estimate)} seconds"
                )
                dpg.set_value("analy_time", self.m_dict["estimate_time"])

            # Convert the list to a DataFrame
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
            # Output as CSV
            df_results.to_csv(file_path, index=False)
            print(
                f"[{movie_index}/{total_movies}] Done '{base_name}' "
                f"-> {file_path}",
                flush=True,
            )

            video.release()
            if self.m_dict["create_video"]:
                out.release()
            movie_count = movie_count - 1
            self.m_dict["no_movies"] = f"Leaving movies: {int(movie_count)} movies"
            dpg.set_value("no_mov", self.m_dict["no_movies"]),

        self.m_dict["estimate_time"] = "Estimated remaining time: none"
        self.m_dict["no_movies"] = "Leaving movies: none"
        dpg.set_value("analy_time", self.m_dict["estimate_time"])
        dpg.set_value("no_mov", self.m_dict["no_movies"])
        dpg.set_value("movie_progress_bar", 1.0)
        dpg.configure_item("movie_progress_bar", overlay="Done")
        dpg.enable_item("analyze_btn")
        dpg.enable_item("create_movie")
        print("=== Movie analysis complete ===", flush=True)

    def create_video(self):
        dpg.set_value("cr_analy_time", "Estimated remaining time: calculating...")
        yolo_model = load_yolo_model(self.yolo_model_path)

        # Get the file name (without extension)
        base_name = os.path.basename(self.mov_path)
        file_name_without_ext = os.path.splitext(base_name)[0]

        # Join the new file name with the specified output directory
        out_movie_path = os.path.join(
            self.out_path, file_name_without_ext + "_render_" + ".mp4"
        )

        # Load the input video
        cap = cv2.VideoCapture(self.mov_path)
        # Output video settings
        out = cv2.VideoWriter(
            out_movie_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            cap.get(cv2.CAP_PROP_FPS),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )

        # Get the number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        process_times = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            start_time = time.time()  # Processing start time
            frame = cv2.flip(frame, 0)

            if not ret:
                self.m_dict["cr_estimate_time"] = (
                    f"Estimated remaining time: Processing"
                )
                dpg.set_value("cr_analy_time", self.m_dict["cr_estimate_time"])
                break

            # Object detection
            result = yolo_model(frame)

            # Draw the detection results
            result.render()  # render() draws the detection results

            result_frame = result.ims[0]

            # Write the frame to the output video
            out.write(result_frame)

            frame_count += 1

            end_time = time.time()  # Processing end time
            process_time = end_time - start_time  # Processing time for this frame
            process_times.append(process_time)  # Save the processing time to the list

            # Average frame processing time
            avg_process_time = sum(process_times) / len(process_times)

            # Number of remaining frames
            remaining_frames = total_frames - frame_count

            # Estimate of the remaining processing time
            remaining_time_estimate = avg_process_time * remaining_frames
            self.m_dict["cr_estimate_time"] = (
                f"Estimated remaining time: {int(remaining_time_estimate)} seconds"
            )
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
        # print(label)
        # self.m_dict["yolo_detection_farme"] = img
        # cv2.imshow('prj_view2', img)
        return img

    def get_colormap(self, label_names, colormap_name):
        colormap = {}
        cmap = plt.get_cmap(colormap_name)
        label_ids = list(range(len(label_names)))
        for i in range(len(label_ids)):
            rgb = [int(d) for d in np.array(cmap(float(i) / len(label_ids))) * 255][:3]
            colormap[label_ids[i]] = tuple(rgb)

        return colormap

    def analyze_image(self):
        dpg.disable_item("analyze_img_btn")

        dpg.set_value("analy_state", "Analyzing...")
        dpg.set_value("image_progress_bar", 0.0)
        dpg.configure_item("image_progress_bar", overlay="0%")

        yolo_model = load_yolo_model(self.yolo_model_path)

        # Get class names
        self.class_names = yolo_model.names

        self.colormap = self.get_colormap(self.class_names, "gist_rainbow")

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
                raise IOError(f"Could not read image file: {self.img_path}")
            if self.m_dict["v_flip"]:
                frame = cv2.flip(frame, 0)

            if self.m_dict["h_flip"]:
                frame = cv2.flip(frame, 1)

            yolo_result = yolo_model(frame)

            for *box, conf, cls in yolo_result.xyxy[
                0
            ]:  # List in xyxy format (top-left x, top-left y, bottom-right x, bottom-right y, confidence, class)
                x_center = (box[0].item() + box[2].item()) / 2
                y_center = (box[1].item() + box[3].item()) / 2
                class_name = self.class_names[int(cls.item())]

                # Save the results to the list
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

                result_frame = self.drawing(frame, box, conf, cls)

            # Write the frame to the output video
            result_file_path = os.path.join(
                self.out_path, file_name_without_ext + "_render.png"
            )
            cv2.imwrite(result_file_path, result_frame)

            progress = (image_index + 1) / image_count if image_count > 0 else 0.0
            dpg.set_value("image_progress_bar", progress)
            dpg.configure_item("image_progress_bar", overlay=f"{image_index + 1}/{image_count}")

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
                "confidence",
                "class",
                "class_name",
            ],
        )
        # Output as CSV
        df_results.to_csv(file_path, index=False)
        print(f"=== Image analysis complete -> {file_path} ===", flush=True)

        dpg.set_value("analy_state", "Done!")
        dpg.set_value("image_progress_bar", 1.0)
        dpg.configure_item("image_progress_bar", overlay="Done")
        dpg.enable_item("analyze_img_btn")


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
