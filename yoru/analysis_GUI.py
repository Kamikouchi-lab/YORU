import os
import subprocess
import sys
import time
from cProfile import label
from multiprocessing import Manager, Process

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

sys.path.append("../yoru")


from yoru.libs.analysis import yolo_analysis, yolo_analysis_image
# try:
from yoru.libs.file_operation_analysis import file_dialog_tk
from yoru.libs.init_analysis import init_analysis


class analyze_GUI:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict
        self.fd_tk = file_dialog_tk(self.m_dict)

        self.file_path = "./web/image/YORU_logo.png"

        if self.file_path:
            print("File: " + self.file_path)
        else:
            print("Open-file dialog")

        self.vid = cv2.imread(self.file_path)
        self.height, self.width, _ = self.vid.shape
        self.framecount = 1
        self.current_frame_num = 1
        self.frame = self.vid
        self.process_frame()
        self.grab_count = 0
        self.speed = 1

    def process_frame(self):
        if self.width >= self.height:
            self.im_win_width = 400
            self.im_win_height = self.height * (400 / self.width)
        else:
            self.im_win_width = self.width * (400 / self.height)
            self.im_win_height = 400

        # 画面のフリップ
        if self.m_dict["v_flip"]:
            self.frame = cv2.flip(self.frame, 0)
        else:
            pass

        if self.m_dict["h_flip"]:
            self.frame = cv2.flip(self.frame, 1)
        else:
            pass

        # フレームのリサイズ
        self.frame_re = cv2.resize(
            self.frame, dsize=(int(self.im_win_width), int(self.im_win_height))
        )
        # 新しいフレームの作成 (全て黒で埋められたフレーム)
        base_frame = np.zeros((400, 400, 3), np.uint8)
        # リサイズしたフレームを新しいフレームの中央に配置
        h, w = self.frame_re.shape[:2]
        base_frame[
            int(400 / 2 - h / 2) : int(400 / 2 + h / 2),
            int(400 / 2 - w / 2) : int(400 / 2 + w / 2),
            :,
        ] = self.frame_re
        # 更新
        self.frame_re = base_frame

    def startDPG(self):
        dpg.create_context()
        dpg.configure_app(
            init_file="./config/custom_layout_analysis.ini",
            docking=True,
            docking_space=True,
        )

        dpg.create_viewport(title="YORU - Video Analysis", width=800, height=800)

        # GUI-settings
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=400,
                height=400,
                default_value=self.frame_to_data(self.frame_re),
                tag="imwin_tag0",
                format=dpg.mvFormat_Float_rgb,
            )
            dpg.add_raw_texture(
                width=400,
                height=400,
                default_value=self.frame_to_data(self.frame_re),
                tag="imwin_tag1",
                format=dpg.mvFormat_Float_rgb,
            )

        imager_window1 = dpg.generate_uuid()
        imager_window2 = dpg.generate_uuid()

        # imager-window
        # images analysis window
        with dpg.window(label="Analyzing Images", id=imager_window2):
            dpg.add_text(default_value="Analyzing Images")
            with dpg.group(horizontal=True):
                dpg.add_text(label="Model path", default_value="Model Path")
                dpg.add_input_text(
                    tag="Model_path_2", readonly=True, hint="Path/to/model"
                )
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.fd_tk.model_file_open(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_text(label="Images path", default_value="Images Path")
                dpg.add_input_text(
                    tag="input_image_path", readonly=True, hint="Path/to/images"
                )
                dpg.add_button(
                    label="Select Images",
                    callback=lambda: self.image_select_bt(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_text(label="Output Dir", default_value="Result Directory")
                dpg.add_input_text(
                    tag="Output_Directory_Path2",
                    readonly=True,
                    hint="Path/to/result/directory",
                )
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.fd_tk.Out_dir_open(),
                    enabled=True,
                )
            dpg.add_separator()
            dpg.add_text(label="title", default_value="Preview")
            dpg.add_image("imwin_tag1", width=400, height=400)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Previous image",
                    tag="previous_image",
                    callback=lambda: self.previous_image_bt(),
                )
                dpg.add_text(label="space1", default_value="     ")
                dpg.add_button(
                    label="Next image",
                    tag="next_image",
                    callback=lambda: self.next_image_bt(),
                )
                dpg.add_text(tag="image_num_state", default_value="    none")
            dpg.add_separator()
            dpg.add_text(label="title2", default_value="start analyzing")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="YOLO analysis images",
                    tag="analyze_img_btn",
                    width=100,
                    height=30,
                    callback=lambda: self.analyze_image(),
                    enabled=True,
                )
                with dpg.group(horizontal=False):
                    dpg.add_text(
                        label="Analyze state", tag="analy_state", default_value="none"
                    )
            dpg.add_separator()
            with dpg.group(horizontal=False):
                dpg.add_button(
                    label="Quit",
                    tag="quit_btn_2",
                    callback=lambda: self.quit_cb(),
                    enabled=True,
                )
                dpg.add_button(
                    label="Back to home",
                    tag="home_btn_2",
                    callback=lambda: self.home_cb(),
                    enabled=True,
                )

        # movie analysis window
        with dpg.window(label="Analyzing Movies", id=imager_window1):
            dpg.add_text(default_value="Analyzing Movies")
            with dpg.group(horizontal=True):
                dpg.add_text(label="Model path", default_value="Model Path")
                dpg.add_input_text(
                    tag="Model_path", readonly=True, hint="Path/to/model"
                )
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.fd_tk.model_file_open(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_text(label="Movie path", default_value="Movie Path")
                dpg.add_input_text(
                    tag="Input file Path", readonly=True, hint="Path/to/movies"
                )
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.movie_select_bt(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_text(label="Output Dir", default_value="Result Directory")
                dpg.add_input_text(
                    tag="Output_Directory_Path",
                    readonly=True,
                    hint="Path/to/result/directory",
                )
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.fd_tk.Out_dir_open(),
                    enabled=True,
                )
            dpg.add_separator()
            dpg.add_text(label="title", default_value="Preview")
            dpg.add_image("imwin_tag0", width=400, height=400)
            dpg.add_slider_int(
                label=" Frame",
                default_value=0,
                min_value=0,
                max_value=self.framecount - 2,
                tag="frame_bar",
                width=400,
                callback=lambda: self.slide_bar_cb(),
                enabled=False,
            )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="streaming movie",
                    default_value=False,
                    tag="streamingChkBox",
                    callback=lambda: self.stream_cb(),
                    enabled=False,
                )
                dpg.add_text(
                    label="speed_explanation", default_value="    Streaming speed"
                )
                dpg.add_combo(
                    items=[1, 2, 5, 10, 20, 50, 100, 200, 500],
                    tag="speed_list",
                    default_value=1,
                    width=150,
                    callback=lambda: self.list_of_speed(),
                )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Vertical flip",
                    tag="v_flip_state",
                    callback=lambda: self.v_flip_cb(),
                )
                dpg.add_button(
                    label="Horizontal flip",
                    tag="h_flip_state",
                    callback=lambda: self.h_flip_cb(),
                )

                dpg.add_text(label="threshold", default_value="      Confidence threshold")
                dpg.add_input_text(
                    tag="conf_threshold",
                    default_value=self.m_dict["threshold"],
                    width=150,
                    callback=lambda: self.in_thresh(),
                )
            dpg.add_separator()
            dpg.add_text(label="title2", default_value="start analyzing")
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="YOLO analysis",
                    tag="analyze_btn",
                    width=100,
                    height=30,
                    callback=lambda: self.analyze_movie(),
                    enabled=True,
                )
                dpg.add_checkbox(
                    label="Create videos    ",
                    tag="create_movie",
                    default_value=self.m_dict["create_video"],
                    callback=lambda: self.create_condition(),
                )
                dpg.add_checkbox(
                    label="Tracking algorithm       ",
                    tag="tracking_state",
                    default_value=self.m_dict["tracking_state"],
                    callback=lambda: self.tracking_condition(),
                )
                with dpg.group(horizontal=False):
                    dpg.add_text(
                        label="No of movies",
                        tag="no_mov",
                        default_value=self.m_dict["no_movies"],
                    )
                    dpg.add_text(
                        label="Analyze time",
                        tag="analy_time",
                        default_value=self.m_dict["estimate_time"],
                    )
            dpg.add_separator()
            with dpg.group(horizontal=False):
                dpg.add_button(
                    label="Quit",
                    tag="quit_btn",
                    callback=lambda: self.quit_cb(),
                    enabled=True,
                )
                dpg.add_button(
                    label="Back to home",
                    tag="home_btn",
                    callback=lambda: self.home_cb(),
                    enabled=True,
                )

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def run(self):
        self.startDPG()
        while dpg.is_dearpygui_running():
            self.plot_callback()
            dpg.render_dearpygui_frame()
            if self.m_dict["quit"]:  # <-- this line was modified
                if self.m_dict["back_to_home"]:
                    # subprocess.call(["python", "./yoru/app.py"])
                    from yoru import app as YORU

                    YORU.main()
                dpg.destroy_context()
                break

    def plot_callback(self) -> None:
        if dpg.get_value("streamingChkBox"):
            if int(self.speed) + dpg.get_value("frame_bar") > self.framecount - 2:
                dpg.set_value("frame_bar", 0)
            else:
                dpg.set_value("frame_bar", int(self.speed) + dpg.get_value("frame_bar"))
            self.slide_bar_cb()

    def slide_bar_cb(self):
        self.current_frame_num = dpg.get_value("frame_bar")
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        self.status, self.frame = self.vid.read()
        self.process_frame()

        dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))

    def file_open(self):
        if self.file_path:
            print("File: " + self.file_path)
        else:
            print("Failed open files")

        self.vid = cv2.VideoCapture(self.file_path)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.framecount = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0
        self.status, self.frame = self.vid.read()
        self.process_frame()
        print("Movie size: ", self.width, self.height)
        dpg.configure_item("frame_bar", max_value=self.framecount - 2)
        dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))
        dpg.enable_item("streamingChkBox")
        dpg.enable_item("frame_bar")

    def file_open_image(self):
        if self.file_path_image:
            print("File: " + self.file_path_image)
        else:
            print("Failed open image")
        self.frame = cv2.imread(self.file_path_image)
        self.height, self.width, _ = self.frame.shape
        self.process_frame()
        print("Image size: ", self.width, self.height)
        dpg.set_value("imwin_tag1", self.frame_to_data(self.frame_re))

    def stream_cb(self):
        if dpg.get_value("streamingChkBox"):
            dpg.disable_item("frame_bar")
            dpg.disable_item("v_flip_state")
            dpg.disable_item("h_flip_state")
        else:
            dpg.enable_item("frame_bar")
            dpg.enable_item("v_flip_state")
            dpg.enable_item("h_flip_state")
        pass

    def frame_to_data(self, frame):
        # raw image streaming
        frame_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.texture_data = np.true_divide(frame_data.ravel(), 255.0)
        data = np.asfarray(self.texture_data.ravel(), dtype="f")
        return data

    def list_of_speed(self):
        tf = dpg.get_value("speed_list")
        self.speed = tf

    def movie_select_bt(self):
        self.fd_tk.input_file_open()
        self.file_path = self.m_dict["input_path"][0]
        self.file_open()

    def image_select_bt(self):
        self.fd_tk.input_file_open_image()
        self.current_image_num = 0
        print(self.m_dict["input_path_image"])
        self.file_path_image = self.m_dict["input_path_image"][
            int(self.current_image_num)
        ]
        self.image_num = len(self.m_dict["input_path_image"]) - 1
        print(self.image_num)
        self.file_open_image()

    def v_flip_cb(self):
        self.status, self.frame = self.vid.read()
        if self.m_dict["v_flip"]:
            self.m_dict["v_flip"] = False
        else:
            self.m_dict["v_flip"] = True
        self.process_frame()
        dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))

    def h_flip_cb(self):
        self.status, self.frame = self.vid.read()
        if self.m_dict["h_flip"]:
            self.m_dict["h_flip"] = False
        else:
            self.m_dict["h_flip"] = True
        self.process_frame()
        dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))

    def previous_image_bt(self):
        if self.current_image_num <= 0:
            self.current_image_num = self.image_num
        else:
            self.current_image_num -= 1

        self.image_state_des = (
            "    " + str(self.current_image_num + 1) + "/" + str(self.image_num + 1)
        )
        dpg.set_value("image_num_state", self.image_state_des)
        self.file_path_image = self.m_dict["input_path_image"][
            int(self.current_image_num)
        ]
        self.file_open_image()

    def next_image_bt(self):
        if self.current_image_num >= self.image_num:
            self.current_image_num = 0
        else:
            self.current_image_num += 1
        self.image_state_des = (
            "    " + str(self.current_image_num + 1) + "/" + str(self.image_num + 1)
        )
        dpg.set_value("image_num_state", self.image_state_des)
        self.file_path_image = self.m_dict["input_path_image"][
            int(self.current_image_num)
        ]
        self.file_open_image()

    def quit_cb(self):
        print("quit_pushed")
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def home_cb(self):
        print("Back home")
        self.m_dict["back_to_home"] = True
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def analyze_movie(self):
        print("Start analyzing ....")
        self.yolo_analysis = yolo_analysis(self.m_dict)
        self.yolo_analysis.analyze()
        print("Analysis complete!!")

    def analyze_image(self):
        print("Start analyzing ....")
        self.yolo_analysis = yolo_analysis_image(self.m_dict)
        self.yolo_analysis.analyze_image()
        print("Analysis complete!!")

    def create_condition(self):
        tf = dpg.get_value("create_movie")
        self.m_dict["create_video"] = tf

    def tracking_condition(self):
        tf = dpg.get_value("tracking_state")
        self.m_dict["tracking_state"] = tf
    
    def in_thresh(self):
        tf = dpg.get_value("conf_threshold")
        self.m_dict["threshold"] = float(tf)

    def __del__(self):
        print("=== GUI window quit ===")


def main():
    with Manager() as manager:
        d = manager.dict()

        # initialize m_dict with init_analysis
        init = init_analysis(m_dict=d)

        gui = analyze_GUI(m_dict=d)
        process_pool = []
        prc_gui = Process(target=gui.run)
        process_pool.append(prc_gui)  # <-- this line was added
        prc_gui.start()
        prc_gui.join()


if __name__ == "__main__":
    main()
