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
from yoru.libs.yolo_wrapper import load_yolo_model


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

        dpg.create_viewport(title="YORU - Video Analysis", width=900, height=860)

        # Theme
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # Backgrounds
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,              (18, 24, 42),    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,               (22, 30, 52),    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,               (22, 30, 52),    category=dpg.mvThemeCat_Core)
                # Title bar
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg,               (25, 70, 130),   category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,         (35, 95, 165),   category=dpg.mvThemeCat_Core)
                # Tabs
                dpg.add_theme_color(dpg.mvThemeCol_Tab,                   (25, 70, 130),   category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered,            (50, 115, 185),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TabActive,             (45, 110, 180),  category=dpg.mvThemeCat_Core)
                # Buttons
                dpg.add_theme_color(dpg.mvThemeCol_Button,                (35, 95, 165),   category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,         (55, 125, 200),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,          (20, 70, 140),   category=dpg.mvThemeCat_Core)
                # Frame (inputs, combos, checkboxes)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,               (30, 42, 68),    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,        (38, 55, 88),    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,         (45, 65, 100),   category=dpg.mvThemeCat_Core)
                # Slider
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,            (60, 130, 210),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,      (85, 160, 235),  category=dpg.mvThemeCat_Core)
                # Scrollbar
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,           (18, 24, 42),    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,         (45, 80, 140),   category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered,  (60, 100, 165),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive,   (75, 120, 185),  category=dpg.mvThemeCat_Core)
                # Separator & check
                dpg.add_theme_color(dpg.mvThemeCol_Separator,             (50, 85, 140),   category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark,             (80, 180, 240),  category=dpg.mvThemeCat_Core)
                # Text
                dpg.add_theme_color(dpg.mvThemeCol_Text,                  (230, 230, 230), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,          (120, 140, 170), category=dpg.mvThemeCat_Core)
                # Header / collapsible
                dpg.add_theme_color(dpg.mvThemeCol_Header,                (35, 80, 145),   category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,         (50, 100, 170),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,          (25, 65, 125),   category=dpg.mvThemeCat_Core)
                # Plot (progress bar fill)
                dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram,         (35, 120, 200),  category=dpg.mvThemeCat_Core)
                # Style vars
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  6,     category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,    12, 10, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,    5,     category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,     6,  4,  category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,      8,  6,  category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,     4,     category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize,      12,    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding,      4,     category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4,    category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,    5,     category=dpg.mvThemeCat_Core)
        dpg.bind_theme(global_theme)

        # Section header theme (accent-colored text)
        with dpg.theme() as _sec_hdr_theme:
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 180, 240), category=dpg.mvThemeCat_Core)

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

        # --- Analyzing Images window ---
        with dpg.window(label="Analyzing Images", id=imager_window2):
            # Setup
            dpg.bind_item_theme(dpg.add_text(default_value="Setup"), _sec_hdr_theme)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Model Path      ")
                dpg.add_input_text(
                    tag="Model_path_2", readonly=True, hint="Path/to/model", width=250
                )
                dpg.add_spacer(width=4)
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.model_select_bt(),
                )
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Images Path     ")
                dpg.add_input_text(
                    tag="input_image_path", readonly=True, hint="Path/to/images", width=250
                )
                dpg.add_spacer(width=4)
                dpg.add_button(
                    label="Select Files",
                    callback=lambda: self.image_select_bt(),
                )
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Result Directory")
                dpg.add_input_text(
                    tag="Output_Directory_Path2",
                    readonly=True,
                    hint="Path/to/result/directory",
                    width=250,
                )
                dpg.add_spacer(width=4)
                dpg.add_button(
                    label="Select Directory",
                    callback=lambda: self.fd_tk.Out_dir_open(),
                )
            # Preview
            dpg.add_spacer(height=4)
            dpg.bind_item_theme(dpg.add_text(default_value="Preview"), _sec_hdr_theme)
            dpg.add_separator()
            dpg.add_image("imwin_tag1", width=400, height=400)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="< Previous",
                    tag="previous_image",
                    callback=lambda: self.previous_image_bt(),
                )
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="Next >",
                    tag="next_image",
                    callback=lambda: self.next_image_bt(),
                )
                dpg.add_spacer(width=8)
                dpg.add_text(tag="image_num_state", default_value="none")
            # Analysis
            dpg.add_spacer(height=4)
            dpg.bind_item_theme(dpg.add_text(default_value="Start Analyzing"), _sec_hdr_theme)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Run Analysis",
                    tag="analyze_img_btn",
                    width=120,
                    height=30,
                    callback=lambda: self.analyze_image(),
                )
                dpg.add_spacer(width=8)
                with dpg.group(horizontal=False):
                    dpg.add_text(tag="analy_state", default_value="Ready")
                    dpg.add_progress_bar(
                        tag="image_progress_bar",
                        default_value=0.0,
                        width=300,
                        overlay="0%",
                    )
            # Navigation
            dpg.add_spacer(height=4)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Back to Home",
                    tag="home_btn_2",
                    callback=lambda: self.home_cb(),
                )
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="Quit",
                    tag="quit_btn_2",
                    callback=lambda: self.quit_cb(),
                )

        # --- Analyzing Movies window ---
        with dpg.window(label="Analyzing Movies", id=imager_window1):
            # Setup
            dpg.bind_item_theme(dpg.add_text(default_value="Setup"), _sec_hdr_theme)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Model Path      ")
                dpg.add_input_text(
                    tag="Model_path", readonly=True, hint="Path/to/model", width=250
                )
                dpg.add_spacer(width=4)
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.model_select_bt(),
                )
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Movie Path      ")
                dpg.add_input_text(
                    tag="Input file Path", readonly=True, hint="Path/to/movies", width=250
                )
                dpg.add_spacer(width=4)
                dpg.add_button(
                    label="Select Files",
                    callback=lambda: self.movie_select_bt(),
                )
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Result Directory")
                dpg.add_input_text(
                    tag="Output_Directory_Path",
                    readonly=True,
                    hint="Path/to/result/directory",
                    width=250,
                )
                dpg.add_spacer(width=4)
                dpg.add_button(
                    label="Select Directory",
                    callback=lambda: self.fd_tk.Out_dir_open(),
                )
            # Preview
            dpg.add_spacer(height=4)
            dpg.bind_item_theme(dpg.add_text(default_value="Preview"), _sec_hdr_theme)
            dpg.add_separator()
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
                    label="Streaming",
                    default_value=False,
                    tag="streamingChkBox",
                    callback=lambda: self.stream_cb(),
                    enabled=False,
                )
                dpg.add_spacer(width=12)
                dpg.add_text(default_value="Speed")
                dpg.add_combo(
                    items=[1, 2, 5, 10, 20, 50, 100, 200, 500],
                    tag="speed_list",
                    default_value=1,
                    width=100,
                    callback=lambda: self.list_of_speed(),
                )
                dpg.add_spacer(width=12)
                dpg.add_button(
                    label="Vertical Flip",
                    tag="v_flip_state",
                    callback=lambda: self.v_flip_cb(),
                )
                dpg.add_spacer(width=4)
                dpg.add_button(
                    label="Horizontal Flip",
                    tag="h_flip_state",
                    callback=lambda: self.h_flip_cb(),
                )
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Confidence Threshold")
                dpg.add_spacer(width=8)
                dpg.add_input_text(
                    tag="conf_threshold",
                    default_value=self.m_dict["threshold"],
                    width=100,
                    callback=lambda: self.in_thresh(),
                )
            # Analysis
            dpg.add_spacer(height=4)
            dpg.bind_item_theme(dpg.add_text(default_value="Start Analyzing"), _sec_hdr_theme)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Run Analysis",
                    tag="analyze_btn",
                    width=120,
                    height=30,
                    callback=lambda: self.analyze_movie(),
                )
                dpg.add_spacer(width=8)
                dpg.add_checkbox(
                    label="Create Video",
                    tag="create_movie",
                    default_value=self.m_dict["create_video"],
                    callback=lambda: self.create_condition(),
                )
                dpg.add_spacer(width=8)
                dpg.add_checkbox(
                    label="Tracking",
                    tag="tracking_state",
                    default_value=self.m_dict["tracking_state"],
                    callback=lambda: self.tracking_condition(),
                )
            with dpg.group(horizontal=False):
                dpg.add_text(
                    tag="no_mov",
                    default_value=self.m_dict["no_movies"],
                )
                dpg.add_text(
                    tag="analy_time",
                    default_value=self.m_dict["estimate_time"],
                )
                dpg.add_progress_bar(
                    tag="movie_progress_bar",
                    default_value=0.0,
                    width=300,
                    overlay="0%",
                )
            with dpg.group(tag="tracking_exclude_group", show=self.m_dict["tracking_state"]):
                dpg.add_spacer(height=4)
                dpg.add_text(default_value="Exclude classes from tracking:")
                with dpg.child_window(
                    tag="tracking_class_checkboxes", height=90, width=300, border=True
                ):
                    dpg.add_text(
                        tag="tracking_cls_placeholder",
                        default_value="(load model first)",
                    )
            # Navigation
            dpg.add_spacer(height=4)
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Back to Home",
                    tag="home_btn",
                    callback=lambda: self.home_cb(),
                )
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="Quit",
                    tag="quit_btn",
                    callback=lambda: self.quit_cb(),
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

    def model_select_bt(self):
        self.fd_tk.model_file_open()
        self.update_class_list()

    def update_class_list(self):
        model_path = self.m_dict.get("model_path", "")
        if not model_path or not os.path.isfile(str(model_path)):
            return
        try:
            yolo_model = load_yolo_model(str(model_path))
            class_names = yolo_model.names
        except Exception as e:
            print(f"Failed to load class names: {e}")
            return
        dpg.delete_item("tracking_class_checkboxes", children_only=True)
        self.m_dict["tracking_exclude_classes"] = []
        for cls_id, cls_name in class_names.items():
            dpg.add_checkbox(
                label=cls_name,
                tag=f"exclude_cls_{cls_id}",
                default_value=False,
                parent="tracking_class_checkboxes",
                callback=lambda s, a, u=cls_id: self.toggle_exclude_class(u, a),
            )

    def toggle_exclude_class(self, cls_id, value):
        exclude = list(self.m_dict.get("tracking_exclude_classes", []))
        if value:
            if cls_id not in exclude:
                exclude.append(cls_id)
        else:
            if cls_id in exclude:
                exclude.remove(cls_id)
        self.m_dict["tracking_exclude_classes"] = exclude

    def tracking_condition(self):
        tf = dpg.get_value("tracking_state")
        self.m_dict["tracking_state"] = tf
        dpg.configure_item("tracking_exclude_group", show=tf)

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
