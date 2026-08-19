# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import logging
import os
import threading
from multiprocessing import Manager, Process

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from yoru.gui_base import apply_default_theme, frame_to_data_rgb, process_frame as _process_frame
from yoru.libs.analysis import yolo_analysis, yolo_analysis_image
from yoru.libs.file_operation_analysis import file_dialog_tk
from yoru.libs.init_analysis import init_analysis
from yoru.libs.plugins import get_detector

logger = logging.getLogger(__name__)

PREVIEW_SIZE = 400


class analyze_GUI:
    def __init__(self, m_dict=None):
        self.m_dict = m_dict if m_dict is not None else {}
        self.fd_tk = file_dialog_tk(self.m_dict)

        self.file_path = "./web/image/YORU_logo.png"

        if self.file_path:
            logger.info("File: %s", self.file_path)
        else:
            logger.info("Open-file dialog")

        self.vid = cv2.imread(self.file_path)
        self.height, self.width, _ = self.vid.shape
        self.framecount = 1
        self.current_frame_num = 1
        self.frame = self.vid
        self.process_frame()
        self.grab_count = 0
        self.speed = 1
        self._job_active = False

    def process_frame(self):
        self.frame_re = _process_frame(
            self.frame, PREVIEW_SIZE,
            v_flip=self.m_dict.get("v_flip", False),
            h_flip=self.m_dict.get("h_flip", False),
        )

    def startDPG(self):
        dpg.create_context()
        dpg.configure_app(
            init_file="./logs/custom_layout_analysis.ini",
            docking=True,
            docking_space=True,
        )

        dpg.create_viewport(title="YORU - Video Analysis", width=1000, height=800, max_width=1000, max_height=800)

        # Theme
        apply_default_theme()

        # Section header theme (accent-colored text)
        with dpg.theme() as _sec_hdr_theme:
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (100, 180, 240), category=dpg.mvThemeCat_Core)

        # GUI-settings
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                width=PREVIEW_SIZE,
                height=PREVIEW_SIZE,
                default_value=self.frame_to_data(self.frame_re),
                tag="imwin_tag0",
                format=dpg.mvFormat_Float_rgb,
            )
            dpg.add_raw_texture(
                width=PREVIEW_SIZE,
                height=PREVIEW_SIZE,
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
            dpg.add_image("imwin_tag1", width=PREVIEW_SIZE, height=PREVIEW_SIZE)
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
            dpg.add_image("imwin_tag0", width=PREVIEW_SIZE, height=PREVIEW_SIZE)
            dpg.add_slider_int(
                label=" Frame",
                default_value=0,
                min_value=0,
                max_value=self.framecount - 2,
                tag="frame_bar",
                width=PREVIEW_SIZE,
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

    def _sync_analysis_progress(self) -> None:
        """Copy worker-thread progress out of m_dict into the widgets.

        Called once per rendered frame so that a long analysis no longer blocks
        the GUI (the work itself runs on a background thread).
        """
        progress = self.m_dict.get("movie_progress", 0.0)
        dpg.set_value("movie_progress_bar", progress)
        dpg.configure_item("movie_progress_bar", overlay=f"{int(progress * 100)}%")
        dpg.set_value("analy_time", self.m_dict.get("estimate_time", ""))
        dpg.set_value("no_mov", self.m_dict.get("no_movies", ""))

        img_progress = self.m_dict.get("image_progress", 0.0)
        dpg.set_value("image_progress_bar", img_progress)
        dpg.configure_item(
            "image_progress_bar",
            overlay=self.m_dict.get("image_progress_label", "0%"),
        )
        dpg.set_value("analy_state", self.m_dict.get("analy_state", "Ready"))

        if self._job_active and not self.m_dict.get("analysis_running", False):
            self._job_active = False
            dpg.enable_item("analyze_btn")
            dpg.enable_item("create_movie")
            dpg.enable_item("analyze_img_btn")

    def plot_callback(self) -> None:
        self._sync_analysis_progress()
        if dpg.get_value("streamingChkBox"):
            try:
                speed = int(self.speed)
            except (ValueError, TypeError):
                speed = 1
            if speed + dpg.get_value("frame_bar") > self.framecount - 2:
                dpg.set_value("frame_bar", 0)
            else:
                dpg.set_value("frame_bar", speed + dpg.get_value("frame_bar"))
            self.slide_bar_cb()

    def slide_bar_cb(self):
        self.current_frame_num = dpg.get_value("frame_bar")
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        self.status, self.frame = self.vid.read()
        self.process_frame()

        dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))

    def file_open(self):
        if self.file_path:
            logger.info("File: %s", self.file_path)
        else:
            logger.warning("Failed open files")

        self.vid = cv2.VideoCapture(self.file_path)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.framecount = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0
        self.status, self.frame = self.vid.read()
        self.process_frame()
        logger.info("Movie size: %s x %s", self.width, self.height)
        dpg.configure_item("frame_bar", max_value=self.framecount - 2)
        dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))
        dpg.enable_item("streamingChkBox")
        dpg.enable_item("frame_bar")

    def file_open_image(self):
        if self.file_path_image:
            logger.info("File: %s", self.file_path_image)
        else:
            logger.warning("Failed open image")
        self.frame = cv2.imread(self.file_path_image)
        self.height, self.width, _ = self.frame.shape
        self.process_frame()
        logger.info("Image size: %s x %s", self.width, self.height)
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
        return frame_to_data_rgb(frame)

    def list_of_speed(self):
        tf = dpg.get_value("speed_list")
        self.speed = tf

    def movie_select_bt(self):
        self.fd_tk.input_file_open()
        paths = self.m_dict.get("input_path", [])
        if not paths:
            return
        self.file_path = paths[0]
        self.file_open()

    def image_select_bt(self):
        self.fd_tk.input_file_open_image()
        images = self.m_dict.get("input_path_image", [])
        if not images:
            return
        self.current_image_num = 0
        self.file_path_image = images[0]
        self.image_num = len(images) - 1
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
        logger.info("quit_pushed")
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def home_cb(self):
        logger.info("Back home")
        self.m_dict["back_to_home"] = True
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def _start_job(self, worker, what: str) -> bool:
        """Disable the run buttons and launch *worker* on a daemon thread."""
        if self.m_dict.get("analysis_running", False):
            logger.info("An analysis is already running")
            return False

        model_path = str(self.m_dict.get("model_path", ""))
        if not os.path.isfile(model_path):
            self.m_dict["analy_state"] = "Error: select a model file first"
            logger.error("No model selected (model_path=%r)", model_path)
            return False

        self.m_dict["analysis_running"] = True
        self.m_dict["analysis_error"] = ""
        self.m_dict["analy_state"] = f"Analyzing {what}..."
        self._job_active = True
        dpg.disable_item("analyze_btn")
        dpg.disable_item("create_movie")
        dpg.disable_item("analyze_img_btn")
        threading.Thread(target=self._run_job, args=(worker,), daemon=True).start()
        return True

    def _run_job(self, worker) -> None:
        logger.info("Start analyzing ....")
        try:
            worker()
            logger.info("Analysis complete!!")
        except Exception as e:
            logger.exception("Analysis failed")
            self.m_dict["analysis_error"] = f"{type(e).__name__}: {e}"
            self.m_dict["analy_state"] = f"Error: {type(e).__name__}: {e}"
        finally:
            self.m_dict["analysis_running"] = False

    def analyze_movie(self):
        def _work():
            self.yolo_analysis = yolo_analysis(self.m_dict)
            self.yolo_analysis.analyze()

        self._start_job(_work, "movies")

    def analyze_image(self):
        def _work():
            self.yolo_analysis = yolo_analysis_image(self.m_dict)
            self.yolo_analysis.analyze_image()

        self._start_job(_work, "images")

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
            detector = get_detector("auto", str(model_path))
            class_names = detector.names
        except (OSError, RuntimeError, ImportError) as e:
            logger.error("Failed to load class names: %s", e)
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
        try:
            self.m_dict["threshold"] = float(tf)
        except (ValueError, TypeError):
            pass

    def __del__(self):
        logger.info("=== GUI window quit ===")


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
