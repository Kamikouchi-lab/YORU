# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import datetime
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter
import tkinter.filedialog as filedialog
from multiprocessing import Manager, Process

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from yoru.gui_base import apply_default_theme, frame_to_data_rgba
from yoru.gui_layout import GuiSession
from yoru.gui_lifecycle import run_gui
from yoru.libs.detection import yolo_detection
from yoru.libs.drawing import yolo_drawing
from yoru.libs.file_operation_realtime import file_dialog_tk
from yoru.libs.gui_error import GuiErrorMixin
from yoru.libs.imager import capture_streamCV2, capture_streamMSS, select_run
from yoru.libs.init_realtime import init_asovi
from yoru.libs.trigger import read_condition, yolo_trigger
from yoru.libs.util import loadingParam
from yoru.libs.realtime_state import clear_detection
from yoru.libs.realtime_process import run_workers

DEFAULT_CONFIG_PATH = "./config/yoru_default.yaml"


class camGUI(GuiErrorMixin):
    def __init__(self, config_file=[], m_dict={}):
        self.m_dict = m_dict
        self.t0 = m_dict["t0"]
        self.config_name = config_file
        self.conf = loadingParam(config_file).yml
        self.fd_tk = file_dialog_tk(self.m_dict)
        self.read_condi = read_condition(self.m_dict)
        self.class_list = [self.m_dict["trigger_class"]]
        self.img_file_path = "./web/image/YORU_logo.png"
        self.logo_img = cv2.imread(self.img_file_path)
        self.m_dict["yolo_detection_frame"] = cv2.resize(
            self.logo_img,
            dsize=(self.m_dict["camera_width"], self.m_dict["camera_height"]),
        )

        # load the COM list
        self.read_condi.list_com_ports()

        # load the trigger plugins
        self.read_condi.list_plugins()

        self.texture_data = np.true_divide(self.m_dict["current_camera_frame"], 255.0)

    def startDPG(self):
        # The live camera and the detection view are meant to be watched at the
        # same time, so they open side by side rather than stacked on top of
        # each other, and the arrangement is saved.
        self.session = GuiSession(
            "realtime", "YORU - Real-time Process",
            width=1280, height=880, docking=True,
        )
        self.session.begin()
        apply_default_theme()
        self.session.add_layout_menu()

        frame = self.m_dict["current_camera_frame"]
        self.frameSize = np.shape(frame)
        frame_h, frame_w = self.frameSize[0], self.frameSize[1]

        # imager-window
        with dpg.texture_registry(show=False):
            # A dynamic texture holds four floats per pixel.  The seed value
            # used to be ``np.ones((width, height, 3), np.uint8)`` -- three
            # channels, and with the two axes the wrong way round -- so
            # DearPyGui read a quarter past the end of the buffer the first
            # time it uploaded the texture and the process died on frame one,
            # before the window ever appeared.  Both textures are seeded with
            # a black frame in exactly the form plot_callback goes on to feed
            # them, so what starts on screen and what replaces it agree.
            blank = frame_to_data_rgba(np.zeros((frame_h, frame_w, 3), np.uint8))
            dpg.add_dynamic_texture(
                width=frame_w,
                height=frame_h,
                default_value=blank,
                tag="imwin_tag0",
            )
            dpg.add_dynamic_texture(
                width=frame_w,
                height=frame_h,
                default_value=blank,
                tag="imwin_tag1",
            )

        with dpg.window(**self.session.window_kwargs("Camera", "realtime_camera")):
            dpg.add_image("imwin_tag0", width=480, height=360)
            dpg.add_slider_float(
                label=" [FPS]",
                default_value=0,
                min_value=0,
                max_value=500,
                tag="camFPS_bar",
            )
            dpg.add_text(default_value=" ")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Export dir")
                dpg.add_input_text(
                    tag="export_dir_path",
                    readonly=True,
                    default_value=self.m_dict["export"],
                )
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.fd_tk.Out_dir_open(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="File name: ")
                dpg.add_input_text(
                    default_value=self.m_dict["FileNameHead"] + "_",
                    callback=lambda: self.change_fileName(),
                    tag="fileName",
                )
                dpg.add_text(default_value=".avi")
            dpg.add_checkbox(
                label="streaming data",
                default_value=False,
                tag="streamingChkBox",
                callback=lambda: self.logging(),
            )
            dpg.add_separator()
            # Same shape and order as every other screen: leaving is on the
            # left, quitting is on the right, and neither is a bare unsized
            # button sitting under the one above it.
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Back to Home",
                    tag="home_btn",
                    width=150,
                    height=30,
                    callback=lambda: self.home_cb(),
                    enabled=True,
                )
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="Quit",
                    tag="quit_btn",
                    width=100,
                    height=30,
                    callback=lambda: self.quit_cb(),
                    enabled=True,
                )
        # YOLO window
        with dpg.window(
            **self.session.window_kwargs("Detection & Trigger", "realtime_detection")
        ):
            dpg.add_image("imwin_tag1", width=480, height=360)
            dpg.add_text(default_value="YOLO")
            with dpg.group(horizontal=True):
                dpg.add_input_text(
                    tag="File Path",
                    readonly=True,
                    default_value=self.m_dict["yolo_model"],
                )
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.fd_tk.open_file(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Check model path",
                    tag="test_btn",
                    callback=lambda: self.test_cb(self.m_dict),
                    enabled=True,
                )
                dpg.add_button(
                    label="YOLO model reload",
                    callback=lambda: self.yolo_model_reload(),
                    enabled=True,
                )
            dpg.add_checkbox(
                label="YORU detection",
                default_value=False,
                tag="yolocheckbox",
                callback=lambda: self.yolo_condition(),
            )

            dpg.add_separator()
            dpg.add_text(default_value="Trigger")
            with dpg.group(horizontal=True):
                dpg.add_text(label="tri_cl", default_value="Trigger Class:")
                dpg.add_combo(
                    items=self.class_list,
                    tag="class_list",
                    default_value=self.m_dict["trigger_class"],
                    width=150,
                    callback=lambda: self.list_of_class(),
                )
                dpg.add_button(
                    label="YOLO model class load",
                    callback=lambda: self.load_yolo_class(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_text(label="COM_list", default_value="COM:")
                dpg.add_combo(
                    items=self.m_dict["COM_list"],
                    tag="com_list",
                    default_value=self.m_dict["arduino_com"],
                    width=100,
                    callback=lambda: self.list_in_com(),
                )
                dpg.add_button(
                    label="COM list load",
                    callback=lambda: self.load_COM_list(),
                    enabled=True,
                )
                dpg.add_text(label="pin_input", default_value="  Pin No.:")
                dpg.add_input_text(
                    tag="pin_in",
                    default_value=str(self.m_dict["pin"]),
                    width=100,
                    hint="integer only",
                    callback=lambda: self.pin_input(),
                )
            with dpg.group(horizontal=True):
                dpg.add_text(label="title", default_value="Trigger Plugin:")
                dpg.add_combo(
                    items=self.m_dict["plugins"],
                    tag="plugin_list",
                    default_value=self.m_dict["in_plugin_name"],
                    width=150,
                    callback=lambda: self.list_in_plugin(),
                )
            dpg.add_checkbox(
                label="Trigger condition",
                default_value=False,
                tag="trigger_checkbox",
                callback=lambda: self.trigger_condition(),
            )

        self.session.finish(default_layout={
            "realtime_camera": (0.0, 0.0, 0.5, 1.0),
            "realtime_detection": (0.5, 0.0, 0.5, 1.0),
        })

    def plot_callback(self) -> None:
        now = datetime.datetime.now()
        self.total_time = time.perf_counter() - self.t0

        # Image
        if self.conf["hardware"]["use_camera"]:
            dpg.set_value("imwin_tag0", self.frame_to_data())
            dpg.set_value("camFPS_bar", self.m_dict["camera_fps"])

            # YOLO detection
            dpg.set_value("imwin_tag1", self.yolo_frame_to_data())

    def frame_to_data(self):
        # raw image streaming
        data = np.true_divide(
            cv2.cvtColor(self.m_dict["current_camera_frame"], cv2.COLOR_BGR2RGBA), 255
        )
        # print("a")
        # cv2.imshow("camera", data)
        # self.texture_data = np.true_divide(self.m_dict["current_camera_frame"].ravel(), 255.0)
        # data = np.asfarray(self.texture_data.ravel(), dtype='f')
        return data

    def yolo_frame_to_data(self):
        # YOLO detection streaming
        # data_yolo = np.true_divide(self.m_dict["yolo_detection_frame"], 255.0)
        # self.texture_data_2 = cv2.cvtColor(self.m_dict["yolo_detection_frame"], cv2.COLOR_BGR2RGBA)
        # self.texture_data_2 = np.true_divide(self.texture_data_2, 255.0)
        # self.texture_data_2 = np.true_divide(self.m_dict["yolo_detection_frame"], 255.0)
        # data_yolo = np.asfarray(self.texture_data_2.ravel(), dtype="f")
        # data_yolo = np.asfarray(self.m_dict["yolo_detection_frame"].ravel(), dtype="f")
        data_yolo = np.true_divide(
            cv2.cvtColor(self.m_dict["yolo_detection_frame"], cv2.COLOR_BGR2RGBA), 255
        )
        # cv2.imshow("camera2", data_yolo)
        # cv2.imshow("camera3", self.m_dict["yolo_detection_frame"])
        return data_yolo

    def change_fileName(self):
        self.m_dict["FileNameHead"] = dpg.get_value("fileName")
        self.m_dict["current_camera_frame"]

    def run(self):
        self._plot_error_shown = False
        run_gui(self, dpg, self.startDPG, self._render_frame, self._shutdown,
                reopen_home=False)

    def _render_frame(self):
        try:
            self.plot_callback()
            self._plot_error_shown = False
        except Exception as exc:
            if not self._plot_error_shown:
                self._report_error("Real-time display error", exc)
                self._plot_error_shown = True

    def _shutdown(self):
        self.m_dict["stream"] = False
        self.m_dict["Trigger"] = False
        clear_detection(self.m_dict)

    def quit_cb(self):
        self.m_dict["quit"] = True

    def home_cb(self):
        self.m_dict["back_to_home"] = True
        self.quit_cb()

    def test_cb(self, m_dict):
        print("model path:" + self.m_dict["yolo_model"])

    def yolo_model_reload(self):
        self.m_dict["yolo_process_state"] = False
        self.m_dict["detection_generation"] = self.m_dict.get("detection_generation", 0) + 1
        clear_detection(self.m_dict)
        self.m_dict["Trigger"] = False
        time.sleep(2)
        self.m_dict["yolo_process_state"] = True

        print("reload yolo model complete")

    def yolo_condition(self):
        tf = dpg.get_value("yolocheckbox")
        self.m_dict["yolo_detection"] = False
        self.m_dict["detection_generation"] = self.m_dict.get("detection_generation", 0) + 1
        clear_detection(self.m_dict)
        self.m_dict["yolo_detection"] = tf

    def trigger_condition(self):
        tf = dpg.get_value("trigger_checkbox")
        self.m_dict["Trigger"] = tf

    def load_yolo_class(self):
        self.class_list = self.m_dict["class_name_list"]
        self.class_list.append("None")
        dpg.configure_item("class_list", items=self.class_list)
        print(self.class_list)

    def load_COM_list(self):
        self.read_condi.list_com_ports()
        # self.m_dict["COM_list"].append("None")
        self.com_list = self.m_dict["COM_list"]
        self.com_list.append("None")
        dpg.configure_item("com_list", items=self.com_list)
        print(self.m_dict["COM_list"])

    def list_in_com(self):
        tf = dpg.get_value("com_list")
        self.m_dict["arduino_com"] = tf

    def pin_input(self):
        tf = dpg.get_value("pin_in")
        # pyfirmata indexes board.digital[pin]; a str here would break the trigger.
        try:
            self.m_dict["pin"] = int(tf)
        except (TypeError, ValueError):
            print(f"{tf!r} is not an integer pin number")
            dpg.set_value("pin_in", str(self.m_dict["pin"]))

    def list_in_plugin(self):
        tf = dpg.get_value("plugin_list")
        self.m_dict["in_plugin_name"] = tf

    def list_of_class(self):
        tf = dpg.get_value("class_list")
        self.m_dict["trigger_class"] = tf

    def logging(self):
        enabled = dpg.get_value("streamingChkBox")
        if enabled:
            self.currentLogFileName = dpg.get_value("fileName") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S_%f")
            self.m_dict["curLog"] = self.currentLogFileName
        self.m_dict["stream"] = enabled


def _run_gui(config_path, state):
    camGUI(config_file=config_path, m_dict=state).run()


def _run_capture(state):
    capture = (capture_streamMSS(m_dict=state) if state["stream_MSS"]
               else capture_streamCV2(srcCam=state["camera_id"], m_dict=state))
    capture.run()


def _run_detection(state):
    yolo_detection(m_dict=state).detect(state)


def _run_drawing(state):
    yolo_drawing(m_dict=state).YOLOdraw(state)


def _run_trigger(state):
    yolo_trigger(m_dict=state).init_trigger()


def main(confFileName):
    back_to_home = False
    with Manager() as manager:
        state = manager.dict()
        init_asovi(config_file=confFileName, m_dict=state)
        if state["stream_MSS"]:
            select_run(m_dict=state).main()
            if state.get("quit", False):
                return
        state["camera_imshow"] = False
        processes = [
            Process(name="GUI", target=_run_gui, args=(confFileName, state)),
            Process(name="capture", target=_run_capture, args=(state,)),
            Process(name="detection", target=_run_detection, args=(state,)),
            Process(name="drawing", target=_run_drawing, args=(state,)),
            Process(name="trigger", target=_run_trigger, args=(state,)),
        ]
        run_workers(processes, state)
        back_to_home = state.get("back_to_home", False)
    if back_to_home:
        from yoru import app
        app.main()


def _parse_args(argv=None):
    """Resolve the condition-file path from the command line.

    Launched by ``yoru.app`` as ``python -m yoru.realtime_yoru_GUI <config.yaml>``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="yoru.realtime_yoru_GUI",
        description="YORU real-time detection / closed-loop process.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG_PATH,
        help="Condition YAML file to load",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    config_path = args.config
    if not os.path.isfile(config_path):
        print(f"[yoru] Condition file not found: {config_path}", file=sys.stderr)
        raise SystemExit(1)
    main(config_path)
