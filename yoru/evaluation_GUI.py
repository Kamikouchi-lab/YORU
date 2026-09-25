# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import datetime
import os
import subprocess
import sys

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import yaml

from yoru.gui_base import apply_default_theme, process_frame as _process_frame
from yoru.gui_layout import GuiSession
from yoru.libs.evaluation_calculation import Evaluator, EvaluationImageAnalyzer
from yoru.libs.file_operation_create_label import file_dialog_tk
from yoru.libs.gui_error import GuiErrorMixin
from yoru.libs.init_evaluation import init_evaluater



class model_eval_gui(GuiErrorMixin):
    def __init__(self, m_dict={}):
        print("Evaluater-gui")
        self.m_dict = m_dict
        self.file_path = "./web/image/YORU_logo.png"

        if self.file_path:
            print("File: " + self.file_path)
        else:
            print("Open-file dialog")

        self.image = cv2.imread(self.file_path)
        self.height, self.width, _ = self.image.shape
        self.framecount = 1
        self.current_frame_num = 1
        self.frame = self.image
        self.process_frame()
        self.grab_count = 0
        self.speed = 1
        self.fd_tk = file_dialog_tk(self.m_dict)

    def process_frame(self):
        self.frame_re = _process_frame(self.frame, 400)

    def gui_configure(self):
        self.session = GuiSession(
            "evaluation", "YORU - Evaluation", width=1060, height=820
        )
        self.session.begin()

        # The screen used to carry a three-colour theme of its own -- an orange
        # title bar and green tabs, on a window that has neither -- so the only
        # thing it actually changed was the text colour.  The shared theme is
        # what the rest of YORU looks like.
        apply_default_theme()
        self.session.add_layout_menu()

        # GUI-settings
        with dpg.window(**self.session.window_kwargs("Evaluation", "evaluation_main")):
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step1: Load project and model file     ")
                dpg.add_text(tag="step1_state", default_value="Yet")
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Project Config file")
                dpg.add_input_text(
                    tag="config_path", readonly=True, hint="Path/to/config.yaml",
                    width=-124,
                )
                dpg.add_button(
                    label="Select File",
                    width=116,
                    callback=lambda: self.fd_tk.config_file_open(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                # Padded to the same width as the row above: the labels are
                # what the two path fields start after, so they have to match
                # or the fields step.
                dpg.add_text(default_value="Model Path         ")
                dpg.add_input_text(
                    tag="Model_path", readonly=True, hint="Path/to/model",
                    width=-124,
                )
                dpg.add_button(
                    label="Select File",
                    width=116,
                    callback=lambda: self.fd_tk.model_file_open(),
                    enabled=True,
                )
            dpg.add_button(
                label="Load project config",
                tag="load_btn",
                width=175,
                height=30,
                callback=lambda: self.load_pr_dir(),
                enabled=True,
            )
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step2: YORU Frame Capture     ")
                dpg.add_text(tag="step2_state", default_value="Yet")
            dpg.add_button(
                label="Run YORU Frame Capture",
                tag="grab_btn",
                width=200,
                height=30,
                callback=lambda: self.grab_bt(),
                enabled=True,
            )
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step3: Labeling     ")
                dpg.add_text(tag="step3_state", default_value="Yet")
            dpg.add_button(
                label="Run LabelImg",
                tag="labelimg_btn",
                width=150,
                height=30,
                callback=lambda: self.labelImg_bt(),
                enabled=True,
            )
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step4: Create YOLO data     ")
                dpg.add_text(tag="step4_state", default_value="Yet")
            dpg.add_button(
                label="Prediction",
                tag="yolo_detection_bt",
                width=150,
                height=30,
                callback=lambda: self.yolo_detection(),
                enabled=True,
            )
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step5: Evaluate Model     ")
                dpg.add_text(tag="step5_state", default_value="Yet")
            dpg.add_button(
                label="Calculate APs",
                tag="cal_aps_btn",
                width=150,
                height=30,
                callback=lambda: self.cal_aps_btn(),
                enabled=True,
            )

            dpg.add_separator()
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

        # setup
        self.session.finish(fill_window="evaluation_main")
        # listener = keyboard.Listener(on_press=self.on_key_press)
        # listener.start()

    def run(self):
        self.gui_configure()
        while dpg.is_dearpygui_running():
            self.plot_callback()
            dpg.render_dearpygui_frame()
            if self.m_dict["quit"]:
                if self.m_dict["back_to_home"]:
                    # subprocess.call(["python", "app.py"])
                    from yoru import app as YORU

                    YORU.main()
                dpg.destroy_context()
                break

    def plot_callback(self) -> None:
        if dpg.get_value("streamingChkBox"):
            if 1 + dpg.get_value("frame_bar") > self.framecount - 2:
                dpg.set_value("frame_bar", 0)
            else:
                dpg.set_value("frame_bar", 1 + dpg.get_value("frame_bar"))
            self.slide_bar_cb()

    def list_of_speed(self):
        tf = dpg.get_value("speed_list")
        self.speed = tf

    def load_pr_dir(self):
        print("load project")
        try:
            cfg = self.m_dict.get("config_file_path", "")
            if not cfg or not os.path.exists(cfg):
                raise FileNotFoundError(
                    "Project config file is not selected or does not exist. "
                    "Please select a valid config.yaml."
                )

            with open(cfg, "r") as yf:
                data = yaml.safe_load(yf)
            self.m_dict["project_dir"] = data["project_dir"]

            if data.get("evaluation_info_date"):
                # Load existing evaluation information
                self.m_dict["data_dir"]      = data["evaluate_data_dir"]
                self.m_dict["result_dir"]    = data["evaluate_result_dir"]
                self.m_dict["pr_curve_dir"]  = data["evaluate_pr_curve_dir"]
            else:
                # Create a model_evaluation folder under project_dir
                base = os.path.join(self.m_dict["project_dir"], "model_evaluation")
                folder_name = base
                i = 1
                # Append a suffix if it already exists
                while os.path.exists(folder_name):
                    folder_name = f"{base}_{i}"
                    i += 1
                os.makedirs(folder_name, exist_ok=True)

                # Create data/results/pr_curves directories underneath
                self.m_dict["data_dir"]     = os.path.join(folder_name, "data")
                os.makedirs(self.m_dict["data_dir"], exist_ok=True)

                self.m_dict["result_dir"]   = os.path.join(folder_name, "results")
                os.makedirs(self.m_dict["result_dir"], exist_ok=True)

                self.m_dict["pr_curve_dir"] = os.path.join(self.m_dict["result_dir"], "pr_curves")
                os.makedirs(self.m_dict["pr_curve_dir"], exist_ok=True)

                # Append to YAML
                with open(cfg, "a") as yf:
                    yaml.dump({
                        "evaluate_result_dir":    self.m_dict["result_dir"],
                        "evaluate_data_dir":      self.m_dict["data_dir"],
                        "evaluate_pr_curve_dir":  self.m_dict["pr_curve_dir"],
                        "evaluation_info_date":   datetime.date.today(),
                    }, yf)
                print("add class info in yaml file")

            print("load complete")
            dpg.set_value("step1_state", "Complete!!")
        except Exception as e:
            self._report_error("Failed to load project config", e)
            dpg.set_value("step1_state", "Error")

    def grab_bt(self):
        # Launch as a module with this interpreter: running the file directly
        # puts yoru/ (not the repo root) on sys.path and breaks "from yoru...".
        # Popen, not call, so the GUI keeps rendering.
        try:
            subprocess.Popen([sys.executable, "-m", "yoru.grab_GUI"])
        except OSError as e:
            self._report_error("Failed to launch Frame Capture", e)
            dpg.set_value("step2_state", "Error")
            return
        dpg.set_value("step2_state", "Complete!!")

    def labelImg_bt(self):
        try:
            subprocess.Popen(["labelImg"])
        except OSError as e:
            self._report_error("Failed to launch LabelImg", e)
            dpg.set_value("step3_state", "Error")
            return
        dpg.set_value("step3_state", "Complete!!")

    def yolo_detection(self):
        try:
            yolo_det = EvaluationImageAnalyzer(self.m_dict)
            yolo_det.analyze_image()
            dpg.set_value("step4_state", "Complete!!")
        except Exception as e:
            self._report_error("Prediction failed", e)
            dpg.set_value("step4_state", "Error")

    def cal_aps_btn(self):
        try:
            data_dir = self.m_dict.get("data_dir")
            if not data_dir:
                raise RuntimeError(
                    "Evaluation data directory is not set. "
                    "Please load the project config first (Step1)."
                )
            evaluator = Evaluator(self.m_dict)
            evaluator.run_evaluation(data_dir)
            dpg.set_value("step5_state", "Complete!!")
        except Exception as e:
            self._report_error("AP calculation failed", e)
            dpg.set_value("step5_state", "Error")

    def quit_cb(self):
        print("quit_pushed")
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def home_cb(self):
        print("Back home")
        self.m_dict["back_to_home"] = True
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def __del__(self):
        if hasattr(self, "m_dict"):
            self.m_dict["quit"] = True
        print("=== GUI window quit ===")


def main():
    d = {}
    init = init_evaluater(m_dict=d)
    d["quit"] = False
    evaluaterWin = model_eval_gui(d)
    evaluaterWin.run()


if __name__ == "__main__":
    main()
