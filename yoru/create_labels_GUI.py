# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import datetime
import os
import sys

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import yaml

from yoru.gui_base import apply_default_theme, process_frame as _process_frame
from yoru.gui_layout import GuiSession
from yoru.libs.create_labels import yolo_analysis_image
from yoru.libs.create_yaml_train import is_obb_project
from yoru.libs.file_operation_create_label import file_dialog_tk
from yoru.libs.gui_error import GuiErrorMixin
from yoru.libs.init_create_label import init_create_label



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
        # The title said "YORU - Evaluation": this screen was cloned from the
        # evaluation GUI and the window caption came along with it.
        self.session = GuiSession(
            "create_labels", "YORU - Create Labels", width=1000, height=780
        )
        self.session.begin()
        apply_default_theme()
        self.session.add_layout_menu()

        # GUI-settings
        with dpg.window(
            **self.session.window_kwargs("Create Labels", "create_labels_main")
        ):
            dpg.add_text(default_value="Step 1: Load project and model file")
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
                # Padded to the width of the label above so the two path fields
                # start at the same place.
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
            dpg.add_text(default_value="Step 2: Labeling")
            dpg.add_button(
                label="Run LabelImg",
                tag="labelimg_btn",
                width=150,
                height=30,
                callback=lambda: self.labelImg_bt(),
                enabled=True,
            )
            dpg.add_separator()
            dpg.add_text(default_value="Step 3: Create YOLO data")
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
        self.session.finish(fill_window="create_labels_main")
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
            file_path = self.m_dict.get("config_file_path", "")
            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(
                    "Project config file is not selected or does not exist. "
                    "Please select a valid config.yaml."
                )
            with open(file_path, "r") as yf:
                data = yaml.safe_load(yf)
                self.m_dict["project_dir"] = data["project_dir"]
                # The project decides the annotation format, the same as in the
                # training GUI; labelImg is told explicitly below.
                self.m_dict["obb"] = is_obb_project(data)
                if data.get("evaluation_info_date"):
                    self.m_dict["datas_dir"] = data["evaluate_datas_dir"]
                    self.m_dict["result_dir"] = data["evaluate_result_dir"]
                    self.m_dict["pr_curve_dir"] = data["evaluate_pr_curve_dir"]

                else:
                    base = os.path.join(
                        self.m_dict["project_dir"], "model_evaluation"
                    )
                    folder_name = base
                    i = 1
                    while os.path.exists(folder_name):
                        folder_name = f"{base}_{i}"
                        i += 1
                    os.makedirs(folder_name, exist_ok=True)
                    print(folder_name)
                    self.m_dict["datas_dir"] = os.path.join(folder_name, "datas")
                    os.makedirs(self.m_dict["datas_dir"])

                    self.m_dict["result_dir"] = os.path.join(folder_name, "result")
                    os.makedirs(self.m_dict["result_dir"])

                    self.m_dict["pr_curve_dir"] = os.path.join(
                        self.m_dict["result_dir"], "pr_curves"
                    )
                    os.makedirs(self.m_dict["pr_curve_dir"])

                    with open(file_path, "a") as yf:
                        yaml.dump(
                            {
                                "evaluate_result_dir": self.m_dict["result_dir"],
                                "evaluate_datas_dir": self.m_dict["datas_dir"],
                                "evaluate_pr_curve_dir": self.m_dict["pr_curve_dir"],
                                "evaluation_info_date": datetime.date.today(),
                            },
                            yf,
                        )
                    print("add class info in yaml file")

                print(f"load complete")
        except Exception as e:
            self._report_error("Failed to load project config", e)

    def labelImg_bt(self):
        """Open the bundled labelImg on the evaluation images.

        The argv is built here rather than passed straight through from
        ``sys.argv``: this GUI's own arguments have nothing to do with
        labelImg's, and one of them arriving as a positional would open the
        wrong directory.  ``--obb`` / ``--no-obb`` is always passed, so the
        format follows the loaded project instead of whatever labelImg
        remembered from last time.
        """
        import subprocess
        import sys

        cmd = [sys.executable, "-m", "yoru.labelimg.labelimg"]
        datas_dir = self.m_dict.get("datas_dir") or ""
        if datas_dir and os.path.isdir(datas_dir):
            classes_txt = os.path.join(datas_dir, "classes.txt")
            cmd += [
                datas_dir,
                classes_txt if os.path.isfile(classes_txt) else "",
                datas_dir,
            ]
        cmd.append("--obb" if self.m_dict.get("obb") else "--no-obb")

        try:
            # Popen, not an in-process QApplication: Qt's event loop and
            # DearPyGui's cannot both own this process.
            subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
        except OSError as e:
            self._report_error("Failed to launch LabelImg", e)

    def yolo_detection(self):
        try:
            yolo_det = yolo_analysis_image(self.m_dict)
            yolo_det.analyze_image()
        except Exception as e:
            self._report_error("Label generation (prediction) failed", e)

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
    init = init_create_label(m_dict=d)
    d["quit"] = False
    evaluaterWin = model_eval_gui(d)
    evaluaterWin.run()


if __name__ == "__main__":
    main()
