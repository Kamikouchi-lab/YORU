import datetime
import os
import subprocess
import sys
import time
from multiprocessing import Manager, Process

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import yaml
from pynput import keyboard

sys.path.append("../yoru")

from yoru.libs.evaluation_calculation import Evaluator, yolo_analysis_image
from yoru.libs.file_operation_evaluation import file_dialog_tk
from yoru.libs.init_evaluation import init_evaluater

# import yoru.app as YORU

# except(ModuleNotFoundError):
#     from libs.file_operation_evaluation import file_dialog_tk
#     from libs.init_evaluation import init_evaluater
#     from libs.evaluation_calculation import yolo_analysis_image, Evaluator
#     import app as YORU


class model_eval_gui:
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
        if self.width >= self.height:
            self.im_win_width = 400
            self.im_win_height = self.height * (400 / self.width)
        else:
            self.im_win_width = self.width * (400 / self.height)
            self.im_win_height = 400
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

    def gui_configure(self):
        dpg.create_context()
        dpg.configure_app(
            init_file="./config/custom_layout_evaluater_gui.ini",
            docking=True,
            docking_space=True,
        )
        dpg.create_viewport(title="YORU - Evaluation", width=960, height=900)

        # GUI-settings

        imager_window = dpg.generate_uuid()
        with dpg.window(label="Evaluater window", id=imager_window):
            dpg.add_text(default_value="Step1: Load project and model file")
            with dpg.group(horizontal=True):
                dpg.add_text(
                    label="Select Project Config file",
                    default_value="Select Project Config file",
                )
                dpg.add_input_text(
                    tag="config_path", readonly=True, hint="Path/to/config.yaml"
                )
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.fd_tk.config_file_open(),
                    enabled=True,
                )
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
            dpg.add_button(
                label="Load project config",
                tag="load_btn",
                width=150,
                height=30,
                callback=lambda: self.load_pr_dir(),
                enabled=True,
            )
            dpg.add_separator()
            dpg.add_text(default_value="Step2: Run Grab GUI")
            dpg.add_button(
                label="Run Grab GUI",
                tag="grab_btn",
                width=150,
                height=30,
                callback=lambda: self.grab_bt(),
                enabled=True,
            )
            dpg.add_separator()
            dpg.add_text(default_value="Step3: Labeling")
            dpg.add_button(
                label="Run LabelImg",
                tag="labelimg_btn",
                width=150,
                height=30,
                callback=lambda: self.labelImg_bt(),
                enabled=True,
            )
            dpg.add_separator()
            dpg.add_text(default_value="Step4: Create YOLO data")
            dpg.add_button(
                label="Prediction",
                tag="yolo_detection_bt",
                width=150,
                height=30,
                callback=lambda: self.yolo_detection(),
                enabled=True,
            )
            dpg.add_separator()
            dpg.add_text(default_value="Step5: Evaluate Model")

            dpg.add_button(
                label="Calculate APs",
                tag="cal_aps_btn",
                width=150,
                height=30,
                callback=lambda: self.cal_aps_btn(),
                enabled=True,
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

        # setup
        dpg.setup_dearpygui()
        dpg.show_viewport()
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
        file_path = self.m_dict["config_file_path"]
        if not os.path.exists(file_path):
            print("Don't find a project")
            return None
        with open(file_path, "r") as yf:
            data = yaml.safe_load(yf)
            self.m_dict["project_dir"] = data["project_dir"]
            if data.get("evaluation_info_date"):
                self.m_dict["datas_dir"] = data["evaluate_datas_dir"]
                self.m_dict["result_dir"] = data["evaluate_result_dir"]
                self.m_dict["pr_curve_dir"] = data["evaluate_pr_curve_dir"]

            else:
                folder_name = self.m_dict["project_dir"] + "/model_evaluation"
                i = 1
                while os.path.exists(folder_name):
                    folder_name = os.path.join(
                        self.m_dict["project_dir"], "/model_evaluation_" + str(i)
                    )
                    i += 1
                os.makedirs(folder_name)
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

    def grab_bt(self):
        subprocess.call(["python", "grab_gui.py"])

    def labelImg_bt(self):
        subprocess.call(["python", "./labelImg/labelImg.py"])

    def yolo_detection(self):
        yolo_det = yolo_analysis_image(self.m_dict)
        yolo_det.analyze_image()

    def cal_aps_btn(self):
        evaluator = Evaluator(self.m_dict)
        evaluator.run_evaluation(self.m_dict["datas_dir"])

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
        if hasattr(self, "quit"):
            self.m_dict["quit"] = True
        print("=== GUI window quit ===")
        dpg.destroy_context()


def main():
    d = {}
    init = init_evaluater(m_dict=d)
    d["quit"] = False
    evaluaterWin = model_eval_gui(d)
    evaluaterWin.run()


if __name__ == "__main__":
    main()
