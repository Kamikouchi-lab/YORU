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
from cProfile import label
from multiprocessing import Manager, Process
from turtle import Screen

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import pyglet

# try:


sys.path.append("../yoru")

# from yoru.libs.Window import Window
from yoru.libs.detection import yolo_detection
from yoru.libs.drawing import yolo_drawing
from yoru.libs.file_operation_realtime import file_dialog_tk
from yoru.libs.imager import capture_streamCV2, capture_streamMSS, select_run
from yoru.libs.init_realtime import init_asovi
from yoru.libs.trigger import read_condition, yolo_trigger
from yoru.libs.util import loadingParam

# except(ModuleNotFoundError):
#     # import app
#     from libs.file_operation_realtime import file_dialog_tk
#     from libs.imager import capture_streamCV2, capture_streamMSS, select_run
#     from libs.init_realtime import init_asovi
#     # from triggers.trigger_default import yolo_trigger, read_com
#     # from libs.trigger_fix import read_condition, yolo_trigger
#     from libs.trigger import read_condition, yolo_trigger
#     from libs.util import loadingParam
#     # from libs.Window import Window
#     from libs.detection import yolo_detection
#     from libs.drawing import yolo_drawing
#     import app as YORU


class camGUI:
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

        # COMリストの読み込み
        self.read_condi.list_com_ports()

        # trigger pluginsの読み込み
        self.read_condi.list_plugins()

        # flatten camera data to a 1 d stricture
        data = np.flip(self.m_dict["current_camera_frame"], 2).ravel()
        data = np.asfarray(data, dtype="f")
        self.texture_data = np.true_divide(self.m_dict["current_camera_frame"], 255.0)

    def startDPG(self):
        dpg.create_context()
        dpg.configure_app(
            init_file="./logs/custom_layout_camGUI_.ini",
            docking=True,
            docking_space=True,
        )
        dpg.create_viewport(title="YORU - Real-time analysis", width=1280, height=700)

        imager_window = dpg.generate_uuid()
        imager_window2 = dpg.generate_uuid()

        frame = self.m_dict["current_camera_frame"]
        self.frameSize = np.shape(frame)
        print(self.frameSize)

        # imager-window
        with dpg.texture_registry(show=False):
            imgwhite = np.ones((self.frameSize[1], self.frameSize[0], 3), np.uint8)
            dpg.add_dynamic_texture(
                width=self.frameSize[1],
                height=self.frameSize[0],
                default_value=imgwhite,
                tag="imwin_tag0",
            )  # , format=dpg.mvFormat_Float_rgb)
            # dpg.add_raw_texture(width=self.m_dict["camera_width"]*self.m_dict["camera_scale"],
            #                     height=self.m_dict["camera_height"]*self.m_dict["camera_scale"],
            #                     default_value=self.texture_data,
            #                     tag="imwin_tag0", format=dpg.mvFormat_Float_rgb)

            dpg.add_dynamic_texture(
                width=self.frameSize[1],
                height=self.frameSize[0],
                default_value=imgwhite,
                tag="imwin_tag1",
            )
            # dpg.add_raw_texture(
            #     width=self.m_dict["camera_width"] * self.m_dict["camera_scale"],
            #     height=self.m_dict["camera_height"] * self.m_dict["camera_scale"],
            #     default_value=self.texture_data,
            #     tag="imwin_tag1",
            #     format=dpg.mvFormat_Float_rgb,
            # )

        with dpg.window(label="Image window1", id=imager_window):
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
            dpg.add_button(
                label="Quit",
                tag="quit_btn",
                callback=lambda: self.quit_cb(),
                enabled=True,
            )
            dpg.add_button(
                label="Back to Home",
                tag="home_btn",
                callback=lambda: self.home_cb(),
                enabled=True,
            )
        # YOLO window
        with dpg.window(label="Processed Image", id=imager_window2):
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
                label="YOLO detection",
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
                    default_value=self.m_dict["pin"],
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

        dpg.setup_dearpygui()
        dpg.show_viewport()

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
        self.startDPG()
        while dpg.is_dearpygui_running():
            self.plot_callback()
            dpg.render_dearpygui_frame()
            if self.m_dict["quit"]:
                if self.m_dict["back_to_home"]:
                    # subprocess.call(["python", "app.py"])

                    from yoru import app as YORU

                    YORU.main()
                dpg.destroy_context()  # <-- moved from __del__
                break

    def quit_cb(self):
        print("quit_pushed")
        # self.currentLogFile.write("# Streaming finished: " + str(datetime.datetime.now()) + "\r")
        # self.currentLogFile.close()
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def home_cb(self):
        print("Back to home")
        self.m_dict["back_to_home"] = True
        self.m_dict["quit"] = True
        time.sleep(0.5)
        dpg.destroy_context()  # <-- moved from __del__

    def test_cb(self, m_dict):
        print("model path:" + self.m_dict["yolo_model"])

    def yolo_model_reload(self):
        self.m_dict["yolo_process_state"] = False
        self.m_dict["Trigger"] = False
        time.sleep(2)
        self.m_dict["yolo_process_state"] = True

        print("reload yolo model complete")

    def yolo_condition(self):
        tf = dpg.get_value("yolocheckbox")
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
        self.m_dict["pin"] = tf
        # if isinstance(tf, int):
        #     self.m_dict["baudrate"] = tf
        # else:
        #     print(f"{tf} is not int")
        #     dpg.set_value("baudrate_in", self.m_dict["baudrate"])
        #     time.sleep(0.5)

    def list_in_plugin(self):
        tf = dpg.get_value("plugin_list")
        self.m_dict["in_plugin_name"] = tf

    def list_of_class(self):
        tf = dpg.get_value("class_list")
        self.m_dict["trigger_class"] = tf

    def logging(self):
        tf = dpg.get_value("streamingChkBox")
        if tf:  # streaming start
            dt = datetime.datetime.now()
            fnhead = dpg.get_value("fileName")
            self.currentLogFileName = fnhead + dt.strftime("%Y%m%d-%H%M%S_%f") 
            # self.currentLogFile = open(
            #     self.m_dict["export"] + "/" + self.currentLogFileName, "a+"
            # )
            # self.currentLogFile.write("# Streaming start: " + str(dt) + "\r")
            # self.currentLogFile.write(
            #     "# Date, total time, Count, Speed, Track, Position, Dark, Z-stage, Gain_d, di0, di1, di2, di3, "
            #     + "\r"
            # )
            self.m_dict["curLog"] = self.currentLogFileName
            self.m_dict["stream"] = tf
        else:
            self.m_dict["stream"] = tf
            ret = shutil.copyfile(
                self.config_name,
                self.m_dict["export"] + "/" + self.currentLogFileName + ".yaml",
            )
            # print("Saved config: ", ret)
            # self.currentLogFile.close()

    def __del__(self):
        if hasattr(self, "quit"):
            self.m_dict["quit"] = True
        if hasattr(self, "eventLogfile_man"):
            self.eventLogfile_man.close()

        print("=== GUI window quit ===")
        dpg.destroy_context()

        # Check if Python is in the process of shutting down
        if not sys.is_finalizing():
            pyglet.app.exit()


def main(confFileName):
    with Manager() as manager:
        d = manager.dict()
        d["initialized"] = False
        init_md = init_asovi(config_file=confFileName, m_dict=d)

        # MSS or Camera
        # d["stream_MSS"] = False
        if d["stream_MSS"]:
            SR = select_run(m_dict=d)
            SR.main()
            imgWin = capture_streamMSS(m_dict=d)
        else:
            imgWin = capture_streamCV2(srcCam=d["camera_id"], m_dict=d)

        gui = camGUI(config_file=confFileName, m_dict=d)
        yolo_det = yolo_detection(m_dict=d)
        yolo_draw = yolo_drawing(m_dict=d)
        yolo_tri = yolo_trigger(m_dict=d)
        gui = camGUI(config_file=confFileName, m_dict=d)

        d["camera_imshow"] = False

        process_pool = []

        prc_imager = Process(target=imgWin.run)
        prc_gui = Process(target=gui.run)
        prc_yolo = Process(target=yolo_det.detect, args=(d,))
        prc_yolo_draw = Process(target=yolo_draw.YOLOdraw, args=(d,))
        prc_tri = Process(target=yolo_tri.init_trigger)

        prc_gui.start()
        prc_imager.start()
        prc_yolo.start()
        prc_yolo_draw.start()
        prc_tri.start()

        prc_gui.join()
        prc_imager.join()
        prc_yolo.join()
        prc_yolo_draw.join()
        prc_tri.join()

        print(d)


if __name__ == "__main__":
    confFileName = "./config/yoru_demo_display.yaml"
    main(confFileName)

    # with Manager() as manager:
    #     d = manager.dict()
    #     d["initialized"] = False
    #     init_md = init_asovi(config_file=confFileName, m_dict=d)

    #     # MSS or Camera
    #     d["stream_MSS"] = False
    #     if d["stream_MSS"]:
    #         SR = select_run(m_dict=d)
    #         SR.main()
    #         imgWin = capture_streamMSS(m_dict=d)
    #     else:
    #         imgWin = capture_streamCV2(srcCam=d["camera_id"], m_dict=d)

    #     gui = camGUI(config_file=confFileName, m_dict=d)
    #     yolo_det = yolo_detection(m_dict=d)
    #     yolo_draw = yolo_drawing(m_dict=d)
    #     yolo_tri = yolo_trigger(m_dict=d)
    #     gui = camGUI(config_file=confFileName, m_dict=d)

    #     d["camera_imshow"] = False

    #     process_pool = []

    #     prc_imager = Process(target=imgWin.run)
    #     prc_gui = Process(target=gui.run)
    #     prc_yolo = Process(target=yolo_det.detect, args=(d,))
    #     prc_yolo_draw = Process(target=yolo_draw.YOLOdraw, args=(d,))
    #     prc_tri = Process(target=yolo_tri.init_trigger)

    #     prc_gui.start()
    #     prc_imager.start()
    #     prc_yolo.start()
    #     prc_yolo_draw.start()
    #     prc_tri.start()

    #     prc_gui.join()
    #     prc_imager.join()
    #     prc_yolo.join()
    #     prc_yolo_draw.join()
    #     prc_tri.join()

    #     print(d)
