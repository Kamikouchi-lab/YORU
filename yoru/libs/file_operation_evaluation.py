import os
import tkinter
import tkinter.filedialog as filedialog
from multiprocessing import Manager, Process

import dearpygui.dearpygui as dpg


class file_dialog_tk:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

    def config_file_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select config.yaml file",
            filetypes=[("Config file", "config.yaml config.yml")],  # ファイルフィルタ
        )
        root.destroy()
        dpg.set_value("config_path", file_path)
        self.m_dict["config_file_path"] = file_path
        print(self.m_dict["config_file_path"])

    def model_file_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="select YOLO model",
            filetypes=[("YOLO model file", ".pt .pth")],  # ファイルフィルタ
            # initialdir = "./" # 自分自身のディレクトリ
        )
        root.destroy()
        dpg.set_value("Model_path", file_path)
        self.m_dict["model_path"] = file_path

    def input_file_open_image(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilenames(
            title="select YOLO model",
            filetypes=[
                ("image file", ".jpeg .jpg .tiff .png .gif")
            ],  # ファイルフィルタ
            # initialdir = "./" # 自分自身のディレクトリ
        )
        root.destroy()
        dpg.set_value("input_image_path", file_path)
        self.m_dict["input_path_image"] = file_path

    def result_dir_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askdirectory()
        root.destroy()

        self.m_dict["datas_dir"] = os.path.join(file_path, "datas")
        self.m_dict["result_dir"] = os.path.join(file_path, "result")
        dpg.set_value("directory_Path", file_path)
