import tkinter
import tkinter.filedialog as filedialog
from multiprocessing import Manager, Process

import dearpygui.dearpygui as dpg


class file_dialog_tk:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

    def model_file_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select YOLO model",
            filetypes=[("YOLO model file", ".pt .pth")],  # ファイルフィルタ
            # initialdir = "./" # 自分自身のディレクトリ
        )
        root.destroy()
        dpg.set_value("Model_path", file_path)
        dpg.set_value("Model_path_2", file_path)
        self.m_dict["model_path"] = file_path

    def input_file_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilenames(
            title="Select movie files",
            filetypes=[
                ("movie files", ".mp4 .wmv .avi .mov .mkv .m4v")
            ],  # ファイルフィルタ
            # initialdir = "./" # 自分自身のディレクトリ
        )
        root.destroy()
        dpg.set_value("Input file Path", file_path)
        self.m_dict["input_path"] = file_path

    def input_file_open_image(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilenames(
            title="Select image files",
            filetypes=[
                ("image file", ".jpeg .jpg .tiff .png .gif")
            ],  # ファイルフィルタ
            # initialdir = "./" # 自分自身のディレクトリ
        )
        root.destroy()
        dpg.set_value("input_image_path", file_path)
        self.m_dict["input_path_image"] = file_path

    def Out_dir_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askdirectory()
        root.destroy()
        dpg.set_value("Output_Directory_Path", file_path)
        dpg.set_value("Output_Directory_Path2", file_path)
        self.m_dict["output_path"] = file_path
