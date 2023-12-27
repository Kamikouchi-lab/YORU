import tkinter
import tkinter.filedialog as filedialog
from multiprocessing import Manager, Process

import dearpygui.dearpygui as dpg


class file_dialog_tk:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

    def video_file_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select Video file",
            filetypes=[("Movie file", ".mp4 .wmv .avi .mov .mkv")],  # ファイルフィルタ
        )
        root.destroy()
        dpg.set_value("video_path", file_path)
        return file_path

    def grab_dir_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askdirectory()
        root.destroy()
        dpg.set_value("grab_path", file_path)
        return file_path
