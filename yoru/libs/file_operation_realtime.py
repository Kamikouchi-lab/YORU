import tkinter
import tkinter.filedialog as filedialog
from multiprocessing import Manager, Process

import dearpygui.dearpygui as dpg


class file_dialog_tk:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

    def open_file(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="select YOLO model",
            filetypes=[("YOLO model file", ".pt .pth")],  # ファイルフィルタ
            initialdir="./",  # 自分自身のディレクトリ
        )
        root.destroy()
        dpg.set_value("File Path", file_path)
        self.m_dict["yolo_model"] = file_path

    def Out_dir_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askdirectory()
        root.destroy()
        dpg.set_value("export_dir_path", file_path)
        self.m_dict["export"] = file_path


if __name__ == "__main__":
    dpg.create_context()
    confFileName = "yolo_default.yaml"
    with Manager() as manager:
        d = manager.dict()
        tkgui = file_dialog_tk(m_dict=d)

        with dpg.window(label="Main Window", tag="Main Window"):
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="File Path")
                dpg.add_button(label="Select File", callback=tkgui.open_file)

        dpg.create_viewport(title=f"File Example", width=800, height=600)
        dpg.set_primary_window("Main Window", True)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
