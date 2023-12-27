import glob
import os
import random
import shutil
from collections import Counter
import tkinter
import tkinter.filedialog as filedialog
from multiprocessing import Manager, Process

import dearpygui.dearpygui as dpg


class file_move_random:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

    def read_txt_files(self):
        # Specify your directory path here
        self.directory_path = self.m_dict["all_label_dir"]
        # Create a dictionary to hold the results
        self.result_dict = {}
        self.move_files_dict = {}
        # Use glob to get all .txt files in the directory
        for self.txt_file in glob.glob(self.directory_path + "/*.txt"):
            # Get the filename only from the initial file variable
            filename = os.path.basename(self.txt_file)
            # calssファイルを読み込まないようにする
            if filename == "classes.txt":
                continue

            with open(self.txt_file, "r") as f:
                lines = f.readlines()

            numbers = [int(line.split()[0]) for line in lines]
            counts = Counter(numbers)
            try:
                most_common_element = counts.most_common()[-1][0]
                base_name = os.path.splitext(filename)[0]
                self.result_dict[base_name] = most_common_element
            except IndexError:
                pass

        counts = Counter(self.result_dict.values())
        class_list = list(counts.keys())
        select_dict = {}
        self.move_files_dict["train"] = []
        self.move_files_dict["val"] = []

        for i in class_list:
            keys_list = [key for key, value in self.result_dict.items() if value == i]
            select_dict[i] = keys_list
            # Calculate the number of elements to select
            num_to_select = (
                len(select_dict[i]) // 5
            )  # Use integer division to get an integer 80%

            # Randomly select elements from the list
            selected_elements = random.sample(select_dict[i], num_to_select)
            self.move_files_dict["val"].extend(selected_elements)
            not_selected_elements = list(set(select_dict[i]) - set(selected_elements))
            self.move_files_dict["train"].extend(not_selected_elements)

        print(self.move_files_dict)

    def move(self):
        print("start")
        self.read_txt_files()
        for i in self.move_files_dict["train"]:
            source_file_label = self.m_dict["all_label_dir"] + "/" + i + ".txt"
            source_file_image = self.m_dict["all_label_dir"] + "/" + i + ".png"
            destination_dir_label = self.m_dict["project_dir"] + "/train/labels"
            destination_dir_image = self.m_dict["project_dir"] + "/train/images"
            try:
                shutil.copy(source_file_label, destination_dir_label)
                shutil.copy(source_file_image, destination_dir_image)
            except FileNotFoundError:
                print(f"Don't find {i} files")

        for i in self.move_files_dict["val"]:
            source_file_label = self.m_dict["all_label_dir"] + "/" + i + ".txt"
            source_file_image = self.m_dict["all_label_dir"] + "/" + i + ".png"
            destination_dir_label = self.m_dict["project_dir"] + "/val/labels"
            destination_dir_image = self.m_dict["project_dir"] + "/val/images"
            try:
                shutil.copy(source_file_label, destination_dir_label)
                shutil.copy(source_file_image, destination_dir_image)
            except FileNotFoundError:
                print(f"Don't find {i} files")
        print("complete")



class file_dialog_tk:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict

    def pro_dir_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askdirectory()
        root.destroy()
        dpg.set_value("project_path", file_path)
        self.m_dict["project_path"] = file_path

    def class_txt_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="select class file",
            filetypes=[("Classes file", "classes.txt")],  # ファイルフィルタ
            initialdir="./",  # 自分自身のディレクトリ
        )
        root.destroy()
        dpg.set_value("classes_path", file_path)
        self.m_dict["classes_path"] = file_path

    def dataset_file_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="select dataset",
            filetypes=[("config file", "config.yml .yaml")],  # ファイルフィルタ
        )
        root.destroy()
        dpg.set_value("yaml_file_path", file_path)
        self.m_dict["yaml_path"] = file_path

    def input_file_open(self):
        root = tkinter.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="select YOLO model",
            filetypes=[("movie file", ".mp4 .wmv .avi")],  # ファイルフィルタ
            initialdir="./",  # 自分自身のディレクトリ
        )
        root.destroy()
        dpg.set_value("Input file Path", file_path)
        self.m_dict["input_path"] = file_path


if __name__ == "__main__":
    d = {}
    d[
        "all_label_dir"
    ] = "C:/Users/nokai/Desktop/230719_ando_copulation_detection_YORU/labels/labels_fit_chamber"
    d[
        "project_dir"
    ] = "C:/Users/nokai/Desktop/230719_ando_copulation_detection_YORU/labels"
    d["quit"] = False
    fmrd = file_move_random(d)
    fmrd.move()
