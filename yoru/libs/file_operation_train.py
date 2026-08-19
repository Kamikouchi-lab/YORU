# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import glob
import os
import random
import shutil
import tkinter
import tkinter.filedialog as filedialog
from collections import Counter
from multiprocessing import Manager, Process

import dearpygui.dearpygui as dpg


class file_move_random:
    #: Image formats looked for alongside a YOLO .txt label file.
    IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, m_dict={}):
        self.m_dict = m_dict

    def _find_image(self, base_name):
        """Path of the image belonging to *base_name*, whatever its extension."""
        directory = self.m_dict["all_label_dir"]
        for ext in self.IMAGE_SUFFIXES:
            candidate = os.path.join(directory, base_name + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def read_txt_files(self):
        # Specify your directory path here
        self.directory_path = self.m_dict["all_label_dir"]
        # Create a dictionary to hold the results
        self.result_dict = {}
        self.move_files_dict = {}
        # sorted() so the split depends only on the seed, not on the filesystem
        for self.txt_file in sorted(glob.glob(self.directory_path + "/*.txt")):
            # Get the filename only from the initial file variable
            filename = os.path.basename(self.txt_file)
            # calssファイルを読み込まないようにする
            if filename == "classes.txt":
                continue

            with open(self.txt_file, "r") as f:
                lines = f.readlines()

            # skip blank lines: a trailing newline used to raise IndexError here
            numbers = [int(line.split()[0]) for line in lines if line.split()]
            if not numbers:
                continue
            counts = Counter(numbers)

            # NOTE: most_common()[-1] is the *least* frequent class in this
            # image, not the most frequent. That is kept deliberately -
            # stratifying on the rarest class present keeps rare classes in both
            # splits, and changing it would silently reshuffle existing
            # projects. Only the misleading variable name was corrected.
            rarest_class = counts.most_common()[-1][0]
            base_name = os.path.splitext(filename)[0]
            self.result_dict[base_name] = rarest_class

        counts = Counter(self.result_dict.values())
        class_list = sorted(counts.keys())
        select_dict = {}
        self.move_files_dict["train"] = []
        self.move_files_dict["val"] = []

        # A dedicated Random instance makes the split reproducible without
        # disturbing the global random state. Override via m_dict["split_seed"].
        rng = random.Random(self.m_dict.get("split_seed", 0))

        for i in class_list:
            keys_list = [key for key, value in self.result_dict.items() if value == i]
            select_dict[i] = keys_list
            # 20% of each stratum goes to val, the remaining 80% to train
            num_to_select = len(select_dict[i]) // 5

            selected_elements = rng.sample(select_dict[i], num_to_select)
            self.move_files_dict["val"].extend(selected_elements)
            # keep source order instead of set arithmetic, whose iteration order
            # varies between runs
            chosen = set(selected_elements)
            self.move_files_dict["train"].extend(
                k for k in keys_list if k not in chosen
            )

        print(self.move_files_dict)

    def move(self):
        print("start")
        self.read_txt_files()
        skipped = []
        for split in ("train", "val"):
            label_dir = self.m_dict["project_dir"] + "/" + split + "/labels"
            image_dir = self.m_dict["project_dir"] + "/" + split + "/images"
            os.makedirs(label_dir, exist_ok=True)
            os.makedirs(image_dir, exist_ok=True)
            for i in self.move_files_dict[split]:
                source_file_label = self.m_dict["all_label_dir"] + "/" + i + ".txt"
                # Accept any supported image format, not only .png: JPEG
                # datasets were previously skipped without an explanation.
                source_file_image = self._find_image(i)
                if source_file_image is None:
                    skipped.append(i)
                    continue
                try:
                    shutil.copy(source_file_label, label_dir)
                    shutil.copy(source_file_image, image_dir)
                except FileNotFoundError:
                    skipped.append(i)

        if skipped:
            listed = ", ".join(skipped[:10])
            more = " ..." if len(skipped) > 10 else ""
            print(f"Skipped {len(skipped)} item(s) with no matching image: {listed}{more}")
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
    d["all_label_dir"] = (
        "C:/Users/nokai/Desktop/230719_ando_copulation_detection_YORU/labels/labels_fit_chamber"
    )
    d["project_dir"] = (
        "C:/Users/nokai/Desktop/230719_ando_copulation_detection_YORU/labels"
    )
    d["quit"] = False
    fmrd = file_move_random(d)
    fmrd.move()
