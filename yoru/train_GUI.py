import os
import subprocess
import sys
from cProfile import label
from multiprocessing import Manager, Process

import dearpygui.dearpygui as dpg
import yaml

# if __name__ == "__main__":
    # Run directly

import sys
sys.path.append('../yoru')

# try:
from yoru.libs.create_yaml_train import create_project
from yoru.libs.file_operation_train import file_move_random, file_dialog_tk
from yoru.libs.init_train import init_train, loadingParam
from yoru.grab_GUI import main as grab_main
# import yoru.app as YORU

# except(ModuleNotFoundError):
#     from libs.create_yaml_train import create_project
#     from libs.file_operation_train import file_move_random, file_dialog_tk
#     from libs.init_train import init_train, loadingParam
#     from grab_GUI import main as grab_main
#     import app as YORU
# else:
#     # from .libs import threshold
#     from .libs.create_yaml_train import create_project
#     from yoru.libs.file_operation_train import file_move_random, file_dialog_tk
#     from yoru.libs.init_train import init_train, loadingParam


class yolo_train:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict
        self.fd_tk = file_dialog_tk(self.m_dict)

    def startDPG(self):
        dpg.create_context()
        dpg.configure_app(
            init_file="./logs/custom_layout_train_gui.ini",
            docking=True,
            docking_space=True,
        )
        dpg.create_viewport(title="Training Model", width=940, height=850)
        imager_window = dpg.generate_uuid()

        # theme
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Tab, (55, 140, 23), category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_TabHovered,
                    (100, 140, 23),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_TitleBg, (200, 140, 23), category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_Text, (230, 230, 230), category=dpg.mvThemeCat_Core
                )

                dpg.add_theme_style(
                    dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core
                )

        dpg.bind_theme(global_theme)

        # add a font registry
        # with dpg.font_registry():
        #     # first argument ids the path to the .ttf or .otf file
        #     default_font = dpg.add_font("./fonts/myriad-pro-cufonfonts/MYRIADPRO-REGULAR.OTF", 15)
        # # set font of specific widget
        # dpg.bind_font(default_font)

        # imager-window

        with dpg.window(label="YORU - Train", id=imager_window):
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step1: Creating project     ")
                dpg.add_text(tag="step1_state", default_value="Yet")
            with dpg.group(horizontal=True):
                dpg.add_text(label="project_dir", default_value="Project Directory")
                dpg.add_input_text(
                    tag="project_path", readonly=True, hint="Path/to/project"
                )
                dpg.add_button(
                    label="Select Directory",
                    callback=lambda: self.fd_tk.pro_dir_open(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Load YOLO project",
                    tag="load_btn",
                    width=150,
                    height=30,
                    callback=lambda: self.load_pr_dir(),
                    enabled=True,
                )
                dpg.add_text(default_value="     or     ")
                dpg.add_input_text(
                    tag="pro_name",
                    default_value="",
                    width=200,
                    hint="YOLO project name",
                )
                dpg.add_button(
                    label="Create YOLO project",
                    tag="cre_btn",
                    width=150,
                    height=30,
                    callback=lambda: self.create_pr_dir(),
                    enabled=True,
                )
            dpg.add_separator()

            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step2: Screenshot GUI     ")
                dpg.add_text(tag="step2_state", default_value="Yet")
            dpg.add_button(
                label="Run Grab GUI",
                tag="grab_btn",
                width=150,
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
                dpg.add_text(default_value="Step4: Prepare image and txt flies     ")
                dpg.add_text(tag="step4_state", default_value="Yet")
            dpg.add_button(
                label="Move Label Imgaes",
                tag="move_label_images",
                callback=lambda: self.flie_move_bt(),
                enabled=False,
            )
            dpg.add_text(
                "    -project_name\n"
                "         |-all_label_images - classes.txt\n"
                "         |-train   ~80% of labeled data\n"
                "         |  |- images - *.png\n"
                "         |  | \n"
                "         |  |- labels - *.txt\n"
                "         | \n"
                "         |- val    ~20% of labeled data\n"
                "             |- images - *.png\n"
                "             |    \n"
                "             |- labels - *.txt\n"
                "\n"
            )
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step5: Creating yaml file     ")
                dpg.add_text(tag="step5_state", default_value="Yet")
            with dpg.group(horizontal=True):
                dpg.add_text(label="class_path_in", default_value="classes.txt Path")
                dpg.add_input_text(
                    tag="classes_path", readonly=True, hint="Path/to/classes.txt"
                )
                dpg.add_button(
                    label="Select Path",
                    callback=lambda: self.fd_tk.class_txt_open(),
                    enabled=True,
                )
            dpg.add_button(
                label="Add class info in YAML file",
                tag="cre_yaml_btn",
                width=200,
                height=30,
                callback=lambda: self.add_class_file(),
                enabled=True,
            )
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step6: Train dataset     ")
                dpg.add_text(tag="step6_state", default_value="Yet")
            with dpg.group(horizontal=True):
                dpg.add_text(label="yaml_path", default_value="YAML Path")
                dpg.add_input_text(
                    tag="yaml_file_path", readonly=True, hint="Path/to/config.yaml"
                )
                dpg.add_button(
                    label="Select File",
                    callback=lambda: self.fd_tk.dataset_file_open(),
                    enabled=True,
                )
            dpg.add_text(default_value="Training condition")
            with dpg.group(horizontal=True):
                dpg.add_text(label="weight", default_value="    Weight")
                dpg.add_combo(
                    items=self.m_dict["weight_list"],
                    tag="weight_list",
                    default_value=self.m_dict["weight"],
                    width=150,
                    callback=lambda: self.list_of_weight(),
                )
            with dpg.group(horizontal=True):
                dpg.add_text(label="epoch_num", default_value="    Epoc")
                dpg.add_input_text(
                    tag="epoc_num_in",
                    default_value=self.m_dict["epoch"],
                    width=150,
                    callback=lambda: self.in_epoch(),
                )
            with dpg.group(horizontal=True):
                dpg.add_text(label="img_num", default_value="    Image size (width)")
                dpg.add_input_text(
                    tag="img_num_in",
                    default_value=self.m_dict["img"],
                    width=150,
                    callback=lambda: self.in_img(),
                )
            with dpg.group(horizontal=True):
                dpg.add_text(label="batch_num", default_value="    batch")
                dpg.add_input_text(
                    tag="batch_num_in",
                    default_value=self.m_dict["batch"],
                    width=150,
                    callback=lambda: self.in_batch(),
                )
            dpg.add_button(
                label="Train YOLOv5",
                tag="str_btn",
                width=100,
                height=30,
                callback=lambda: self.run_yolov5(),
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

        dpg.setup_dearpygui()
        dpg.show_viewport()

    def plot_callback(self) -> None:
        a = 1

    def run(self):
        self.startDPG()
        while dpg.is_dearpygui_running():
            self.plot_callback()
            dpg.render_dearpygui_frame()
            if self.m_dict["quit"]:  # <-- this line was modified
                if self.m_dict["back_to_home"]:
                    # subprocess.call(["python", "app.py"])
                    from yoru import app as YORU
                    YORU.main()
                dpg.destroy_context()
                break

    def create_pr_dir(self):
        print("create project")
        self.m_dict["project_name"] = dpg.get_value("pro_name")
        self.m_dict["project_dir"] = (
            self.m_dict["project_path"] + "/" + self.m_dict["project_name"]
        )
        file_path = self.m_dict["project_dir"] + "/config.yaml"
        if not os.path.exists(file_path):
            os.makedirs(self.m_dict["project_dir"])
            os.makedirs(self.m_dict["project_dir"] + "/train")
            os.makedirs(self.m_dict["project_dir"] + "/train/images")
            os.makedirs(self.m_dict["project_dir"] + "/train/labels")
            os.makedirs(self.m_dict["project_dir"] + "/val")
            os.makedirs(self.m_dict["project_dir"] + "/val/images")
            os.makedirs(self.m_dict["project_dir"] + "/val/labels")
            os.makedirs(self.m_dict["project_dir"] + "/all_label_images")
            self.m_dict["yaml_path"] = self.m_dict["project_dir"] + "/config.yaml"
            self.m_dict["all_label_dir"] = (
                self.m_dict["project_dir"] + "/all_label_images"
            )
            cr_project = create_project(self.m_dict)
            cr_project.create_yaml()

            dpg.set_value("yaml_file_path", self.m_dict["yaml_path"])
            dpg.enable_item("move_label_images")
            dpg.set_value("step1_state", "Complete!!")
        else:
            print(f"The project already exists.")
            dpg.enable_item("move_label_images")

    def load_pr_dir(self):
        print("load project")
        self.m_dict["project_dir"] = self.m_dict["project_path"]
        file_path = self.m_dict["project_dir"] + "/config.yaml"
        if not os.path.exists(file_path):
            print("please create a project")
            return None
        with open(file_path, "r") as yf:
            data = yaml.safe_load(yf)
            self.m_dict["yaml_path"] = data["yaml_path"]
            self.m_dict["all_label_dir"] = (
                self.m_dict["project_dir"] + "/all_label_images"
            )
            dpg.set_value("yaml_file_path", self.m_dict["yaml_path"])
            print(f"load complete")
            dpg.enable_item("move_label_images")
            dpg.set_value("step1_state", "Complete!!")

    def grab_bt(self):
        # print("quit_pushed")
        # self.m_dict["quit"] = True
        subprocess.call(["python", "./yoru/grab_GUI.py"])
        dpg.set_value("step2_state", "Complete!!")
        # dpg.destroy_context()  # <-- moved from __del__

    def labelImg_bt(self):
        # print("quit_pushed")
        # self.m_dict["quit"] = True
        subprocess.call(["python", "./libs/labelImg/labelImg.py"])
        dpg.set_value("step3_state", "Complete!!")
        # dpg.destroy_context()  # <-- moved from __del__

    def quit_cb(self):
        print("quit_pushed")
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def home_cb(self):
        print("Back home")
        self.m_dict["back_to_home"] = True
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def add_class_file(self):
        cr_project = create_project(self.m_dict)
        if "classes.txt" in self.m_dict["classes_path"] and os.path.exists(
            self.m_dict["classes_path"]
        ):
            cr_project.add_class_info()
        dpg.set_value("step5_state", "Complete!!")

    def list_of_weight(self):
        tf = dpg.get_value("weight_list")
        self.m_dict["weight"] = tf

    def in_epoch(self):
        tf = dpg.get_value("epoc_num_in")
        self.m_dict["epoch"] = tf

    def flie_move_bt(self):
        self.fmrd = file_move_random(self.m_dict)
        print(self.m_dict["all_label_dir"])
        self.fmrd.move()
        dpg.disable_item("move_label_images")
        dpg.set_value("step4_state", "Complete!!")

    def in_img(self):
        tf = dpg.get_value("img_num_in")
        self.m_dict["img"] = tf

    def in_batch(self):
        tf = dpg.get_value("batch_num_in")
        self.m_dict["batch"] = tf

    def run_yolov5(self):
        # train
        cmd = [
            "python",
            "./libs/yolov5/train.py",
            "--imgsz",
            str(self.m_dict["img"]),
            "--batch-size",
            str(self.m_dict["batch"]),
            "--epochs",
            str(self.m_dict["epoch"]),
            "--data",
            str(self.m_dict["yaml_path"]),
            "--weights",
            str(self.m_dict["weight"]),
            "--project",
            str(self.m_dict["project_dir"]),
        ]

        self.patience = False

        if self.patience:
            cmd.extend(["--patience", str(0)])

        cr_project = create_project(self.m_dict)
        cr_project.add_training_info()
        print("added the information in yaml file")

        try:
            # subprocessを使用して学習スクリプトを非同期で実行します。
            subprocess.Popen(cmd)
            print("start training ")
        except Exception as e:
            print("error: ", e)

        dpg.set_value("step6_state", "Complete!!")

    def __del__(self):
        print("=== GUI window quit ===")


def main():
    print("Starting training GUI")
    with Manager() as manager:
        d = manager.dict()

        # initialize m_dict with init_analysis
        init = init_train(m_dict=d)

        gui = yolo_train(m_dict=d)
        process_pool = []
        prc_gui = Process(target=gui.run)
        process_pool.append(prc_gui)  # <-- this line was added
        prc_gui.start()
        prc_gui.join()


if __name__ == "__main__":
    main()
