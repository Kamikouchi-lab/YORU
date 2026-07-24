import subprocess
import sys
import time
from multiprocessing import Manager, Process

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from pynput import keyboard

sys.path.append("../yoru")

# try:
from yoru.libs.file_operation_grab import file_dialog_tk
from yoru.libs.gui_error import GuiErrorMixin

# except(ModuleNotFoundError):
#     from libs.file_operation_grab import file_dialog_tk


class grab_gui(GuiErrorMixin):
    def __init__(self, m_dict={}):
        print("Refine-gui")
        self.m_dict = m_dict
        self.file_path = "./web/image/YORU_logo.png"

        if self.file_path:
            print("File: " + self.file_path)
        else:
            print("Open-file dialog")

        self.vid = cv2.imread(self.file_path)
        self.height, self.width, _ = self.vid.shape
        self.framecount = 1
        self.current_frame_num = 1
        self.frame = self.vid
        self.process_frame()
        self.grab_count = 0
        self.speed = 1

    def process_frame(self):
        if self.width >= self.height:
            self.im_win_width = 600
            self.im_win_height = self.height * (600 / self.width)
        else:
            self.im_win_width = self.width * (600 / self.height)
            self.im_win_height = 600
        # Resize the frame
        self.frame_re = cv2.resize(
            self.frame, dsize=(int(self.im_win_width), int(self.im_win_height))
        )
        # Create a new frame (filled entirely with black)
        base_frame = np.zeros((600, 600, 3), np.uint8)
        # Place the resized frame at the center of the new frame
        h, w = self.frame_re.shape[:2]
        base_frame[
            int(600 / 2 - h / 2) : int(600 / 2 + h / 2),
            int(600 / 2 - w / 2) : int(600 / 2 + w / 2),
            :,
        ] = self.frame_re
        # Update
        self.frame_re = base_frame

    def gui_configure(self):
        dpg.create_context()
        dpg.configure_app(
            init_file="./config/custom_layout_grab.ini",
            docking=True,
            docking_space=True,
        )
        dpg.create_viewport(title="ASoVi-GUI beta 0.5", width=960, height=900)

        # GUI-settings
        with dpg.texture_registry(show=False):
            # imgwhite = np.ones((self.frameSize[1], self.frameSize[0], 3), np.uint8)
            dpg.add_dynamic_texture(
                width=600,
                height=600,
                default_value=self.frame_to_data(self.frame_re),
                tag="imwin_tag0",
            )
            # dpg.add_raw_texture(
            #     width=600,
            #     height=600,
            #     default_value=self.frame_to_data(self.frame_re),
            #     tag="imwin_tag0",
            #     format=dpg.mvFormat_Float_rgb,
            # )
        imager_window = dpg.generate_uuid()
        with dpg.window(label="Image window", id=imager_window):
            dpg.add_text(label="space1", default_value="    ")
            with dpg.group(horizontal=True):
                dpg.add_text(label="video_dir", default_value="Video file path")
                dpg.add_input_text(
                    tag="video_path", readonly=True, hint="Path/to/movie"
                )
                dpg.add_button(
                    label="Select Video",
                    callback=lambda: self.file_open(),
                    enabled=True,
                )
            dpg.add_text(label="space2", default_value="    ")
            dpg.add_image("imwin_tag0", width=600, height=600)
            dpg.add_slider_int(
                label=" Frame",
                default_value=0,
                min_value=0,
                max_value=self.framecount - 2,
                tag="frame_bar",
                width=600,
                callback=lambda: self.slide_bar_cb(),
                enabled=False,
            )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="streaming movie",
                    default_value=False,
                    tag="streamingChkBox",
                    callback=lambda: self.stream_cb(),
                    enabled=False,
                )
                dpg.add_button(
                    tag="plus_frame",
                    label="+ frame",
                    callback=lambda: self.advance_frame_bt(),
                )
                dpg.add_button(
                    tag="minus_frame",
                    label="- frame",
                    callback=lambda: self.reverse_frame_bt(),
                )
                dpg.add_combo(
                    items=[1, 2, 5, 10, 20, 50, 100, 200, 500],
                    tag="speed_list",
                    default_value=1,
                    width=150,
                    callback=lambda: self.list_of_speed(),
                )

            with dpg.group(horizontal=True):
                dpg.add_text(label="grab_dir", default_value="Save Directory")
                dpg.add_input_text(
                    tag="grab_path", readonly=True, hint="Path/to/save/frame"
                )
                dpg.add_button(
                    label="Select Directory",
                    callback=lambda: self.select_grab_dir(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_text(label="grab_name", default_value="Frame name")
                dpg.add_input_text(
                    tag="save_name", default_value="", width=200, hint="Save frame name"
                )
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Grab Current Frame", callback=lambda: self.grab_btn_cb()
                )
                dpg.add_text(
                    label="Counter",
                    tag="count_frames",
                    default_value=str(self.grab_count) + " frames",
                )
                dpg.add_button(
                    label="Count reset", callback=lambda: self.count_reset_bt()
                )

            dpg.add_separator()
            with dpg.group(horizontal=False):
                dpg.add_button(
                    label="Quit",
                    tag="quit_btn",
                    callback=lambda: self.quit_cb(),
                    enabled=True,
                )

        # setup
        dpg.setup_dearpygui()
        dpg.show_viewport()
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.start()

    def run(self):
        self.gui_configure()
        while dpg.is_dearpygui_running():
            self.plot_callback()
            dpg.render_dearpygui_frame()
            if self.m_dict["quit"]:
                break

    def plot_callback(self) -> None:
        if dpg.get_value("streamingChkBox"):
            if (
                int(self.speed) + dpg.get_value("frame_bar")
                > self.framecount - int(self.speed) - 1
            ):
                dpg.set_value("frame_bar", 0)
            else:
                dpg.set_value("frame_bar", int(self.speed) + dpg.get_value("frame_bar"))
            self.slide_bar_cb()

    def file_open(self):
        try:
            self.fd_tk = file_dialog_tk(self.m_dict)
            self.file_path = self.fd_tk.video_file_open()

            if not self.file_path:
                # Do nothing if the dialog was cancelled
                print("No video file selected", flush=True)
                return

            print("File: " + self.file_path, flush=True)
            self.vid = cv2.VideoCapture(self.file_path)
            if not self.vid.isOpened():
                raise IOError(f"Could not open movie file: {self.file_path}")
            self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.framecount = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_num = 0
            self.status, self.frame = self.vid.read()
            if not self.status or self.frame is None:
                raise IOError(
                    f"Could not read frames from movie file: {self.file_path}"
                )
            self.process_frame()
            print("Movie size: ", self.width, self.height, flush=True)
            dpg.configure_item("frame_bar", max_value=self.framecount - 2)
            dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))
            dpg.enable_item("streamingChkBox")
            dpg.enable_item("frame_bar")
        except Exception as e:
            # Roll back to a safe state so the streaming loop and frame buttons
            # do not later operate on a broken capture (cv2.resize(None) crash).
            if getattr(self, "vid", None) is not None:
                self.vid.release()
            self.vid = None
            self.frame = None
            self.status = False
            if dpg.does_item_exist("streamingChkBox"):
                dpg.set_value("streamingChkBox", False)
                dpg.disable_item("streamingChkBox")
            if dpg.does_item_exist("frame_bar"):
                dpg.disable_item("frame_bar")
            self._report_error("Failed to open video file", e)

    # Shortcut key settings
    def on_key_press(self, key):
        try:
            if key == keyboard.Key.right:
                self.advance_frame_bt()
            elif key == keyboard.Key.left:
                self.reverse_frame_bt()
            elif (
                key == keyboard.Key.alt_l or key == keyboard.Key.alt_r
            ):  # 'ctrl_l' represents the left Ctrl key
                self.grab_btn_cb()
        except AttributeError:
            pass

    def select_grab_dir(self):
        self.fd_tk = file_dialog_tk(self.m_dict)
        self.grab_dir = self.fd_tk.grab_dir_open()

    def slide_bar_cb(self):
        if getattr(self, "vid", None) is None:
            return
        self.current_frame_num = dpg.get_value("frame_bar")
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        self.status, self.frame = self.vid.read()
        if not self.status or self.frame is None:
            return
        self.process_frame()

        dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))

    def list_of_speed(self):
        tf = dpg.get_value("speed_list")
        self.speed = tf

    def advance_frame_bt(self):
        if self.current_frame_num < self.framecount - 2:
            self.current_frame_num = self.current_frame_num + int(self.speed)
            if self.current_frame_num >= self.framecount - 2:
                self.current_frame_num = self.framecount - 2
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
            self.status, self.frame = self.vid.read()
            self.process_frame()
            dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))
            dpg.set_value("frame_bar", self.current_frame_num)
        else:
            print("final frame")

    def reverse_frame_bt(self):
        if self.current_frame_num > 0:
            self.current_frame_num = self.current_frame_num - int(self.speed)
            if self.current_frame_num <= 0:
                self.current_frame_num = 0
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
            self.status, self.frame = self.vid.read()
            self.process_frame()
            dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))
            dpg.set_value("frame_bar", self.current_frame_num)
        else:
            print("initial frame")

    def grab_btn_cb(self):
        try:
            self.grab_name = dpg.get_value("save_name")
            grab_dir = getattr(self, "grab_dir", None)
            if not (self.grab_name and grab_dir):
                raise ValueError(
                    "Save directory and frame name must be set before grabbing a frame."
                )
            self.grab_path = (
                grab_dir
                + "/"
                + self.grab_name
                + "_"
                + str(self.current_frame_num)
                + ".png"
            )
            print(self.grab_path, flush=True)
            cap = cv2.VideoCapture(self.file_path)
            if not cap.isOpened():
                raise IOError(f"Could not open movie file: {self.file_path}")
            # Move to the specified frame number
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
            # Read the frame
            ret, frame = cap.read()
            cap.release()
            # Save the frame if it was read successfully
            if not ret or frame is None:
                raise IOError(
                    f"Could not read frame {self.current_frame_num} from {self.file_path}"
                )
            if not cv2.imwrite(self.grab_path, frame):
                raise IOError(f"Could not write frame image: {self.grab_path}")
            self.grab_count = self.grab_count + 1
            dpg.set_value("count_frames", str(self.grab_count) + " frames")
        except Exception as e:
            self._report_error("Failed to grab frame", e)

    def count_reset_bt(self):
        self.grab_count = 0
        dpg.set_value("count_frames", str(self.grab_count) + " frames")

    def stream_cb(self):
        if dpg.get_value("streamingChkBox"):
            dpg.disable_item("frame_bar")
            dpg.disable_item("minus_frame")
            dpg.disable_item("plus_frame")
        else:
            dpg.enable_item("frame_bar")
            dpg.enable_item("minus_frame")
            dpg.enable_item("plus_frame")
        pass

    def frame_to_data(self, frame):
        # raw image streaming
        # frame_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # self.texture_data = np.true_divide(frame_data.ravel(), 255.0)
        # data = np.asfarray(self.texture_data.ravel(), dtype="f")
        data = np.true_divide(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA), 255)
        return data

    def quit_cb(self):
        print("quit_pushed")
        self.m_dict["quit"] = True
        # subprocess.call(["python", "train_gui.py"])
        dpg.destroy_context()  # <-- moved from __del__

    def __del__(self):
        if hasattr(self, "quit"):
            self.m_dict["quit"] = True
        print("=== GUI window quit ===")
        dpg.destroy_context()


def main():
    d = {}
    d["quit"] = False
    grabWin = grab_gui(d)
    grabWin.run()


if __name__ == "__main__":
    main()
