# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import os

import cv2
import dearpygui.dearpygui as dpg
import numpy as np
from pynput import keyboard

from yoru.gui_base import apply_default_theme, frame_to_data_rgba, process_frame as _process_frame
from yoru.libs.file_operation_grab import file_dialog_tk


class grab_gui:
    def __init__(self, m_dict={}):
        print("Grab-gui")
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
        self.frame_re = _process_frame(self.frame, 600)

    def gui_configure(self):
        dpg.create_context()
        dpg.configure_app(
            init_file="./config/custom_layout_grab.ini",
            docking=True,
            docking_space=True,
        )
        dpg.create_viewport(title="YORU - Frame Capture", width=1000, height=800, max_width=1000, max_height=800)

        # Theme
        apply_default_theme()

        # GUI-settings
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=600,
                height=600,
                default_value=self.frame_to_data(self.frame_re),
                tag="imwin_tag0",
            )
        imager_window = dpg.generate_uuid()
        with dpg.window(label="Frame Capture", id=imager_window):
            # Video Source
            dpg.add_text(default_value="Video Source")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Video Path      ")
                dpg.add_input_text(
                    tag="video_path", readonly=True, hint="Path/to/movie", width=350
                )
                dpg.add_spacer(width=4)
                dpg.add_button(
                    label="Select Video",
                    callback=lambda: self.file_open(),
                    enabled=True,
                )

            # Preview
            dpg.add_spacer(height=4)
            dpg.add_text(default_value="Preview")
            dpg.add_separator()
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
                    label="Streaming",
                    default_value=False,
                    tag="streamingChkBox",
                    callback=lambda: self.stream_cb(),
                    enabled=False,
                )
                dpg.add_spacer(width=12)
                dpg.add_button(
                    tag="minus_frame",
                    label="< Prev",
                    callback=lambda: self.reverse_frame_bt(),
                )
                dpg.add_button(
                    tag="plus_frame",
                    label="Next >",
                    callback=lambda: self.advance_frame_bt(),
                )
                dpg.add_spacer(width=8)
                dpg.add_text(default_value="Speed")
                dpg.add_combo(
                    items=[1, 2, 5, 10, 20, 50, 100, 200, 500],
                    tag="speed_list",
                    default_value=1,
                    width=80,
                    callback=lambda: self.list_of_speed(),
                )

            # Save Frame
            dpg.add_spacer(height=4)
            dpg.add_text(default_value="Save Frame")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Save Directory  ")
                dpg.add_input_text(
                    tag="grab_path", readonly=True, hint="Path/to/save/frame", width=350
                )
                dpg.add_spacer(width=4)
                dpg.add_button(
                    label="Select Directory",
                    callback=lambda: self.select_grab_dir(),
                    enabled=True,
                )
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Frame Name      ")
                dpg.add_input_text(
                    tag="save_name", default_value="", width=200, hint="Save frame name"
                )
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Grab Current Frame",
                    width=160,
                    height=30,
                    callback=lambda: self.grab_btn_cb(),
                )
                dpg.add_spacer(width=8)
                dpg.add_text(
                    tag="count_frames",
                    default_value=str(self.grab_count) + " frames grabbed",
                )
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="Reset Count", callback=lambda: self.count_reset_bt()
                )

            # Navigation
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_button(
                label="Quit",
                tag="quit_btn",
                callback=lambda: self.quit_cb(),
                enabled=True,
            )

        # setup
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self._kb_listener = keyboard.Listener(on_press=self.on_key_press)
        self._kb_listener.start()

    def run(self):
        self.gui_configure()
        try:
            while dpg.is_dearpygui_running():
                self.plot_callback()
                dpg.render_dearpygui_frame()
                if self.m_dict["quit"]:
                    break
        finally:
            if hasattr(self, "_kb_listener"):
                self._kb_listener.stop()

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
        self.fd_tk = file_dialog_tk(self.m_dict)
        self.file_path = self.fd_tk.video_file_open()

        if self.file_path:
            print("File: " + self.file_path)
        else:
            print("Failed open files")

        self.vid = cv2.VideoCapture(self.file_path)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.framecount = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0
        self.status, self.frame = self.vid.read()
        self.process_frame()
        print("Movie size: ", self.width, self.height)
        dpg.configure_item("frame_bar", max_value=self.framecount - 2)
        dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))
        dpg.enable_item("streamingChkBox")
        dpg.enable_item("frame_bar")

    # ショートカットキー設定
    def on_key_press(self, key):
        try:
            if key == keyboard.Key.right:
                self.advance_frame_bt()
            elif key == keyboard.Key.left:
                self.reverse_frame_bt()
            elif (
                key == keyboard.Key.alt_l or key == keyboard.Key.alt_r
            ):  # 'ctrl_l'は左Ctrlキーを表します
                self.grab_btn_cb()
        except AttributeError:
            pass

    def select_grab_dir(self):
        self.fd_tk = file_dialog_tk(self.m_dict)
        self.grab_dir = self.fd_tk.grab_dir_open()

    def slide_bar_cb(self):
        self.current_frame_num = dpg.get_value("frame_bar")
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        self.status, self.frame = self.vid.read()
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
        self.grab_name = dpg.get_value("save_name")
        if not self.grab_name or not hasattr(self, "grab_dir") or not self.grab_dir:
            print("Not selected file dir or name")
            return
        self.grab_path = os.path.join(
            self.grab_dir,
            f"{self.grab_name}_{self.current_frame_num}.png",
        )
        print(self.grab_path)
        # Reuse the existing VideoCapture instead of creating a new one
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.vid.read()
        if ret:
            cv2.imwrite(self.grab_path, frame)
            self.grab_count = self.grab_count + 1
            dpg.set_value("count_frames", str(self.grab_count) + " frames")
        else:
            print("Failed to read frame for grab")

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
        return frame_to_data_rgba(frame)

    def quit_cb(self):
        print("quit_pushed")
        self.m_dict["quit"] = True
        # subprocess.call(["python", "train_gui.py"])
        dpg.destroy_context()  # <-- moved from __del__

    def __del__(self):
        if hasattr(self, "m_dict"):
            self.m_dict["quit"] = True
        print("=== GUI window quit ===")


def main():
    d = {}
    d["quit"] = False
    grabWin = grab_gui(d)
    grabWin.run()


if __name__ == "__main__":
    main()
