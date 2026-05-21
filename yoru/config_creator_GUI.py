"""YORU Real-time Config Creator GUI

Creates a YAML configuration file for the real-time analysis workflow
through a guided DearPyGui interface.  All YAML keys can be set via
widgets; class information is loaded automatically when a .pt model
file is selected.
"""

import glob
import os
import sys
import threading
import tkinter
import tkinter.filedialog as filedialog

import dearpygui.dearpygui as dpg
import serial.tools.list_ports
import yaml

sys.path.append("../yoru")


class ConfigCreatorGUI:
    def __init__(self):
        self.class_list = ["None"]
        self.com_list = self._get_com_ports()
        self.plugin_list = self._get_plugins()
        self.model_type_list = [
            "auto", "yolov5", "yolov8", "yolo11", "rtdetr",
            "fasterrcnn", "maskrcnn", "ssd",
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_com_ports(self):
        ports = serial.tools.list_ports.comports()
        result = [p.device for p in ports]
        return result if result else ["None"]

    def _get_plugins(self):
        paths = glob.glob("./trigger_plugins/*.py", recursive=True)
        result = [os.path.splitext(os.path.basename(p))[0] for p in paths]
        return result if result else ["standard_arduino"]

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------
    def startDPG(self):
        dpg.create_context()
        dpg.configure_app(
            init_file="./logs/custom_layout_config_creator.ini",
            docking=True,
            docking_space=True,
        )
        dpg.create_viewport(title="YORU - Config Creator", width=820, height=950)

        # ── Theme (same palette as train_GUI) ──────────────────────────
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

        # ── Main window ────────────────────────────────────────────────
        with dpg.window(label="YORU - Config Creator", tag="main_window"):

            # ── General ───────────────────────────────────────────────
            dpg.add_text("General")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Project Name      ")
                dpg.add_input_text(
                    tag="cfg_name", default_value="my_project", width=340
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Export Directory  ")
                dpg.add_input_text(
                    tag="cfg_export", default_value="", readonly=True, width=300
                )
                dpg.add_button(
                    label="Select", callback=lambda: self._select_export_dir()
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Export File Name  ")
                dpg.add_input_text(
                    tag="cfg_export_name", default_value="experiment", width=340
                )
            dpg.add_text(" ")

            # ── Model ─────────────────────────────────────────────────
            dpg.add_text("Model")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Model Path (.pt)  ")
                dpg.add_input_text(
                    tag="cfg_model_path", default_value="", readonly=True, width=300
                )
                dpg.add_button(
                    label="Select", callback=lambda: self._select_model_file()
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Model Type        ")
                dpg.add_combo(
                    items=self.model_type_list,
                    tag="cfg_model_type",
                    default_value="auto",
                    width=150,
                )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Auto-start detection on launch",
                    tag="cfg_yolo_detection",
                    default_value=False,
                )
            with dpg.group(horizontal=True):
                dpg.add_checkbox(
                    label="Enable Trigger",
                    tag="cfg_trigger_enable",
                    default_value=False,
                )
            dpg.add_text(" ")

            # ── Class Information (auto-loaded from model) ─────────────
            dpg.add_text("Class Information")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("  Auto-loaded when a .pt file is selected  ")
                dpg.add_text(tag="cfg_class_loading_state", default_value="")
            dpg.add_listbox(
                items=self.class_list,
                tag="cfg_class_list_box",
                num_items=4,
                width=420,
            )
            dpg.add_text(" ")

            # ── Capture Style ─────────────────────────────────────────
            dpg.add_text("Capture Style")
            dpg.add_separator()
            dpg.add_checkbox(
                label="Use screen capture (MSS) instead of camera",
                tag="cfg_stream_mss",
                default_value=False,
            )
            dpg.add_text(" ")

            # ── Trigger ───────────────────────────────────────────────
            dpg.add_text("Trigger")
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Confidence Threshold  ")
                dpg.add_slider_float(
                    tag="cfg_trigger_threshold",
                    default_value=0.3,
                    min_value=0.0,
                    max_value=1.0,
                    width=220,
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Trigger Class         ")
                dpg.add_combo(
                    items=self.class_list,
                    tag="cfg_trigger_class",
                    default_value="None",
                    width=220,
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Arduino COM Port      ")
                dpg.add_combo(
                    items=self.com_list,
                    tag="cfg_arduino_com",
                    default_value=self.com_list[0] if self.com_list else "None",
                    width=110,
                )
                dpg.add_button(
                    label="Reload COM", callback=lambda: self._reload_com()
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Trigger Pin No.       ")
                dpg.add_input_text(
                    tag="cfg_trigger_pin",
                    default_value="13",
                    width=100,
                    hint="integer",
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Trigger Plugin        ")
                dpg.add_combo(
                    items=self.plugin_list,
                    tag="cfg_trigger_style",
                    default_value=(
                        self.plugin_list[0] if self.plugin_list else "standard_arduino"
                    ),
                    width=200,
                )
                dpg.add_button(
                    label="Reload Plugins", callback=lambda: self._reload_plugins()
                )
            dpg.add_text(" ")

            # ── Hardware (Camera) ──────────────────────────────────────
            dpg.add_text("Hardware (Camera)")
            dpg.add_separator()
            dpg.add_checkbox(
                label="Use camera", tag="cfg_use_camera", default_value=True
            )
            with dpg.group(horizontal=True):
                dpg.add_text("Camera ID          ")
                dpg.add_input_text(
                    tag="cfg_camera_id", default_value="0", width=100, hint="integer"
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Width (px)         ")
                dpg.add_input_text(
                    tag="cfg_camera_width",
                    default_value="640",
                    width=100,
                    hint="integer",
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Height (px)        ")
                dpg.add_input_text(
                    tag="cfg_camera_height",
                    default_value="480",
                    width=100,
                    hint="integer",
                )
            with dpg.group(horizontal=True):
                dpg.add_text("Scale              ")
                dpg.add_input_text(
                    tag="cfg_camera_scale",
                    default_value="1",
                    width=100,
                    hint="number",
                )
            with dpg.group(horizontal=True):
                dpg.add_text("FPS                ")
                dpg.add_input_text(
                    tag="cfg_camera_fps",
                    default_value="30",
                    width=100,
                    hint="integer",
                )
            dpg.add_checkbox(
                label="Show OpenCV window (camera_imshow)",
                tag="cfg_camera_imshow",
                default_value=False,
            )
            dpg.add_text(" ")

            # ── Save ──────────────────────────────────────────────────
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_text("Output YAML Path   ")
                dpg.add_input_text(
                    tag="cfg_output_path",
                    default_value="",
                    readonly=True,
                    width=340,
                )
                dpg.add_button(
                    label="Select", callback=lambda: self._select_output_path()
                )
            dpg.add_text(tag="cfg_save_status", default_value="")
            dpg.add_text(" ")
            with dpg.group(horizontal=False):
                dpg.add_button(
                    label="Save Config",
                    tag="cfg_save_btn",
                    width=120,
                    height=30,
                    callback=lambda: self._save_config(),
                )
                dpg.add_button(
                    label="Quit",
                    tag="cfg_quit_btn",
                    callback=lambda: self.quit_cb(),
                )

        dpg.setup_dearpygui()
        dpg.show_viewport()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _select_model_file(self):
        root = tkinter.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select YOLO model file",
            filetypes=[("Model weights", "*.pt"), ("All files", "*.*")],
        )
        root.destroy()
        if not path:
            return
        dpg.set_value("cfg_model_path", path)
        dpg.set_value("cfg_class_loading_state", " Loading...")
        model_type = dpg.get_value("cfg_model_type")
        t = threading.Thread(
            target=self._load_classes, args=(path, model_type), daemon=True
        )
        t.start()

    def _load_classes(self, model_path, model_type):
        """Load class names from the selected model in a background thread."""
        try:
            from yoru.libs.yolo_wrapper import load_yolo_model

            model = load_yolo_model(model_path, model_type)
            names_dict = model.names  # {0: "class0", 1: "class1", ...}
            class_names = [names_dict[i] for i in sorted(names_dict.keys())]
            self.class_list = class_names + ["None"]
            dpg.configure_item("cfg_class_list_box", items=self.class_list)
            dpg.configure_item("cfg_trigger_class", items=self.class_list)
            if class_names:
                dpg.set_value("cfg_trigger_class", class_names[0])
            dpg.set_value("cfg_class_loading_state", f" {len(class_names)} classes loaded")
        except Exception as e:
            dpg.set_value("cfg_class_loading_state", f" Error: {e}")

    def _select_export_dir(self):
        root = tkinter.Tk()
        root.withdraw()
        path = filedialog.askdirectory(title="Select export directory")
        root.destroy()
        if path:
            dpg.set_value("cfg_export", path)

    def _select_output_path(self):
        root = tkinter.Tk()
        root.withdraw()
        path = filedialog.asksaveasfilename(
            title="Save config as",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            dpg.set_value("cfg_output_path", path)

    def _reload_com(self):
        self.com_list = self._get_com_ports()
        dpg.configure_item("cfg_arduino_com", items=self.com_list)

    def _reload_plugins(self):
        self.plugin_list = self._get_plugins()
        dpg.configure_item("cfg_trigger_style", items=self.plugin_list)

    def _save_config(self):
        out_path = dpg.get_value("cfg_output_path")
        if not out_path:
            dpg.set_value("cfg_save_status", "Please select an output path first.")
            return

        def _int(tag, fallback):
            try:
                return int(dpg.get_value(tag))
            except (ValueError, TypeError):
                return fallback

        def _float(tag, fallback):
            try:
                return float(dpg.get_value(tag))
            except (ValueError, TypeError):
                return fallback

        config = {
            "name": dpg.get_value("cfg_name"),
            "export": dpg.get_value("cfg_export"),
            "export_name": dpg.get_value("cfg_export_name"),
            "model": {
                "yolo_detection": bool(dpg.get_value("cfg_yolo_detection")),
                "yolo_model_path": dpg.get_value("cfg_model_path"),
                "yolo_model_type": dpg.get_value("cfg_model_type"),
                "Trigger": bool(dpg.get_value("cfg_trigger_enable")),
            },
            "capture_style": {
                "stream_MSS": bool(dpg.get_value("cfg_stream_mss")),
            },
            "trigger": {
                "trigger_threshold_configuration": round(
                    _float("cfg_trigger_threshold", 0.3), 4
                ),
                "trigger_class": dpg.get_value("cfg_trigger_class"),
                "Arduino_COM": dpg.get_value("cfg_arduino_com"),
                "trigger_pin": _int("cfg_trigger_pin", 13),
                "trigger_style": dpg.get_value("cfg_trigger_style"),
            },
            "hardware": {
                "use_camera": bool(dpg.get_value("cfg_use_camera")),
                "camera_id": _int("cfg_camera_id", 0),
                "camera_width": _int("cfg_camera_width", 640),
                "camera_height": _int("cfg_camera_height", 480),
                "camera_scale": _float("cfg_camera_scale", 1.0),
                "camera_fps": _int("cfg_camera_fps", 30),
                "camera_imshow": bool(dpg.get_value("cfg_camera_imshow")),
            },
        }

        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        dpg.set_value("cfg_save_status", f"Saved: {out_path}")

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    def run(self):
        self.startDPG()
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()

    def quit_cb(self):
        dpg.destroy_context()

    def __del__(self):
        print("=== Config Creator GUI quit ===")


def main():
    gui = ConfigCreatorGUI()
    gui.run()


if __name__ == "__main__":
    main()
