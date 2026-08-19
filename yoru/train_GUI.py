# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import os
import re
import subprocess
import sys
import threading
import time
from multiprocessing import Manager, Process

import dearpygui.dearpygui as dpg
import yaml

from yoru.libs.create_yaml_train import create_project
from yoru.libs.file_operation_train import file_dialog_tk, file_move_random
from yoru.libs.init_train import init_train
from yoru.libs.train_progress import ProgressPrinter
from yoru.libs.train_stop import (
    clear_stop,
    request_stop,
    stop_file_for,
    terminate_process_tree,
)
from yoru.libs.vram_estimate import (
    check_training_vram,
    get_gpu_info,
    read_dataset_stats,
)


class yoru_train:
    def __init__(self, m_dict=None):
        self.m_dict = m_dict if m_dict is not None else {}
        self.fd_tk = file_dialog_tk(self.m_dict)

        # VRAM estimate caches.  The estimate is recomputed every frame from
        # whatever is on screen, so everything that costs more than a dict
        # lookup -- polling the driver, walking the label directory -- is
        # cached and refreshed only when its input actually changed.
        self._gpu_info = None
        self._vram_key = None
        self._dataset_stats = None
        self._dataset_key = None
        self._last_verdict = None
        self._gpu_ready = False
        # A plain flag, not a threading.Event: main() hands this object to
        # Process(target=gui.run), which pickles it, and a lock cannot be
        # pickled.  The poller is a daemon thread, so a lazy stop is enough.
        self._gpu_stop = False
        self._last_valid_nums = (640, 16)

        # Training subprocess state.  The handle only means anything in the
        # process that started it -- the same one that runs the callbacks --
        # so what the render loop needs is mirrored in m_dict instead.
        self._train_proc = None
        self._stop_file = None
        self._stop_ui_state = None

    def startDPG(self):
        dpg.create_context()
        dpg.configure_app(
            init_file="./logs/custom_layout_train_gui.ini",
            docking=True,
            docking_space=True,
        )
        dpg.create_viewport(title="YORU - Training", width=1000, height=800)
        imager_window = dpg.generate_uuid()

        # ── Global theme ──────────────────────────────────────────────────────
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # Backgrounds
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg,            (28, 30, 33),     category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg,             (35, 37, 40),     category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,             (45, 47, 52),     category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,      (60, 63, 68),     category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,       (55, 58, 63),     category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg,             (32, 34, 37),     category=dpg.mvThemeCat_Core)
                # Title bar
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg,             (35, 37, 40),     category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,       (45, 120, 18),    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed,    (28, 30, 33),     category=dpg.mvThemeCat_Core)
                # Tabs
                dpg.add_theme_color(dpg.mvThemeCol_Tab,                 (45, 120, 18),    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered,          (80, 170, 35),    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TabActive,           (65, 150, 25),    category=dpg.mvThemeCat_Core)
                # Buttons (default: muted green)
                dpg.add_theme_color(dpg.mvThemeCol_Button,              (45, 85, 45),     category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,       (65, 115, 65),    category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,        (35, 70, 35),     category=dpg.mvThemeCat_Core)
                # Scrollbar
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,         (22, 24, 27),     category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,       (60, 63, 68),     category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered,(80, 83, 90),     category=dpg.mvThemeCat_Core)
                # Separator / text
                dpg.add_theme_color(dpg.mvThemeCol_Separator,           (65, 70, 75),     category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Text,                (220, 223, 228),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,        (100, 103, 108),  category=dpg.mvThemeCat_Core)
                # Style
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  5,      category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,  6,      category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,   5,      category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,    4,      category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,     8, 6,   category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,    6, 4,   category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,  12, 10,  category=dpg.mvThemeCat_Core)

        dpg.bind_theme(global_theme)

        # ── Per-widget themes ─────────────────────────────────────────────────
        with dpg.theme() as _complete_theme:
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (75, 210, 75),  category=dpg.mvThemeCat_Core)
        self._complete_theme = _complete_theme

        with dpg.theme() as _yet_theme:
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (180, 175, 60), category=dpg.mvThemeCat_Core)
        self._yet_theme = _yet_theme

        with dpg.theme() as _error_theme:
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (210, 60, 60),  category=dpg.mvThemeCat_Core)
        self._error_theme = _error_theme

        with dpg.theme() as _warn_theme:
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 145, 40), category=dpg.mvThemeCat_Core)
        self._warn_theme = _warn_theme

        with dpg.theme() as _info_theme:
            with dpg.theme_component(dpg.mvText):
                dpg.add_theme_color(dpg.mvThemeCol_Text, (150, 155, 165), category=dpg.mvThemeCat_Core)
        self._info_theme = _info_theme

        with dpg.theme() as _train_btn_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (30, 140, 50),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 180, 70),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (20, 110, 38),  category=dpg.mvThemeCat_Core)
        self._train_btn_theme = _train_btn_theme

        with dpg.theme() as _quit_btn_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (130, 38, 38),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (170, 55, 55),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (100, 28, 28),  category=dpg.mvThemeCat_Core)
        self._quit_btn_theme = _quit_btn_theme

        with dpg.theme() as _home_btn_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button,        (38, 70, 130),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (55, 95, 170),  category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (28, 55, 100),  category=dpg.mvThemeCat_Core)
        self._home_btn_theme = _home_btn_theme

        # ── Main window ───────────────────────────────────────────────────────
        with dpg.window(label="YORU - Train", id=imager_window):
            # Step 1
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step 1: Creating project")
                dpg.add_spacer(width=8)
                dpg.add_text(tag="step1_state", default_value="Yet")
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Project Directory")
                dpg.add_input_text(tag="project_path", readonly=True, hint="Path/to/project", width=300)
                dpg.add_button(label="Select Directory", callback=lambda: self.fd_tk.pro_dir_open(), enabled=True)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Load YORU project", tag="load_btn",
                    width=150, height=30, callback=lambda: self.load_pr_dir(), enabled=True,
                )
                dpg.add_text(default_value="  or  ")
                dpg.add_input_text(tag="pro_name", default_value="", width=200, hint="YORU project name")
                dpg.add_button(
                    label="Create YORU project", tag="cre_btn",
                    width=160, height=30, callback=lambda: self.create_pr_dir(), enabled=True,
                )
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Step 2
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step 2: Screenshot GUI")
                dpg.add_spacer(width=8)
                dpg.add_text(tag="step2_state", default_value="Yet")
            dpg.add_button(
                label="Run YORU Frame Capture", tag="grab_btn",
                width=180, height=30, callback=lambda: self.grab_bt(), enabled=True,
            )
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Step 3
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step 3: Labeling")
                dpg.add_spacer(width=8)
                dpg.add_text(tag="step3_state", default_value="Yet")
            dpg.add_button(
                label="Run LabelImg", tag="labelimg_btn",
                width=150, height=30, callback=lambda: self.labelImg_bt(), enabled=True,
            )
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Step 4
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step 4: Prepare images and label files")
                dpg.add_spacer(width=8)
                dpg.add_text(tag="step4_state", default_value="Yet")
            with dpg.group(indent=20):
                dpg.add_button(
                    label="Move Label Images", tag="move_label_images",
                    callback=lambda: self.flie_move_bt(), enabled=False,
                )
                dpg.add_text(
                    "-project_name\n"
                    "     |-all_label_images - classes.txt\n"
                    "     |-train   ~80% of labeled data\n"
                    "     |  |- images - *.png\n"
                    "     |  |- labels - *.txt\n"
                    "     |- val    ~20% of labeled data\n"
                    "         |- images - *.png\n"
                    "         |- labels - *.txt\n"
                )
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Step 5
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step 5: Creating YAML file")
                dpg.add_spacer(width=8)
                dpg.add_text(tag="step5_state", default_value="Yet")
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="classes.txt Path")
                dpg.add_input_text(tag="classes_path", readonly=True, hint="Path/to/classes.txt", width=300)
                dpg.add_button(label="Select Path", callback=lambda: self.fd_tk.class_txt_open(), enabled=True)
            dpg.add_button(
                label="Add class info in YAML file", tag="cre_yaml_btn",
                width=220, height=30, callback=lambda: self.add_class_file(), enabled=True,
            )
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Step 6
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Step 6: Train dataset")
                dpg.add_spacer(width=8)
                dpg.add_text(tag="step6_state", default_value="Yet")
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="YAML Path")
                dpg.add_input_text(tag="yaml_file_path", readonly=True, hint="Path/to/config.yaml", width=300)
                dpg.add_button(label="Select File", callback=lambda: self.fd_tk.dataset_file_open(), enabled=True)
            dpg.add_text(default_value="Training conditions")
            with dpg.group(horizontal=True):
                # Left column: Model settings
                with dpg.group():
                    dpg.add_text("[ Model ]")
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="  Model Family  ")
                        dpg.add_combo(
                            items=self.m_dict["model_family_list"],
                            tag="model_family_combo",
                            default_value=self.m_dict["model_family"],
                            width=150,
                            callback=lambda: self.select_family(),
                        )
                    # YOLO options (shown by default)
                    with dpg.group(tag="yolo_options_group"):
                        with dpg.group(horizontal=True):
                            dpg.add_text(default_value="  YOLO Version  ")
                            dpg.add_combo(
                                items=self.m_dict["yolo_version_list"],
                                tag="yolo_version_combo",
                                default_value=self.m_dict["yolo_version"],
                                width=150,
                                callback=lambda: self.select_version(),
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text(default_value="  Model Size    ")
                            dpg.add_combo(
                                items=self.m_dict["yolo_size_list"],
                                tag="weight_size_combo",
                                default_value=self.m_dict["yolo_size"],
                                width=150,
                                callback=lambda: self.select_size(),
                            )
                    # RT-DETR options (hidden by default)
                    with dpg.group(tag="rtdetr_options_group", show=False):
                        with dpg.group(horizontal=True):
                            dpg.add_text(default_value="  Model Size    ")
                            dpg.add_combo(
                                items=self.m_dict["rtdetr_size_list"],
                                tag="rtdetr_size_combo",
                                default_value=self.m_dict["rtdetr_size"],
                                width=150,
                                callback=lambda: self.select_rtdetr_size(),
                            )
                    # Torchvision options (hidden by default)
                    with dpg.group(tag="tv_options_group", show=False):
                        with dpg.group(horizontal=True):
                            dpg.add_text(default_value="  Backbone      ")
                            dpg.add_combo(
                                items=self.m_dict["tv_backbone_list"],
                                tag="tv_backbone_combo",
                                default_value=self.m_dict["tv_backbone"],
                                width=150,
                                callback=lambda: self.select_backbone(),
                            )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="  Weight File   ")
                        dpg.add_text(tag="weight_display_text", default_value=self.m_dict["weight"])

                dpg.add_spacer(width=30)

                # Right column: Training parameters
                with dpg.group():
                    dpg.add_text("[ Training Params ]")
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="  Epoch         ")
                        dpg.add_input_text(
                            tag="epoc_num_in",
                            default_value=self.m_dict["epoch"],
                            width=120,
                            callback=lambda: self.in_epoch(),
                        )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="  Image Size    ")
                        dpg.add_input_text(
                            tag="img_num_in",
                            default_value=self.m_dict["img"],
                            width=120,
                            callback=lambda: self.in_img(),
                        )
                    with dpg.group(horizontal=True):
                        dpg.add_text(default_value="  Batch         ")
                        dpg.add_input_text(
                            tag="batch_num_in",
                            default_value=self.m_dict["batch"],
                            width=120,
                            callback=lambda: self.in_batch(),
                        )

            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="GPU memory:")
                dpg.add_text(tag="vram_headline", default_value="checking...")
            dpg.add_text(tag="vram_detail", default_value="", wrap=700, indent=12)
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Train Model", tag="str_btn",
                    width=150, height=35, callback=lambda: self.run_yolo(), enabled=True,
                )
                dpg.add_spacer(width=10)
                # Live only while a run is: it asks the trainer to finish the
                # epoch it is on -- and write its checkpoint -- instead of
                # killing it somewhere in the middle of one.
                dpg.add_button(
                    label="Stop after this epoch", tag="stop_btn",
                    width=200, height=35, callback=lambda: self.stop_after_epoch(),
                    enabled=False,
                )
                dpg.add_spacer(width=10)
                # Only offered once a stop is pending, for an epoch too long
                # to sit through.
                dpg.add_button(
                    label="Force stop", tag="force_stop_btn",
                    width=120, height=35, callback=lambda: self.force_stop_prompt(),
                    show=False,
                )
            dpg.add_text(tag="train_stop_text", default_value="", wrap=700)
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text(default_value="Progress: ")
                dpg.add_text(tag="train_progress_text", default_value="---")
                dpg.add_spacer(width=20)
                dpg.add_text(default_value="Remaining: ")
                dpg.add_text(tag="train_eta_text", default_value="---")
            dpg.add_progress_bar(tag="train_progress_bar", default_value=0.0, width=450)
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=6)

            # Bottom buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Back to Home", tag="home_btn",
                    width=120, height=30, callback=lambda: self.home_cb(), enabled=True,
                )
                dpg.add_spacer(width=10)
                dpg.add_button(
                    label="Quit", tag="quit_btn",
                    width=80, height=30, callback=lambda: self.quit_cb(), enabled=True,
                )

        # ── Out-of-memory confirmation ────────────────────────────────────────
        # A modal rather than a passive label: by the time the estimate says
        # the run will not fit, starting it anyway costs the user the minutes
        # it takes CUDA to reach the allocation that fails.
        with dpg.window(
            label="GPU memory warning", tag="vram_modal", modal=True, show=False,
            no_resize=True, no_collapse=True, width=560, height=230, pos=(180, 200),
        ):
            dpg.add_text(tag="vram_modal_text", default_value="", wrap=520)
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Use suggested batch", tag="vram_modal_fix",
                    width=180, height=30, callback=lambda: self.vram_modal_use_batch(),
                )
                dpg.add_button(
                    label="Train anyway", tag="vram_modal_force",
                    width=130, height=30, callback=lambda: self.vram_modal_force(),
                )
                dpg.add_button(
                    label="Cancel", tag="vram_modal_cancel",
                    width=100, height=30, callback=lambda: self.vram_modal_cancel(),
                )

        # ── Force-stop confirmation ───────────────────────────────────────────
        # Killing the trainer costs the epoch it is in the middle of, and with
        # it the validation and best-weights pass that only runs once the epoch
        # loop ends by itself.  The button that does it therefore asks first.
        with dpg.window(
            label="Force stop training", tag="force_stop_modal", modal=True,
            show=False, no_resize=True, no_collapse=True,
            width=520, height=210, pos=(200, 220),
        ):
            dpg.add_text(tag="force_stop_modal_text", default_value="", wrap=480)
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Force stop now", tag="force_stop_confirm",
                    width=170, height=30, callback=lambda: self.force_stop(),
                )
                dpg.add_button(
                    label="Keep training", tag="force_stop_keep",
                    width=150, height=30, callback=lambda: self.force_stop_cancel(),
                )

        # ── Bind per-widget themes ────────────────────────────────────────────
        dpg.bind_item_theme("str_btn",  self._train_btn_theme)
        dpg.bind_item_theme("vram_modal_force", self._quit_btn_theme)
        dpg.bind_item_theme("force_stop_btn", self._quit_btn_theme)
        dpg.bind_item_theme("force_stop_confirm", self._quit_btn_theme)
        dpg.bind_item_theme("train_stop_text", self._warn_theme)
        dpg.bind_item_theme("quit_btn", self._quit_btn_theme)
        dpg.bind_item_theme("home_btn", self._home_btn_theme)
        for _tag in ["step1_state", "step2_state", "step3_state",
                     "step4_state", "step5_state", "step6_state"]:
            dpg.bind_item_theme(_tag, self._yet_theme)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        self._start_gpu_poller()

    def _set_step_state(self, tag: str, state: str) -> None:
        """Update a step state text and apply the matching color theme."""
        dpg.set_value(tag, state)
        if state == "Complete!!":
            dpg.bind_item_theme(tag, self._complete_theme)
        elif state == "Stopped":
            dpg.bind_item_theme(tag, self._warn_theme)
        elif state == "Error":
            dpg.bind_item_theme(tag, self._error_theme)
        else:
            dpg.bind_item_theme(tag, self._yet_theme)

    # ── GPU memory estimate ───────────────────────────────────────────────
    # The driver is polled off the render thread: nvidia-smi takes ~100 ms to
    # answer, which is several dropped frames if it is called from the frame
    # loop.  The loop only reads the value the poller last stored.
    _GPU_POLL_INTERVAL = 2.0

    def _start_gpu_poller(self) -> None:
        """Refresh the cached GPU reading in the background until quit."""

        def _poll():
            while not self._gpu_stop:
                self._gpu_info = get_gpu_info()
                self._gpu_ready = True
                time.sleep(self._GPU_POLL_INTERVAL)

        threading.Thread(target=_poll, daemon=True).start()

    def _invalidate_dataset_stats(self) -> None:
        """Force the next estimate to re-walk the dataset."""
        self._dataset_key = None
        self._vram_key = None

    def _current_vram_inputs(self):
        """Model, image size, batch and dataset YAML as the widgets have them.

        Image Size and Batch are free-text fields, so they are empty or
        half-typed for a moment on every edit.  Those transient values fall
        back to the last ones that parsed, which keeps the estimate steady
        instead of snapping to a default mid-keystroke.
        """
        last_img, last_batch = self._last_valid_nums

        def _int(tag, fallback):
            try:
                value = int(str(dpg.get_value(tag)).strip())
            except (TypeError, ValueError):
                return fallback
            return value if value > 0 else fallback

        imgsz = _int("img_num_in", last_img)
        batch = _int("batch_num_in", last_batch)
        self._last_valid_nums = (imgsz, batch)
        return (
            str(self.m_dict.get("weight", "")),
            imgsz,
            batch,
            str(self.m_dict.get("yaml_path", "")),
        )

    def _current_dataset_stats(self, yaml_path):
        """Dataset stats for *yaml_path*, re-read only when the YAML changes."""
        key = None
        if yaml_path and os.path.exists(yaml_path):
            try:
                key = (yaml_path, os.path.getmtime(yaml_path))
            except OSError:
                key = (yaml_path, 0.0)
        if key != self._dataset_key:
            self._dataset_key = key
            self._dataset_stats = read_dataset_stats(yaml_path) if key else None
        return self._dataset_stats

    def _refresh_vram(self) -> None:
        """Recompute the memory readout when any of its inputs moved."""
        if not self._gpu_ready:
            return
        weight, imgsz, batch, yaml_path = self._current_vram_inputs()
        gpu = self._gpu_info
        key = (weight, imgsz, batch, yaml_path, gpu)
        if key == self._vram_key:
            return
        self._vram_key = key

        stats = self._current_dataset_stats(yaml_path)
        verdict = check_training_vram(weight, imgsz, batch, stats, gpu)
        self._last_verdict = verdict

        dpg.set_value("vram_headline", verdict.headline)
        dpg.set_value("vram_detail", verdict.detail)
        theme = {
            "ok":    self._complete_theme,
            "tight": self._warn_theme,
            "over":  self._error_theme,
        }.get(verdict.level, self._info_theme)
        dpg.bind_item_theme("vram_headline", theme)
        dpg.bind_item_theme("vram_detail", self._info_theme)

    def _show_vram_modal(self, verdict) -> None:
        dpg.set_value(
            "vram_modal_text",
            verdict.headline + "\n\n" + verdict.detail,
        )
        if verdict.suggested_batch:
            dpg.configure_item(
                "vram_modal_fix", show=True,
                label=f"Use Batch {verdict.suggested_batch}",
            )
        else:
            dpg.configure_item("vram_modal_fix", show=False)
        dpg.configure_item("vram_modal", show=True)

    def vram_modal_use_batch(self):
        """Apply the largest batch that is estimated to fit, then train."""
        batch = getattr(self._last_verdict, "suggested_batch", None)
        if batch:
            dpg.set_value("batch_num_in", str(batch))
            self.m_dict["batch"] = str(batch)
            self._refresh_vram()
        dpg.configure_item("vram_modal", show=False)
        self._start_training()

    def vram_modal_force(self):
        dpg.configure_item("vram_modal", show=False)
        self._start_training()

    def vram_modal_cancel(self):
        dpg.configure_item("vram_modal", show=False)

    def _sync_stop_controls(self) -> None:
        """Match the stop controls to what the training subprocess is doing.

        Driven from m_dict by the render loop rather than set directly by
        the monitor thread: DearPyGui is the render thread's to touch, and
        the thread reading the pipe has no business in it.
        """
        active = bool(self.m_dict.get("training_active", False))
        mode = self.m_dict.get("train_stop_mode", "")
        state = (active, mode)
        if state == self._stop_ui_state:
            return
        self._stop_ui_state = state

        dpg.configure_item("str_btn", enabled=not active)
        dpg.configure_item("stop_btn", enabled=active and not mode)
        dpg.configure_item("force_stop_btn", show=active and mode == "graceful")
        if active and mode == "graceful":
            dpg.set_value(
                "train_stop_text",
                "Stopping: the epoch in progress is run to the end and saved first, so this takes until it finishes.",
            )
        elif active:
            dpg.set_value("train_stop_text", "")
        # When the run is over the text is left alone: it holds the outcome
        # message written by _show_training_outcome().

    def _show_training_outcome(self) -> None:
        """Report how the run that just ended finished: done, or stopped."""
        mode = self.m_dict.get("train_stop_mode", "")
        epoch = self.m_dict.get("train_epoch", 0)
        total = self.m_dict.get("train_total_epoch", 0)
        # A stop asked for during the last epoch changes nothing: the run
        # reached its end either way.
        if not mode or (total and epoch >= total):
            self._set_step_state("step6_state", "Complete!!")
            dpg.set_value("train_progress_bar", 1.0)
            dpg.set_value("train_progress_text", "Done!")
            dpg.set_value("train_eta_text", "0s")
            dpg.set_value("train_stop_text", "")
            return

        self._set_step_state("step6_state", "Stopped")
        dpg.set_value("train_progress_text", f"Stopped at epoch {epoch} / {total}")
        dpg.set_value("train_eta_text", "---")
        if mode == "force":
            dpg.set_value(
                "train_stop_text",
                "Force-stopped. The epoch that was running is lost; the checkpoints written before it are in the run folder.",
            )
        else:
            dpg.set_value(
                "train_stop_text",
                f"Stopped after epoch {epoch}. Its weights are saved in the run folder of the project directory.",
            )

    def plot_callback(self) -> None:
        self._refresh_vram()
        self._sync_stop_controls()
        epoch = self.m_dict.get("train_epoch", 0)
        total = self.m_dict.get("train_total_epoch", 0)
        # Only while a run is going: DearPyGui keeps the last value it was
        # given, so re-deriving the line from the epoch counters afterwards
        # would do nothing but overwrite the outcome with "Epoch 12 / 300".
        if total > 0 and self.m_dict.get("training_active", False):
            progress = epoch / total
            dpg.set_value("train_progress_bar", progress)
            dpg.set_value("train_progress_text", f"Epoch {epoch} / {total}")
            eta_sec = self.m_dict.get("train_eta_seconds", None)
            if eta_sec is not None and epoch > 0:
                h = int(eta_sec) // 3600
                m = (int(eta_sec) % 3600) // 60
                s = int(eta_sec) % 60
                if h > 0:
                    dpg.set_value("train_eta_text", f"{h}h {m:02d}m {s:02d}s")
                elif m > 0:
                    dpg.set_value("train_eta_text", f"{m}m {s:02d}s")
                else:
                    dpg.set_value("train_eta_text", f"{s}s")
        if self.m_dict.get("training_done", False):
            self.m_dict["training_done"] = False
            self._show_training_outcome()

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
            for sub in (
                "",
                "/train",
                "/train/images",
                "/train/labels",
                "/val",
                "/val/images",
                "/val/labels",
                "/all_label_images",
            ):
                os.makedirs(self.m_dict["project_dir"] + sub, exist_ok=True)
            self.m_dict["yaml_path"] = self.m_dict["project_dir"] + "/config.yaml"
            self.m_dict["all_label_dir"] = (
                self.m_dict["project_dir"] + "/all_label_images"
            )
            cr_project = create_project(self.m_dict)
            cr_project.create_yaml()

            dpg.set_value("yaml_file_path", self.m_dict["yaml_path"])
            dpg.enable_item("move_label_images")
            self._set_step_state("step1_state", "Complete!!")
        else:
            print("The project already exists.")
            dpg.enable_item("move_label_images")

    def _restore_model_ui(self, weight: str) -> None:
        """ウェイトファイル名からモデルファミリー/バージョン/サイズのUIを復元する。"""
        w = weight.lower()
        if w.startswith("yolov5"):
            # v1 projects: YOLOv5 training is no longer bundled, so restore the
            # same size on YOLO11 instead of showing an unusable selection.
            size = w[6] if len(w) > 6 else "s"
            print(
                f"[yoru] This project was trained with '{weight}'. YOLOv5 training "
                f"is no longer bundled; switching to yolo11{size}.pt."
            )
            family, version = "YOLO", "YOLO11"
        elif w.startswith("yolov8"):
            family, version, size = "YOLO", "YOLOv8", w[6] if len(w) > 6 else "s"
        elif w.startswith("yolo11"):
            family, version, size = "YOLO", "YOLO11", w[6] if len(w) > 6 else "s"
        elif w.startswith("rtdetr-"):
            family, version, size = "RT-DETR", None, w[7] if len(w) > 7 else "l"
        elif w.startswith("fasterrcnn"):
            family, version, size = "Faster R-CNN", None, None
        elif w.startswith("maskrcnn"):
            family, version, size = "Mask R-CNN", None, None
        elif w.startswith("ssd"):
            family, version, size = "SSD", None, None
        else:
            return

        self.m_dict["model_family"] = family
        dpg.set_value("model_family_combo", family)
        dpg.configure_item("yolo_options_group",   show=(family == "YOLO"))
        dpg.configure_item("rtdetr_options_group", show=(family == "RT-DETR"))
        dpg.configure_item("tv_options_group",
                           show=(family in ("Faster R-CNN", "Mask R-CNN", "SSD")))

        if family == "YOLO" and version:
            self.m_dict["yolo_version"] = version
            dpg.set_value("yolo_version_combo", version)
            if size and size in self.m_dict.get("yolo_size_list", []):
                self.m_dict["yolo_size"] = size
                dpg.set_value("weight_size_combo", size)
        elif family == "RT-DETR" and size:
            if size in self.m_dict.get("rtdetr_size_list", []):
                self.m_dict["rtdetr_size"] = size
                dpg.set_value("rtdetr_size_combo", size)

        # Keep the weight in step with the restored combos; this is what remaps
        # a legacy yolov5 selection onto a weight that can actually be trained.
        self.m_dict["weight"] = self._build_weight()
        dpg.set_value("weight_display_text", self.m_dict["weight"])

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
        self.m_dict["all_label_dir"] = self.m_dict["project_dir"] + "/all_label_images"

        # --- Step1: プロジェクト読み込み完了 ---
        dpg.set_value("yaml_file_path", self.m_dict["yaml_path"])
        dpg.enable_item("move_label_images")
        self._set_step_state("step1_state", "Complete!!")

        # --- モデル/ウェイトの復元 (training_date があれば weights キーを優先) ---
        saved_model = data.get("weights") or data.get("Model") or data.get("YOLO_ver")
        if saved_model:
            if not saved_model.endswith(".pt"):
                prefix_map = {"yolov5": "yolov5s", "yolov8": "yolov8s", "yolo11": "yolo11s"}
                saved_model = prefix_map.get(saved_model, saved_model) + ".pt"
            self.m_dict["weight"] = saved_model
            dpg.set_value("weight_display_text", saved_model)
            self._restore_model_ui(saved_model)

        # --- 学習条件の復元 ---
        if data.get("epochs") is not None:
            self.m_dict["epoch"] = data["epochs"]
            dpg.set_value("epoc_num_in", str(data["epochs"]))
        if data.get("image_size") is not None:
            self.m_dict["img"] = data["image_size"]
            dpg.set_value("img_num_in", str(data["image_size"]))
        if data.get("batch-size") is not None:
            self.m_dict["batch"] = data["batch-size"]
            dpg.set_value("batch_num_in", str(data["batch-size"]))

        # --- Step2: all_label_images に画像があれば完了扱い ---
        all_label_dir = self.m_dict["all_label_dir"]
        if os.path.exists(all_label_dir):
            img_exts = {".png", ".jpg", ".jpeg", ".bmp"}
            has_images = any(
                os.path.splitext(f)[1].lower() in img_exts
                for f in os.listdir(all_label_dir)
            )
            if has_images:
                self._set_step_state("step2_state", "Complete!!")

            # --- Step3: all_label_images に classes.txt 以外の .txt があればラベリング完了 ---
            has_labels = any(
                f.endswith(".txt") and f != "classes.txt"
                for f in os.listdir(all_label_dir)
            )
            if has_labels:
                self._set_step_state("step3_state", "Complete!!")

        # --- Step4: train/images にファイルがあれば移動済み ---
        train_images_dir = self.m_dict["project_dir"] + "/train/images"
        if os.path.exists(train_images_dir) and os.listdir(train_images_dir):
            self._set_step_state("step4_state", "Complete!!")
            dpg.disable_item("move_label_images")

        # --- Step5: クラス情報が登録済みなら classes_path を復元して完了表示 ---
        if data.get("add_class_info_date"):
            classes_txt = all_label_dir + "/classes.txt"
            if os.path.exists(classes_txt):
                self.m_dict["classes_path"] = classes_txt
                dpg.set_value("classes_path", classes_txt)
            self._set_step_state("step5_state", "Complete!!")

        # --- Step6: 学習済みなら完了表示 ---
        if data.get("training_date"):
            self._set_step_state("step6_state", "Complete!!")

        self._invalidate_dataset_stats()
        print("load complete")

    def grab_bt(self):
        # print("quit_pushed")
        # self.m_dict["quit"] = True
        subprocess.Popen([sys.executable, "-m", "yoru.grab_GUI"])
        self._set_step_state("step2_state", "Complete!!")
        # dpg.destroy_context()  # <-- moved from __del__

    def labelImg_bt(self):
        # Popen, not call: subprocess.call would freeze the training GUI for as
        # long as labelImg stays open.
        try:
            subprocess.Popen(["labelImg"])
        except OSError as e:
            print(f"Failed to launch labelImg: {e}")
            self._set_step_state("step3_state", "Error")
            return
        self._set_step_state("step3_state", "Complete!!")

    def quit_cb(self):
        print("quit_pushed")
        self._gpu_stop = True
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def home_cb(self):
        print("Back home")
        self._gpu_stop = True
        self.m_dict["back_to_home"] = True
        self.m_dict["quit"] = True
        dpg.destroy_context()  # <-- moved from __del__

    def add_class_file(self):
        classes_path = self.m_dict.get("classes_path", "")
        if "classes.txt" not in classes_path or not os.path.exists(classes_path):
            print(f"Select a valid classes.txt first (got {classes_path!r})")
            self._set_step_state("step5_state", "Error")
            return
        try:
            create_project(self.m_dict).add_class_info()
        except Exception as e:
            print(f"Failed to add class info: {e}")
            self._set_step_state("step5_state", "Error")
            return
        self._set_step_state("step5_state", "Complete!!")

    def _build_weight(self) -> str:
        """選択中のモデルファミリー・バージョン・サイズからウェイトファイル名を生成する。"""
        family = self.m_dict.get("model_family", "YOLO")
        if family == "YOLO":
            prefix_map = {"YOLOv8": "yolov8", "YOLO11": "yolo11"}
            prefix = prefix_map.get(self.m_dict.get("yolo_version", "YOLO11"), "yolo11")
            size = self.m_dict.get("yolo_size", "s")
            return f"{prefix}{size}.pt"
        elif family == "RT-DETR":
            size = self.m_dict.get("rtdetr_size", "l")
            return f"rtdetr-{size}.pt"
        elif family == "Faster R-CNN":
            return "fasterrcnn_resnet50_best.pt"
        elif family == "Mask R-CNN":
            return "maskrcnn_resnet50_best.pt"
        elif family == "SSD":
            return "ssd_vgg16_best.pt"
        return "yolo11s.pt"

    def select_family(self):
        family = dpg.get_value("model_family_combo")
        self.m_dict["model_family"] = family

        # Show/hide family-specific option groups
        dpg.configure_item("yolo_options_group",   show=(family == "YOLO"))
        dpg.configure_item("rtdetr_options_group", show=(family == "RT-DETR"))
        dpg.configure_item("tv_options_group",     show=(family in ("Faster R-CNN", "Mask R-CNN", "SSD")))

        # Update backbone list for torchvision models
        from yoru.libs.init_train import MODEL_FAMILY_CONFIG
        if family in MODEL_FAMILY_CONFIG and "backbones" in MODEL_FAMILY_CONFIG[family]:
            backbones = MODEL_FAMILY_CONFIG[family]["backbones"]
            dpg.configure_item("tv_backbone_combo", items=backbones)
            self.m_dict["tv_backbone"] = backbones[0]
            dpg.set_value("tv_backbone_combo", backbones[0])

        self.m_dict["weight"] = self._build_weight()
        dpg.set_value("weight_display_text", self.m_dict["weight"])

    def select_version(self):
        self.m_dict["yolo_version"] = dpg.get_value("yolo_version_combo")
        self.m_dict["weight"] = self._build_weight()
        dpg.set_value("weight_display_text", self.m_dict["weight"])

    def select_size(self):
        self.m_dict["yolo_size"] = dpg.get_value("weight_size_combo")
        self.m_dict["weight"] = self._build_weight()
        dpg.set_value("weight_display_text", self.m_dict["weight"])

    def select_rtdetr_size(self):
        self.m_dict["rtdetr_size"] = dpg.get_value("rtdetr_size_combo")
        self.m_dict["weight"] = self._build_weight()
        dpg.set_value("weight_display_text", self.m_dict["weight"])

    def select_backbone(self):
        self.m_dict["tv_backbone"] = dpg.get_value("tv_backbone_combo")
        self.m_dict["weight"] = self._build_weight()
        dpg.set_value("weight_display_text", self.m_dict["weight"])

    def in_epoch(self):
        tf = dpg.get_value("epoc_num_in")
        self.m_dict["epoch"] = tf

    def flie_move_bt(self):
        self.fmrd = file_move_random(self.m_dict)
        print(self.m_dict["all_label_dir"])
        self.fmrd.move()
        self._invalidate_dataset_stats()
        dpg.disable_item("move_label_images")
        self._set_step_state("step4_state", "Complete!!")

    def in_img(self):
        tf = dpg.get_value("img_num_in")
        self.m_dict["img"] = tf

    def in_batch(self):
        tf = dpg.get_value("batch_num_in")
        self.m_dict["batch"] = tf

    def stop_after_epoch(self):
        """Ask the trainer to end the run once this epoch is finished."""
        if self._stop_file is None:
            print("[yoru] no training run to stop")
            return
        request_stop(self._stop_file)
        self.m_dict["train_stop_mode"] = "graceful"
        print("[yoru] stop requested: training ends after the current epoch")

    def force_stop_prompt(self):
        """Ask before killing the trainer: the running epoch is lost."""
        epoch = self.m_dict.get("train_epoch", 0)
        total = self.m_dict.get("train_total_epoch", 0)
        dpg.set_value(
            "force_stop_modal_text",
            f"Training is in epoch {epoch} of {total}.\n\n"
            "Forcing a stop kills the training process now. The epoch it is "
            "running is lost, and so is the final validation pass that picks "
            "the best weights. Checkpoints from earlier epochs stay on disk.\n\n"
            "Waiting for the epoch to end keeps all of them.",
        )
        dpg.configure_item("force_stop_modal", show=True)

    def force_stop_cancel(self):
        dpg.configure_item("force_stop_modal", show=False)

    def force_stop(self):
        """Kill the training subprocess, epoch in progress and all."""
        dpg.configure_item("force_stop_modal", show=False)
        proc = self._train_proc
        if proc is None or proc.poll() is not None:
            return
        self.m_dict["train_stop_mode"] = "force"
        print("[yoru] force-stopping the training process")
        # Off the render thread: killing the tree waits on the processes in it.
        threading.Thread(
            target=terminate_process_tree, args=(proc,), daemon=True,
        ).start()

    def _monitor_training(self, proc, total_epochs: int) -> None:
        """Read subprocess stdout line by line, parse epoch progress, update m_dict."""
        # Matches: "Epoch [1/50]" (torchvision) or "      1/100 " (ultralytics/yolov5)
        torchvision_re = re.compile(r"Epoch\s*\[\s*(\d+)/(\d+)\s*\]")
        ultralytics_re = re.compile(r"^\s+(\d+)/(\d+)\s")

        # Ultralytics redraws its progress bar with a carriage return, which
        # the pipe translates into a newline: echo through ProgressPrinter so
        # the console keeps one line per epoch instead of one per batch.
        printer = ProgressPrinter()

        self.m_dict["train_epoch"] = 0
        self.m_dict["train_total_epoch"] = total_epochs
        self.m_dict["train_eta_seconds"] = None

        start_time = None
        start_epoch = 0

        try:
            for raw_line in proc.stdout:
                line = printer.clean(raw_line)
                printer.write(line)  # pass-through to console, one row per epoch
                m = torchvision_re.search(line) or ultralytics_re.match(line)
                if m:
                    current = int(m.group(1))
                    total   = int(m.group(2))
                    if start_time is None:
                        start_time = time.monotonic()
                        start_epoch = current - 1
                    self.m_dict["train_epoch"]       = current
                    self.m_dict["train_total_epoch"] = total
                    elapsed = time.monotonic() - start_time
                    completed = current - start_epoch
                    remaining = total - current
                    if completed > 0:
                        sec_per_epoch = elapsed / completed
                        self.m_dict["train_eta_seconds"] = sec_per_epoch * remaining
        finally:
            # Whatever happened to the pipe, the GUI has to be told the run is
            # over: the Train button stays disabled until it is.
            printer.close()
            proc.wait()
            if not self.m_dict.get("train_stop_mode", ""):
                # Ran to its last epoch: fill the bar even if the closing line
                # was never parsed.
                self.m_dict["train_epoch"] = self.m_dict.get(
                    "train_total_epoch", total_epochs
                )
            self.m_dict["train_eta_seconds"] = None
            # A request the trainer never got to (a forced kill beat it there)
            # would otherwise stop the next run after one epoch.
            clear_stop(self._stop_file)
            self.m_dict["training_active"] = False
            # Last, once the rest of the state is settled: this is the flag the
            # render loop watches.
            self.m_dict["training_done"] = True

    def run_yolo(self):
        """Confirm the run fits in VRAM, then hand it to the backend.

        A passive warning label is not enough here: an over-budget run does
        not fail at once, it fails after CUDA has worked its way to the
        allocation that does not fit, which can be several minutes in.
        """
        verdict = self._last_verdict
        if verdict is not None and verdict.level == "over":
            self._show_vram_modal(verdict)
            return
        self._start_training()

    def _start_training(self):
        """Dispatch training to the correct backend plugin."""
        from yoru.libs.plugins import detect_trainer_backend, get_trainer

        backend = detect_trainer_backend(self.m_dict)
        trainer = get_trainer(backend)

        # One stop file per project, cleared before every run: a request
        # nobody consumed must not end this run after a single epoch.
        self._stop_file = stop_file_for(self.m_dict["project_dir"])
        clear_stop(self._stop_file)

        config = {
            "img_size": int(self.m_dict["img"]),
            "batch_size": int(self.m_dict["batch"]),
            "epochs": int(self.m_dict["epoch"]),
            "data_yaml": str(self.m_dict["yaml_path"]),
            "weights": str(self.m_dict["weight"]),
            "project_dir": str(self.m_dict["project_dir"]),
            "model_family": self.m_dict.get("model_family", "YOLO"),
            "stop_file": str(self._stop_file),
        }

        cr_project = create_project(self.m_dict)
        cr_project.add_training_info()
        print("added the information in yaml file")

        try:
            proc = trainer.train(config)
            print(f"start training ({backend})")
            total = int(self.m_dict.get("epoch", 300))
            self._train_proc = proc
            self.m_dict["train_stop_mode"] = ""
            self.m_dict["training_active"] = True
            t = threading.Thread(
                target=self._monitor_training, args=(proc, total), daemon=True,
            )
            t.start()
        except Exception as e:
            print("error: ", e)
            self._train_proc = None
            self.m_dict["training_active"] = False
            self._set_step_state("step6_state", "Error")

    def __del__(self):
        pass


def main():
    print("Starting training GUI")
    with Manager() as manager:
        d = manager.dict()

        # initialize m_dict with init_analysis
        init = init_train(m_dict=d)

        gui = yoru_train(m_dict=d)
        process_pool = []
        prc_gui = Process(target=gui.run)
        process_pool.append(prc_gui)  # <-- this line was added
        prc_gui.start()
        prc_gui.join()


if __name__ == "__main__":
    main()
