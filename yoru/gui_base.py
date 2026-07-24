# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Shared utilities for DearPyGui-based GUI modules.

Provides:
- ``process_frame``: resize and centre a frame in a square canvas.
- ``frame_to_data_rgb``: convert BGR frame → float32 RGB array for DPG raw textures.
- ``frame_to_data_rgba``: convert BGR frame → float32 RGBA array for DPG dynamic textures.
- ``apply_default_theme``: apply the standard YORU dark-blue theme.
"""

import cv2
import dearpygui.dearpygui as dpg
import numpy as np


def process_frame(frame, preview_size, *, v_flip=False, h_flip=False):
    """Resize *frame* to fit inside a ``preview_size × preview_size`` black canvas.

    Returns the centred canvas (``np.ndarray``).
    """
    h, w = frame.shape[:2]

    if v_flip:
        frame = cv2.flip(frame, 0)
    if h_flip:
        frame = cv2.flip(frame, 1)

    if w >= h:
        new_w = preview_size
        new_h = int(h * (preview_size / w))
    else:
        new_w = int(w * (preview_size / h))
        new_h = preview_size

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((preview_size, preview_size, 3), np.uint8)
    half = preview_size // 2
    canvas[
        half - new_h // 2 : half - new_h // 2 + new_h,
        half - new_w // 2 : half - new_w // 2 + new_w,
        :,
    ] = resized
    return canvas


def frame_to_data_rgb(frame):
    """Convert a BGR ``np.ndarray`` frame to a flat float32 RGB array for DPG raw textures."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.true_divide(rgb.ravel(), 255.0).astype("f")


def frame_to_data_rgba(frame):
    """Convert a BGR ``np.ndarray`` frame to a float32 RGBA array for DPG dynamic textures."""
    return np.true_divide(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA), 255)


def apply_default_theme():
    """Create and bind the standard YORU dark-blue DPG theme."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            # Backgrounds
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (18, 24, 42), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (22, 30, 52), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, (22, 30, 52), category=dpg.mvThemeCat_Core)
            # Title bar
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (25, 70, 130), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (35, 95, 165), category=dpg.mvThemeCat_Core)
            # Tabs
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (25, 70, 130), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, (50, 115, 185), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, (45, 110, 180), category=dpg.mvThemeCat_Core)
            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, (35, 95, 165), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (55, 125, 200), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (20, 70, 140), category=dpg.mvThemeCat_Core)
            # Frame (inputs, combos, checkboxes)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (30, 42, 68), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (38, 55, 88), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (45, 65, 100), category=dpg.mvThemeCat_Core)
            # Slider
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (60, 130, 210), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (85, 160, 235), category=dpg.mvThemeCat_Core)
            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, (18, 24, 42), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, (45, 80, 140), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (60, 100, 165), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, (75, 120, 185), category=dpg.mvThemeCat_Core)
            # Separator & check
            dpg.add_theme_color(dpg.mvThemeCol_Separator, (50, 85, 140), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (80, 180, 240), category=dpg.mvThemeCat_Core)
            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 230, 230), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (120, 140, 170), category=dpg.mvThemeCat_Core)
            # Header / collapsible
            dpg.add_theme_color(dpg.mvThemeCol_Header, (35, 80, 145), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (50, 100, 170), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (25, 65, 125), category=dpg.mvThemeCat_Core)
            # Plot (progress bar fill)
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (35, 120, 200), category=dpg.mvThemeCat_Core)
            # Style vars
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 10, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 12, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 5, category=dpg.mvThemeCat_Core)
    dpg.bind_theme(theme)
    return theme
