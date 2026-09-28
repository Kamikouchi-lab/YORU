"""Opt-in native-window checks, isolated from user layouts and hardware."""

import os
import subprocess
import sys

import pytest

from tests.test_gui_quit import GUIS


NATIVE_CHECK = r'''
import importlib
import pathlib
import sys
import dearpygui.dearpygui as dpg
from yoru import gui_layout

module_name, class_name, setup_name, state_dir = sys.argv[1:]
gui_layout.layout_dir = lambda: pathlib.Path(state_dir)
original_show = dpg.show_viewport
def show_offscreen(*args, **kwargs):
    dpg.set_viewport_pos([-20000, -20000])
    original_show(*args, **kwargs)
dpg.show_viewport = show_offscreen
state = {"quit": False}
initializers = {
    "analysis_GUI": ("init_analysis", "init_analysis"),
    "train_GUI": ("init_train", "init_train"),
    "evaluation_GUI": ("init_evaluation", "init_evaluater"),
    "create_labels_GUI": ("init_create_label", "init_create_label"),
}
if module_name in initializers:
    name, function = initializers[module_name]
    getattr(importlib.import_module("yoru.libs." + name), function)(m_dict=state)
cls = getattr(importlib.import_module("yoru." + module_name), class_name)
if module_name == "config_creator_GUI":
    gui = cls()
elif module_name == "realtime_yoru_GUI":
    from yoru.libs.init_realtime import init_asovi
    config = "config/yoru_default.yaml"
    init_asovi(config_file=config, m_dict=state)
    gui = cls(config_file=config, m_dict=state)
else:
    gui = cls(m_dict=state)
# This test opens the real controls but never starts hardware or model work.
if module_name == "train_GUI":
    gui._start_gpu_poller = lambda: None
setup = getattr(gui, setup_name)
clicked = []
def build_and_schedule_quit():
    setup()
    buttons = [item for item in dpg.get_all_items()
               if dpg.get_item_configuration(item).get("label") == "Quit"]
    assert buttons, "No Quit button found"
    callback = dpg.get_item_callback(buttons[0])
    def click():
        callback()
        clicked.append(True)
    dpg.set_frame_callback(dpg.get_frame_count() + 3, click)
setattr(gui, setup_name, build_and_schedule_quit)
gui.run()
assert clicked and gui.m_dict["quit"]
print("NATIVE_QUIT_OK")
'''


@pytest.mark.gui
@pytest.mark.parametrize("module_name,class_name,setup", GUIS)
def test_native_quit_button_exits_process(repo_root, tmp_path, module_name, class_name, setup):
    env = dict(os.environ, YORU_HOME=str(tmp_path / "home"))
    result = subprocess.run(
        [sys.executable, "-c", NATIVE_CHECK, module_name, class_name, setup, str(tmp_path)],
        cwd=repo_root, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_QUIT_OK" in result.stdout
