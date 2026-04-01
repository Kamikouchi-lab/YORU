# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import json
import subprocess
import sys
from pathlib import Path
from tkinter import Tk, filedialog

import eel

DEFAULT_CONFIG_PATH = Path("./config/yoru_default.yaml")
LOG_DIR = Path("./logs")
CONFIG_LOG_PATH = LOG_DIR / "condition_file_log.json"


class AppState:
    """Holds application state instead of module-level globals."""

    def __init__(self):
        self.condition_file_path: Path = DEFAULT_CONFIG_PATH

    def load_condition_file(self):
        """Load the last-used config file path from the log, or create defaults."""
        try:
            data = json.loads(CONFIG_LOG_PATH.read_text(encoding="utf-8"))
            path = Path(data.get("config_file", str(DEFAULT_CONFIG_PATH)))
            self.condition_file_path = path.resolve()
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            self.condition_file_path = DEFAULT_CONFIG_PATH.resolve()
            self._write_config_log(self.condition_file_path)

    def update_condition_file(self, new_path: Path):
        self.condition_file_path = new_path.resolve()
        self._write_config_log(self.condition_file_path)

    @staticmethod
    def _write_config_log(path: Path):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_LOG_PATH.write_text(
            json.dumps({"config_file": str(path)}), encoding="utf-8"
        )


_state = AppState()


def _launch_module(module_name: str, *extra_args: str):
    """Launch a YORU sub-module in a new console window."""
    cmd = [sys.executable, "-m", module_name, *extra_args]
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    try:
        subprocess.Popen(cmd, **kwargs)
    except OSError as e:
        print(f"Failed to launch {module_name}: {e}", file=sys.stderr)
        eel.displayError(f"Failed to launch {module_name}: {e}")


@eel.expose
def run_realtime_gui():
    cfg = _state.condition_file_path
    if cfg.is_file():
        _launch_module("yoru.realtime_yoru_GUI", str(cfg))
    else:
        msg = f"Config file not found: {cfg}"
        print(msg, file=sys.stderr)
        eel.displayError(msg)


@eel.expose
def show_file_dialog():
    root = Tk()
    root.withdraw()
    try:
        tk_file = filedialog.askopenfilename(
            title="Select Condition file",
            filetypes=[("Condition yaml file", ".yml .yaml")],
        )
    finally:
        root.destroy()

    if tk_file and Path(tk_file).is_file():
        _state.update_condition_file(Path(tk_file))
    else:
        _state.update_condition_file(DEFAULT_CONFIG_PATH)

    resolved = str(_state.condition_file_path)
    eel.displayFilePath(resolved)


@eel.expose
def get_config_path():
    return str(_state.condition_file_path.resolve())


@eel.expose
def run_analysis_gui():
    _launch_module("yoru.analysis_GUI")


@eel.expose
def run_train_gui():
    _launch_module("yoru.train_GUI")


@eel.expose
def run_evaluate_gui():
    _launch_module("yoru.evaluation_GUI")


@eel.expose
def run_config_creator_gui():
    _launch_module("yoru.config_creator_GUI")


def main():
    _state.load_condition_file()
    eel.init("web")
    eel.start("gui_home.html", size=(1024, 768), port=8889)


if __name__ == "__main__":
    main()
