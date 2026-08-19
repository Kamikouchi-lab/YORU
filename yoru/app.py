# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import webview

from yoru.libs.paths import web_dir

DEFAULT_CONFIG_PATH = Path("./config/yoru_default.yaml")
LOG_DIR = Path("./logs")
CONFIG_LOG_PATH = LOG_DIR / "condition_file_log.json"
WEB_DIR = web_dir()


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


class Api:
    """Python API exposed to the frontend via pywebview."""

    def __init__(self, state: AppState):
        self._state = state
        self._window: webview.Window | None = None

    def set_window(self, window: webview.Window):
        self._window = window

    def run_realtime_gui(self):
        cfg = self._state.condition_file_path
        if cfg.is_file():
            _launch_module("yoru.realtime_yoru_GUI", str(cfg))
        else:
            msg = f"Config file not found: {cfg}"
            print(msg, file=sys.stderr)
            if self._window:
                self._window.evaluate_js(f"displayError({json.dumps(msg)})")

    def show_file_dialog(self):
        if not self._window:
            return
        result = self._window.create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("YAML files (*.yml;*.yaml)",),
        )
        if result and len(result) > 0 and Path(result[0]).is_file():
            self._state.update_condition_file(Path(result[0]))
        else:
            self._state.update_condition_file(DEFAULT_CONFIG_PATH)

        resolved = str(self._state.condition_file_path)
        self._window.evaluate_js(
            f"document.getElementById('file-path').innerText = {json.dumps(resolved)}"
        )

    def get_config_path(self):
        return str(self._state.condition_file_path.resolve())

    def run_train_gui(self):
        _launch_module("yoru.train_GUI")

    def run_analysis_gui(self):
        _launch_module("yoru.analysis_GUI")

    def run_evaluate_gui(self):
        _launch_module("yoru.evaluation_GUI")

    def run_config_creator_gui(self):
        _launch_module("yoru.config_creator_GUI")


def main(config: str | None = None):
    """Open the launcher window.

    *config* overrides the remembered condition file (``yoru gui --config X``);
    when omitted the last-used path from the config log is kept.
    """
    state = AppState()
    state.load_condition_file()
    if config:
        path = Path(config)
        if path.is_file():
            state.update_condition_file(path)
        else:
            print(f"[yoru] Config file not found: {config}", file=sys.stderr)
    api = Api(state)

    window = webview.create_window(
        "YORU (Your Optimal Recognition Utility)",
        url=str(WEB_DIR / "gui_home.html"),
        js_api=api,
        width=1000,
        height=800,
    )
    api.set_window(window)
    webview.start()


if __name__ == "__main__":
    main()
