# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import json
import subprocess
import sys
import threading
from pathlib import Path

import webview

DEFAULT_CONFIG_PATH = Path("./config/yoru_default.yaml")
LOG_DIR = Path("./logs")
CONFIG_LOG_PATH = LOG_DIR / "condition_file_log.json"
WEB_DIR = Path(__file__).resolve().parent.parent / "web"


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

def _run_gui_subprocess(command, gui_name):
    """GUIサブプロセスを実行し、エラーがあればEel経由でフロントに通知する"""
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        error_msg = stderr.strip() if stderr.strip() else stdout.strip()
        eel.displayError(gui_name, error_msg)


def _launch_gui(command, gui_name):
    """バックグラウンドスレッドでGUIサブプロセスを起動する"""
    thread = threading.Thread(
        target=_run_gui_subprocess, args=(command, gui_name), daemon=True
    )
    thread.start()


def create_default_json():
    log_dir = "./logs"
    log_file_path = f"{log_dir}/condition_file_log.json"
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
        with open(log_file_path, "r") as file:
            data = json.load(file)
            if "config_file" in data:
                condition_file_path = data["config_file"]
            else:
                condition_file_path = default_condition_file_path
    except (FileNotFoundError, json.JSONDecodeError):
        condition_file_path = default_condition_file_path
        create_default_json()  # デフォルトのJSONファイルを作成


@eel.expose
def run_cam_gui_YMH():
    global condition_file_path
    if not os.path.isfile(condition_file_path):
        eel.displayError("Real-time GUI", "Config file not found: " + condition_file_path)
        return
    _launch_gui(
        [
            sys.executable,
            "-c",
            f"from yoru import realtime_yoru_GUI; realtime_yoru_GUI.main(r'{condition_file_path}')",
        ],
        "Real-time GUI",
    )


@eel.expose
def show_file_dialog():
    global condition_file_path
    root = Tk()
    root.withdraw()  # Tkのルートウィンドウを表示しない
    tk_file = filedialog.askopenfilename(
        title="Select Condition file",
        filetypes=[("Condition yaml file", ".yml .yaml")],  # ファイルフィルタ
    )  # ファイル選択ダイアログを表示
    is_file = os.path.isfile(tk_file)
    if is_file:
        condition_file_path = path_to_ab(tk_file)
        update_json_config_file(condition_file_path)  # JSONファイルを更新
    else:
        condition_file_path = path_to_ab(default_condition_file_path)
    eel.displayFilePath(condition_file_path)  # JavaScript関数にファイルパスを送る
    print(condition_file_path)


def update_json_config_file(new_path):
    data = {"config_file": new_path}
    with open("./logs/condition_file_log.json", "w") as file:
        json.dump(data, file)


def path_to_ab(rel_path):
    p_rel = Path(rel_path)
    p_abu = p_rel.resolve()
    return str(p_abu)


@eel.expose
def print_file_path():
    global condition_file_path
    p_rel = Path(condition_file_path)
    p_abu = p_rel.resolve()
    print(p_abu)
    return str(p_abu)


@eel.expose
def run_analysis_gui():
    _launch_gui(
        [sys.executable, "-c", "from yoru import analysis_GUI; analysis_GUI.main()"],
        "Analysis GUI",
    )


@eel.expose
def run_train_gui():
    _launch_gui(
        [sys.executable, "-c", "from yoru import train_GUI; train_GUI.main()"],
        "Train GUI",
    )
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

@eel.expose
def run_evaluate_gui():
    _launch_gui(
        [sys.executable, "-c", "from yoru import evaluation_GUI; evaluation_GUI.main()"],
        "Evaluate GUI",
    )
    def run_analysis_gui(self):
        _launch_module("yoru.analysis_GUI")

    def run_evaluate_gui(self):
        _launch_module("yoru.evaluation_GUI")

    def run_config_creator_gui(self):
        _launch_module("yoru.config_creator_GUI")


def main():
    state = AppState()
    state.load_condition_file()
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
