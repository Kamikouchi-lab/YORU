import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import Tk, filedialog

import eel

from yoru.libs.user_paths import get_state_file, log_message, setup_logging

# if __name__ == "__main__":


sys.path.append("../yoru")

# try:

# except(ModuleNotFoundError):
#     import analysis_GUI
#     import realtime_yoru_GUI
#     import evaluation_GUI
#     import train_GUI


default_condition_file_path = "./config/yoru_default.yaml"
condition_file_path = default_condition_file_path


def _run_gui_subprocess(command, gui_name):
    """Run the GUI subprocess and, if an error occurs, notify the frontend via Eel."""
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        error_msg = stderr.strip() if stderr.strip() else stdout.strip()
        log_message(
            f"GUI subprocess '{gui_name}' exited with code {proc.returncode}: {error_msg}",
            level=logging.ERROR,
        )
        eel.displayError(gui_name, error_msg)


def _launch_gui(command, gui_name):
    """Launch the GUI subprocess in a background thread."""
    thread = threading.Thread(
        target=_run_gui_subprocess, args=(command, gui_name), daemon=True
    )
    thread.start()


def create_default_json():
    default_data = {"config_file": default_condition_file_path}
    with open(get_state_file(), "w") as file:
        json.dump(default_data, file)


def _migrate_legacy_state_file(state_file):
    """One-time migration of the pre-``~/.yoru`` state file.

    Older versions stored the last-used config in
    ``./logs/condition_file_log.json`` (relative to the working directory). If
    that legacy file exists and the new one does not yet, copy it over so the
    user's last selection is preserved. The legacy file is left in place.
    """
    legacy = Path("./logs/condition_file_log.json")
    try:
        if legacy.is_file() and not state_file.exists():
            with open(legacy, "r") as f:
                data = json.load(f)
            with open(state_file, "w") as f:
                json.dump(data, f)
            log_message(f"Migrated legacy state file {legacy} -> {state_file}")
    except Exception as e:
        log_message(f"Legacy state migration skipped: {e}", level=logging.WARNING)


def load_condition_file():
    global condition_file_path
    state_file = get_state_file()
    _migrate_legacy_state_file(state_file)
    print(path_to_ab(str(state_file)))
    try:
        with open(state_file, "r") as file:
            data = json.load(file)
            if "config_file" in data:
                condition_file_path = data["config_file"]
            else:
                condition_file_path = default_condition_file_path
    except (FileNotFoundError, json.JSONDecodeError):
        condition_file_path = default_condition_file_path
        create_default_json()  # Create the default JSON file


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
    root.withdraw()  # Do not show the Tk root window
    tk_file = filedialog.askopenfilename(
        title="Select Condition file",
        filetypes=[("Condition yaml file", ".yml .yaml")],  # file filter
    )  # Show the file selection dialog
    is_file = os.path.isfile(tk_file)
    if is_file:
        condition_file_path = path_to_ab(tk_file)
        update_json_config_file(condition_file_path)  # Update the JSON file
    else:
        condition_file_path = path_to_ab(default_condition_file_path)
    eel.displayFilePath(condition_file_path)  # Send the file path to the JavaScript function
    print(condition_file_path)


def update_json_config_file(new_path):
    data = {"config_file": new_path}
    with open(get_state_file(), "w") as file:
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


@eel.expose
def run_evaluate_gui():
    _launch_gui(
        [sys.executable, "-c", "from yoru import evaluation_GUI; evaluation_GUI.main()"],
        "Evaluate GUI",
    )


@eel.expose
def run_config_creator_gui():
    subprocess.Popen(
        [sys.executable, "-c", "from yoru import config_creator_GUI; config_creator_GUI.main()"],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )


def main():
    setup_logging()  # write runtime logs to ~/.yoru/logs/yoru.log
    load_condition_file()  # Load the configuration file
    eel.init("web")
    eel.start("gui_home.html", size=(1024, 768), port=8889)


if __name__ == "__main__":
    main()
