import json
import os
import subprocess
import sys
from pathlib import Path
from tkinter import Tk, filedialog

import eel

# if __name__ == "__main__":


sys.path.append("../yoru")

# try:
from yoru import analysis_GUI, evaluation_GUI, realtime_yoru_GUI, train_GUI

# except(ModuleNotFoundError):
#     import analysis_GUI
#     import realtime_yoru_GUI
#     import evaluation_GUI
#     import train_GUI


default_condition_file_path = "./config/yoru_default.yaml"
condition_file_path = default_condition_file_path


def create_default_json():
    log_dir = "./logs"
    log_file_path = f"{log_dir}/condition_file_log.json"

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    default_data = {"config_file": default_condition_file_path}
    with open(log_file_path, "w") as file:
        json.dump(default_data, file)


def load_condition_file():
    global condition_file_path
    log_file_path = "./logs/condition_file_log.json"
    print(path_to_ab(log_file_path))
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
    if os.path.isfile(condition_file_path):
        realtime_yoru_GUI.main(condition_file_path)
        # subprocess.Popen(["python", "cam_gui_YMH.py"])
    else:
        print("not find config file")


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
    analysis_GUI.main()
    # subprocess.Popen(["Python", "analy_GUI.py"])


@eel.expose
def run_train_gui():
    train_GUI.main()


@eel.expose
def run_evaluate_gui():
    evaluation_GUI.main()
    # subprocess.call(["python", "model_eval_gui.py"])


def main():
    load_condition_file()  # 設定ファイルを読み込む
    eel.init("web")
    eel.start("gui_home.html", size=(1024, 768), port=8080)


if __name__ == "__main__":
    main()
