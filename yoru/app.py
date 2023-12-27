import os
import subprocess
from tkinter import Tk, filedialog

import eel

# if __name__ == "__main__":
    
try:
    from yoru import analysis_GUI
    from yoru import realtime_yoru_GUI
    from yoru import evaluation_GUI
    from yoru import train_GUI
except(ModuleNotFoundError):
    import analysis_GUI
    import realtime_yoru_GUI
    import evaluation_GUI
    import train_GUI

# from cam_gui_YMH import mainprocess

# load condition file path
log_file_path = "../logs/condition_file_log.txt"  # ファイル名一覧を取得したいディレクトリのパス
# with open(log_file_path) as f:

# text_file = open(log_file_path, "w") # 書き込み先のテキストファイルを作る

condition_file_path = "../yoru_default.yaml"


@eel.expose
def run_cam_gui_YMH():
    global condition_file_path
    if os.path.isfile(condition_file_path):
        # print("Hello World")
        realtime_yoru_GUI.main(condition_file_path)
        # subprocess.Popen(["python", "cam_gui_YMH.py"])


@eel.expose
def show_file_dialog():
    global condition_file_path
    root = Tk()
    root.withdraw()  # Tkのルートウィンドウを表示しない
    condition_file_path = filedialog.askopenfilename(
        title="Select Condition file",
        filetypes=[("Condition yaml file", ".yml .yaml")],  # ファイルフィルタ-
    )  # ファイル選択ダイアログを表示
    eel.displayFilePath(condition_file_path)  # JavaScript関数にファイルパスを送る
    print(condition_file_path)


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
    eel.init("web")
    eel.start("gui_home.html", size=(1024, 768), port=8080)


if __name__ == "__main__":
    main()
