import io
import json
import os
import platform
import time
from datetime import date, datetime

import cv2
import GPUtil
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import PySimpleGUI as sg
import seaborn as sns
import torch
from PIL import Image
from tqdm import tqdm


def get_system_info():
    uname_info = platform.uname()
    virtual_memory = psutil.virtual_memory()
    gpus = GPUtil.getGPUs()
    system_info = []
    for gpu in gpus:
        info = {
            "OS_system": uname_info.system,
            "OS_release": uname_info.release,
            "OS_version": uname_info.version,
            "CPU_machine": uname_info.machine,
            "CPU_name": uname_info.processor,
            "CPU_physical_cores": psutil.cpu_count(logical=False),
            "CPU_total_cores": psutil.cpu_count(logical=True),
            "CPU_max_frequency": psutil.cpu_freq().max,
            "Memory_total": virtual_memory.total,
            "GPU_ID": gpu.id,
            "GPU_name": gpu.name,
            "GPU_memory_total": gpu.memoryTotal,
        }
        system_info.append(info)
    return system_info


# YOLOモデルのロード
def getModel(model_path):
    model = torch.hub.load("./yolov5", "custom", path=model_path, source="local")
    model = model.to(torch.device("cuda"))
    model = model.eval()
    return model


def calculate_speed(model_path, movie_path, window, progress_bar):
    # モデルの読み込み
    model = getModel(model_path)

    # 動画パスの取得
    video_path = movie_path
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 時間計測
    time_list = []

    with torch.no_grad():
        # 動画処理
        for frame_num in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()

            if not ret:
                break

            t1 = time.perf_counter()
            # YOLOv5の推論を実行
            # with torch.no_grad():
            results = model(frame)

            t2 = time.perf_counter()

            elapsed_time = t2 - t1
            time_list.append((frame_num, elapsed_time))

            # プログレスバーの更新
            progress = (frame_num + 1) / total_frames * 1000  # 1000はProgressBarの最大値
            progress_bar.UpdateBar(progress)
            window.Refresh()  # ウィンドウの内容を更新

        cap.release()
    return time_list


def save_info(model_path, output_dir, time_list):
    # システム情報の取得
    system_info = get_system_info()

    # GPU名を取得
    gpu_name = system_info[0]["GPU_name"]

    # 現在の日時を取得
    now = datetime.now()

    base_name = os.path.basename(model_path)  # ファイル名を含む拡張子を取得
    model_name, _ = os.path.splitext(base_name)  # 拡張子を除いたファイル名を取得
    filename = (
        model_name
        + "_"
        + gpu_name
        + "_"
        + str(now.year)
        + str(now.month)
        + str(now.day)
    )
    system_info[0]["yolo_model_path"] = model_path
    system_info[0]["yolo_model_name"] = model_name
    system_info[0]["file_name"] = filename
    system_info[0]["day"] = str(date.today())

    filename_json = filename + ".json"  # 拡張子の確認と追加
    csv_name = filename + ".csv"  # 拡張子の確認と追加

    result_json = os.path.join(output_dir, filename_json)
    result_csv = os.path.join(output_dir, csv_name)

    # system infoの保存
    with open(result_json, "w") as f:
        json.dump(system_info, f, indent=4)

    # csvの保存
    df = pd.DataFrame(time_list, columns=["Frame", "Time"])
    df.to_csv(result_csv, index=False)
    return filename, system_info


def plot_inference_times(time_list, system_info, output_dir, filename):
    df = pd.DataFrame(time_list, columns=["Frame", "Time"])

    sns.violinplot(y=df["Time"])

    # x軸にすべての情報を表示
    info_str = "\n".join(
        [
            f"YOLO Model: {system_info[0]['yolo_model_name']}",
            f"OS: {system_info[0]['OS_system']}",
            # f"Release: {system_info[0]['OS_release']}",
            f"Processor: {system_info[0]['CPU_name']}",
            f"Memory Total: {system_info[0]['Memory_total']}",
            f"GPU: {system_info[0]['GPU_name']}",
        ]
    )
    plt.xlabel(info_str)

    ymin, ymax = plt.ylim()  # Get the current y-axis limits
    ymax = max(df["Time"]) + (ymax - ymin) * 0.05 * 11
    plt.ylim(0, ymax)
    text_y_offset = (
        ymax - ymin
    ) * 0.05  # Calculate a 5% offset based on the range of y values

    plt.text(0, ymax - 2 * text_y_offset, system_info[0]["OS_system"], ha="center")
    plt.text(0, ymax - 3 * text_y_offset, system_info[0]["CPU_machine"], ha="center")
    plt.text(0, ymax - 4 * text_y_offset, system_info[0]["CPU_name"], ha="center")
    plt.text(0, ymax - 5 * text_y_offset, system_info[0]["CPU_machine"], ha="center")
    plt.text(
        0, ymax - 6 * text_y_offset, str(system_info[0]["Memory_total"]), ha="center"
    )  # Convert memory_total to string
    plt.text(0, ymax - 7 * text_y_offset, system_info[0]["GPU_name"], ha="center")

    plot_path = os.path.join(output_dir, filename + "_plot.png")
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(plot_path)
    plt.close()  # 保存後にプロットをクローズする
    return plot_path


def get_img_data(f):
    """Generate image data using PIL for displaying in PySimpleGUI"""
    img = Image.open(f)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


if __name__ == "__main__":
    layout = [
        [
            sg.Text("YOLOv5 Model Path"),
            sg.InputText(key="model_path", disabled=True),
            sg.FileBrowse(file_types=(("YOLO model", "*.pt"),)),
        ],
        [
            sg.Text("Video Path"),
            sg.InputText(key="video_path", disabled=True),
            sg.FileBrowse(file_types=(("Movie Files", "*.mp4;*.avi;*.mov;*.wav"),)),
        ],
        [
            sg.Text("Output folder"),
            sg.InputText(key="output_folder", disabled=True),
            sg.FolderBrowse(),
        ],
        [sg.Button("Run Inference and Save Results")],
        [sg.ProgressBar(1000, orientation="h", size=(20, 20), key="progressbar")],
        [sg.Image(key="plot_image")],
        [sg.Exit()],
    ]

    window = sg.Window("YOLOv5 Inference Time Measurement", layout)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == "Exit":
            break
        elif event == "Run Inference and Save Results":
            # Update paths from GUI input
            model_path = values["model_path"]
            video_path = values["video_path"]

            # Load model and calculate speeds
            progress_bar = window["progressbar"]
            time_list = calculate_speed(model_path, video_path, window, progress_bar)

            # Save results
            filename, system_info = save_info(
                model_path, values["output_folder"], time_list
            )

            # Save the plot and get its path
            plot_image_path = plot_inference_times(
                time_list, system_info, values["output_folder"], filename
            )

            # Plotの画像をGUIに表示 (修正箇所)
            img_data = get_img_data(plot_image_path)
            window["plot_image"].update(data=img_data)

    window.close()
