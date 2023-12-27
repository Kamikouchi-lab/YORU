import glob
import os

import cv2
from tqdm import tqdm

# リサイズしたいフォルダのパス
folder_path = "C:/Users/nokai/Desktop/231122_andosan_usb_Data/test_data"

# リサイズ後の動画を保存するフォルダのパス (存在しない場合は作成されます)
output_folder = "C:/Users/nokai/Desktop/resized"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# フォルダ内の全てのAVI動画ファイルを取得
for file_path in tqdm(glob.glob(os.path.join(folder_path, "*.avi"))):
    # ビデオキャプチャを作成
    cap = cv2.VideoCapture(file_path)

    # 元の動画のプロパティを取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # 出力ファイルのパスを設定
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_folder, base_name)

    # リサイズ後の動画を保存するためのビデオライターを作成
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))

    # 各フレームを読み込み、リサイズして書き込む
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (640, 480))
        out.write(resized_frame)

    # リソースを解放
    cap.release()
    out.release()

    print(f"{file_path} -> {output_path}")

print("リサイズ完了!")
