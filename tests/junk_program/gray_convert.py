import cv2


def convert_video_to_grayscale(input_file, output_file):
    # 動画を読み込む
    cap = cv2.VideoCapture(input_file)

    # 出力ファイルの設定
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), isColor=False)

    # フレームを読み込み、グレースケールに変換し、書き出す
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(gray_frame)

    # リソースを解放する
    cap.release()
    out.release()


# 使用例
input_file = (
    "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber2.mp4"
)
output_file = "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber2_gray.mp4"
convert_video_to_grayscale(input_file, output_file)

input_file2 = (
    "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber3.mp4"
)
output_file2 = "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber3_gray.mp4"
convert_video_to_grayscale(input_file2, output_file2)
