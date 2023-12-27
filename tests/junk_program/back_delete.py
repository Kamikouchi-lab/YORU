import cv2


def remove_background_from_video(input_file, output_file):
    # 動画を読み込む
    cap = cv2.VideoCapture(input_file)

    # 出力ファイルの設定
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 背景減算器を作成
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # フレームごとに背景減算を適用
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 背景減算を適用
        fg_mask = bg_subtractor.apply(frame)

        # フォアグラウンドだけを取得
        fg = cv2.bitwise_and(frame, frame, mask=fg_mask)

        # 結果を出力
        out.write(fg)

    # リソースを解放する
    cap.release()
    out.release()
    print("complete!!!")


# 使用例
input_file = (
    "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber2.mp4"
)
output_file = "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber2_back_del.mp4"
remove_background_from_video(input_file, output_file)

input_file2 = (
    "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber3.mp4"
)
output_file2 = "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber3_back_del.mp4"
remove_background_from_video(input_file2, output_file2)
