import cv2
import numpy as np


def remove_background_using_average(input_file, output_file, avg_frames=1000):
    # 動画を読み込む
    cap = cv2.VideoCapture(input_file)

    # 出力ファイルの設定
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 背景の平均画像を計算
    avg_background = None
    for i in range(avg_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if avg_background is None:
            avg_background = np.float32(frame)
        else:
            cv2.accumulateWeighted(frame, avg_background, alpha=0.01)

    # 背景を減算
    threshold = 1
    red_threshold = 80
    blue_threshold = 80
    green_threshold = 80
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 背景との差を計算
        # cv2.imshow("diff2",cv2.convertScaleAbs(avg_background))
        # mask = (frame[:,:,2] <= red_threshold) & (frame[:,:,0] <= blue_threshold)& (frame[:,:,1] <= green_threshold)
        # frame[mask] = [0, 0, 255]
        # cv2.imshow("window23", frame)
        diff = cv2.absdiff(frame, cv2.convertScaleAbs(avg_background))

        # フレームから背景を除去
        _, fg_mask = cv2.threshold(diff, 50, 100, cv2.THRESH_BINARY)
        fg_mask[np.where((fg_mask >= threshold).any(axis=2))] = [255, 255, 255]
        cv2.imshow("window", fg_mask)
        # print(_)
        # print(fg_mask)
        # print(frame.shape)
        # print(fg_mask.shape)
        # print(fg_mask.dtype)

        #     #繰り返し分から抜けるためのif文
        key = cv2.waitKey(10)
        if key == 27:
            break
        fg = cv2.bitwise_and(frame, fg_mask)
        # fg2 = cv2.bitwise_not(frame, fg_mask)
        # fg3 = cv2.bitwise_and(frame, fg2)
        cv2.imshow("window2", fg)
        # cv2.imshow("window4", fg2)

        # cv2.imshow("window5", fg3)

        # 結果を出力
        out.write(fg)

    # リソースを解放する
    cap.release()
    out.release()


# 使用例
input_file = (
    "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber2.mp4"
)
output_file = "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber2_back_del_mask.mp4"
remove_background_using_average(input_file, output_file)

input_file2 = (
    "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber3.mp4"
)
output_file2 = "C:/Users/nokai\Desktop/231122_andosan_usb_Data/test_data/P1000348_chamber3_back_del_mask.mp4"
remove_background_using_average(input_file2, output_file2)
