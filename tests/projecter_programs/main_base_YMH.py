import csv
import ctypes
import os
import time

import cv2
import numpy as np
import pandas as pd
import tisgrabber as tis
import torch
import win32gui
from munkres import Munkres
from PIL import Image


# YOLOモデルのロード
def getModel():
    model = torch.hub.load(
        "./yolov5",
        "custom",
        path="C:/Users/nokai/Desktop/in_hashimoto_labos/procam_work/YoRu-Live/yolov5s.pt",
        source="local",
    )
    model = model.to(torch.device("cuda"))
    model = model.eval()
    return model


model = getModel()
# print(model)

# tisgrabber_x64.dllをインポートする
ic = ctypes.cdll.LoadLibrary("./projecter_programs/tisgrabber_x64.dll")
tis.declareFunctions(ic)

# ICImagingControlクラスライブラリを初期化します。
# この関数は、このライブラリの他の関数が呼び出される前に1回だけ呼び出す必要があります。
ic.IC_InitLibrary(0)

# ダイアログ画面を表示，2回目以降はdevice.xmlを参照
hGrabber = tis.openDevice(ic)

# デバイスが有効か確認
if not ic.IC_IsDevValid(hGrabber):
    print("デバイスが有効ではありません。")
    exit()

# ライブスタート開始　引数：0の時非表示、引数：1の時表示
# ライブ画面は非表示にしています。
ic.IC_StartLive(hGrabber, 0)

# プロジェクタの設定
proj_width = 1280
proj_height = 800
proj_pos_x = -proj_width
proj_pos_y = -proj_height

dummy_img = np.zeros((proj_height, proj_width, 3), np.uint8)

# 全画面表示でプロジェクター出力
proj_window_name = "screen"
cv2.namedWindow(proj_window_name, cv2.WINDOW_NORMAL)
# cv2.setWindowProperty(proj_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(proj_window_name, dummy_img)
cv2.waitKey(1)

# ウインドウを移動
proj_win = win32gui.FindWindow(None, proj_window_name)
win32gui.MoveWindow(proj_win, proj_pos_x, proj_pos_y, proj_width, proj_height, True)

# 全画面表示に
cv2.setWindowProperty(proj_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow(proj_window_name, dummy_img)
cv2.waitKey(1)


# カメラ画像を取得
def capture_image(num_trial=-1):
    while True:
        if ic.IC_SnapImage(hGrabber, 2000) == tis.IC_SUCCESS:
            # 成功で抜ける
            break
        if num_trial == 0:
            # 試行回数の上限
            return None
        num_trial = num_trial - 1
        continue

    # 画像記述の変数を宣言する
    Width = ctypes.c_long()
    Height = ctypes.c_long()
    BitsPerPixel = ctypes.c_int()
    colorformat = ctypes.c_int()

    # 画像の説明の値を取得する
    ic.IC_GetImageDescription(hGrabber, Width, Height, BitsPerPixel, colorformat)

    # バッファサイズを計算
    bpp = int(BitsPerPixel.value / 8.0)
    buffer_size = Width.value * Height.value * BitsPerPixel.value

    # イメージデータを取得する
    imagePtr = ic.IC_GetImagePtr(hGrabber)

    imagedata = ctypes.cast(imagePtr, ctypes.POINTER(ctypes.c_ubyte * buffer_size))

    # numpyの配列を作成する
    image = np.ndarray(
        buffer=imagedata.contents,
        dtype=np.uint8,
        shape=(Height.value, Width.value, bpp),
    )
    return image


# プロジェクター出力
def projection_image(image, wait=False):
    cv2.imshow("screen", image)
    if wait:
        cv2.waitKey(1)


# サークルグリッドパターンの生成
def generate_circle_grid(
    proj_width=1280,
    proj_height=800,
    circle_num_h=5,
    circle_num_w=8,
    circle_radius=16,
    circle_interval=96,
):
    proj_img = np.ones((proj_height, proj_width, 3), np.uint8) * 255

    points = []

    for y in range(circle_num_h):
        # y = circle_num_h - y - 1
        for x in range(circle_num_w):
            # カメラ・プロジェクタ配置の都合でxを逆順に
            x = circle_num_w - x - 1

            center_x = x - (circle_num_w - 1) / 2
            center_y = y - (circle_num_h - 1) / 2
            center_x = center_x * circle_interval
            center_y = center_y * circle_interval
            center_x = center_x + proj_width / 2
            center_y = center_y + proj_height / 2
            center_x = int(center_x)
            center_y = int(center_y)

            points.append((center_x, center_y))

            cv2.circle(
                proj_img, (center_x, center_y), circle_radius, (0, 0, 0), thickness=-1
            )

    points = np.array(points)
    return proj_img, points


# キャリブレーション，パターン投影して検出したら進む
circle_grid_image, proj_centers = generate_circle_grid(
    proj_width=proj_width, proj_height=proj_height
)
projection_image(circle_grid_image)

# while True:
#     cam_image = capture_image()
#     resized_img = cv2.resize(cam_image,(640, 480))
#     cv2.imshow('Window', resized_img)
#     found, corners = cv2.findCirclesGrid(cam_image, (8, 5), cv2.CALIB_CB_ASYMMETRIC_GRID)
#     #cv2.drawChessboardCorners(cam_image, (8, 5), corners, found)
#     #resized_img = cv2.resize(cam_image,(640, 480))
#     #cv2.imshow('Window', resized_img)

#     key = cv2.waitKey(1)
#     if found:
#         # 検出できた
#         break

# # プロジェクタ・カメラ間のホモグラフィー変換を計算
# corners = corners.reshape(proj_centers.shape)
# cam2proj_mat, _ = cv2.findHomography(corners.astype(np.float32), proj_centers.astype(np.float32))
# print(cam2proj_mat)

cam2proj_mat = np.load("./projecter_programs/camera_calibration.npy")
print(cam2proj_mat)


# カメラ画像上の座標値をプロジェクタ画像上の座標値に変換
def cam2proj_point_coord(p):
    p = np.array([p[0], p[1], 1.0])
    p = cam2proj_mat @ p
    p = (int(p[0] / p[2]), int(p[1] / p[2]))
    return p


# 映像描写
white_proj_img = np.ones((proj_height, proj_width, 3), np.uint8) * 255
black_proj_img = np.ones((proj_height, proj_width, 3), np.uint8) * 0
color_proj_img = np.ones((proj_height, proj_width, 3), np.uint8) * 130
proj_image = white_proj_img.copy()
projection_image(proj_image)

# 円の半径 (px)
circle_radiius = 10

time_result = [(0, 0)]
munkres = Munkres()


def cal_id(pre_mat, cur_mat):
    # 　距離計算
    # if len(pre_mat) == len(cur_mat):
    # if len(pre_mat) == 0 or len(cur_mat)== 0:
    #    pass
    # if len(cur_mat)== 0:
    #    return [(i, -1) for i in range(len(cur_mat))]

    # if len(pre_mat) == 0:
    #    return [(-1, i) for i in range(len(cur_mat))]

    actual_cur_num = len(cur_mat)
    actual_pre_num = len(pre_mat)

    while len(cur_mat) > len(pre_mat):
        pre_mat.append((-1000, -1000))
    while len(pre_mat) > len(cur_mat):
        cur_mat.append((-1000, -1000))

    pre_mat = torch.tensor(pre_mat).type(torch.float64)
    cur_mat = torch.tensor(cur_mat).type(torch.float64)
    matrix = torch.cdist(pre_mat, cur_mat)
    # matrix = np.zeros((len(pre_mat), len(cur_mat)))
    # for i in range(len(pre_mat)):
    #     for j in range(len(cur_mat)):
    #         matrix[i, j] = np.linalg.norm(np.array(pre_mat[i]) - np.array(cur_mat[j]))

    matrix = matrix.numpy()
    match_mat = munkres.compute(matrix)

    ret_match_mat = []

    for i, j in match_mat:
        if i >= actual_pre_num:
            i = -1
        if j >= actual_cur_num:
            j = -1
        ret_match_mat.append((i, j))

    return ret_match_mat


# def decide_id():


# 以前の位置情報を入力する
# pre_center_pos = [1]
pre_center_pos = []


while True:
    start_time = time.perf_counter()

    cam_image = capture_image()
    t1 = time.perf_counter()
    # proj_image = cv2.warpPerspective(cam_image, cam2proj_mat, (cam_image.shape[1], cam_image.shape[0]))
    # resized_img = cv2.resize(cam_image,(640, 480))

    # YOLOの認識
    with torch.no_grad():
        # サイズ縮小1/4
        # cam_image = cv2.resize(cam_image, (cam_image.shape[1] // 4, cam_image.shape[0] // 4))
        detection = model(cam_image)
    t2 = time.perf_counter()
    results = detection.xyxy[0].detach().cpu().numpy()

    t3 = time.perf_counter()
    cur_center_pos = []
    try:
        for box in results:
            pos_lt = np.array([box[0], box[1]])
            pos_rd = np.array([box[2], box[3]])
            pos_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

            # 画面端描写
            # pos_lt = np.array([0, 0])
            # pos_rd = np.array([800, 800])

            calibration_pos_lt = cam2proj_point_coord(pos_lt)
            calibration_pos_rd = cam2proj_point_coord(pos_rd)
            calibration_pos_center = cam2proj_point_coord(pos_center)

            cur_center_pos.append(calibration_pos_center)

            cv2.rectangle(
                proj_image,
                pt1=calibration_pos_lt,
                pt2=calibration_pos_rd,
                color=(0, 135, 0),
                thickness=-1,
                lineType=cv2.LINE_4,
                shift=0,
            )
            # cv2.circle(proj_image,
            #     center=calibration_pos_center,
            #     radius=circle_radiius,
            #     color=(135, 135, 0),
            #   thickness=-1,
            #   lineType=cv2.LINE_4,
            #   shift=0)
        # id_matrix = cal_id(pre_center_pos, cur_center_pos)
        # print(id_matrix)
        pre_center_pos = cur_center_pos

    except IndexError:
        pass

    t4 = time.perf_counter()

    projection_image(proj_image)
    t5 = time.perf_counter()
    detection.render()
    cv2.imshow("Window", cam_image)
    cv2.imshow("prj_view", proj_image)
    t6 = time.perf_counter()
    # 描写のし直し
    proj_image = black_proj_img.copy()

    time_list = [
        start_time,
        t1 - start_time,
        t2 - t1,
        t3 - t2,
        t4 - t3,
        t5 - t4,
        t6 - t5,
    ]
    time_result.append(time_list)

    key = cv2.waitKey(1)
    if key == ord(" "):
        print("cap")
        continue
    if key == ord("q"):
        break
    if key == ord("r"):
        projection_image(white_proj_img)
        cv2.waitKey(10)
        break

# 後処理
cv2.destroyAllWindows()
ic.IC_StopLive(hGrabber)
ic.IC_ReleaseGrabber(hGrabber)
ic.IC_CloseLibrary()
# # リストをデータフレームに変換
# df_results = pd.DataFrame(time_result, columns=['s', 't1', 't2', 't3', 't4', 't5', 't6'])
# # csvとして出力
# df_results.to_csv("./time_result.csv", index=False)
