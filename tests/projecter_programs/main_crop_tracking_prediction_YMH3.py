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

###クロップして，ハエの体軸を算出###


# YOLOモデルのロード
def getModel():
    model = torch.hub.load(
        "./YoRu-Live/yolov5",
        "custom",
        path="C:/Users/nokai/Desktop/in_hashimoto_labos/procam_work/YoRu-Live/yolo_model/best.pt",
        source="local",
    )
    model = model.to(torch.device("cuda"))
    model = model.eval()
    return model


model = getModel()
# print(model)

# tisgrabber_x64.dllをインポートする
ic = ctypes.cdll.LoadLibrary("./src_prepare/tisgrabber_x64.dll")
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

cam2proj_mat = np.load("camera_calibration.npy")
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
# proj_image = white_proj_img.copy()
proj_image = black_proj_img.copy()
detection_image = white_proj_img.copy()

prediction_image = white_proj_img.copy()

crop_img = np.ones((50, 50, 3), np.uint8) * 0
crop_img_draw = np.ones((500, 500, 3), np.uint8) * 0
crop_img_add = np.ones((100, 100, 3), np.uint8) * 0
# detection_image = black_proj_img.copy()
projection_image(proj_image)

# 円の半径 (px)
circle_radiius = 5

time_result = [(0, 0)]
munkres = Munkres()


def cal_id(pre_mat, cur_mat):
    actual_cur_num = len(cur_mat)
    actual_pre_num = len(pre_mat)

    while len(cur_mat) > len(pre_mat):
        pre_mat.append((-1000, -1000))
    while len(pre_mat) > len(cur_mat):
        cur_mat.append((-1000, -1000))

    if cur_mat is None:
        return None
    pre_mat = torch.tensor(pre_mat).type(torch.float64)
    cur_mat = torch.tensor(cur_mat).type(torch.float64)

    matrix = torch.cdist(pre_mat, cur_mat)
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


# ハエの輪郭解析
def fly_direction_detect(image):
    # グレースケール化
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 閾値処理
    ret, thresh = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        # 楕円にフィットする最小の矩形を取得
        if len(cnt) > 5:
            ellipse = cv2.fitEllipse(cnt)

    return ellipse


def ellipse_long_radius(ellipse):
    # 長径と短径を取得s
    a, b = ellipse[1][0], ellipse[1][1]
    # 楕円の中心点
    center = ellipse[0]
    # 楕円の傾き角度
    angle = ellipse[2]

    # 長径に沿った直線を描くための端点を計算
    if angle < 90:
        angle = -angle
    else:
        angle = 180 - angle

    angle_rad = np.deg2rad(angle)
    x1 = int(center[0] + max(a, b) / 2 * np.sin(angle_rad))
    y1 = int(center[1] + max(a, b) / 2 * np.cos(angle_rad))
    x2 = int(center[0] - max(a, b) / 2 * np.sin(angle_rad))
    y2 = int(center[1] - max(a, b) / 2 * np.cos(angle_rad))

    return (x1, y1), (x2, y2)


# 次の点の予測の関数定義
def linear_regression(x, y):
    n = len(x)
    sum_x, sum_y, sum_x2, sum_xy = 0, 0, 0, 0

    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_x2 += x[i] * x[i]
        sum_xy += x[i] * y[i]

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    b = (sum_y - m * sum_x) / n

    return m, b


def predict_next_point(coordinates):
    x_coords = [coord[0] for coord in coordinates]
    y_coords = [coord[1] for coord in coordinates]

    m_x, b_x = linear_regression(list(range(len(x_coords))), x_coords)
    m_y, b_y = linear_regression(list(range(len(y_coords))), y_coords)

    next_x = m_x * len(x_coords) + b_x
    next_y = m_y * len(y_coords) + b_y

    return (next_x, next_y)


# 以前の位置情報を入力する
pre_center_pos = []
# history_pos = []

# トラッキング用
pre_ids = []
global_counter = 0
tracking_id1 = 0
tracking_id2 = 1
tracking_id3 = 2

# 位置情報予測用
pre_position = []
next_position_pre = (0, 0)

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

    if results.size:
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

            cur_center_pos.append(pos_center)

            cv2.circle(
                proj_image,
                center=calibration_pos_center,
                radius=circle_radiius,
                color=(255, 0, 0),
                thickness=-1,
                lineType=cv2.LINE_4,
                shift=0,
            )
        id_matrix = cal_id(pre_center_pos, cur_center_pos)

        # トラッキングの実装
        cur_ids = []
        # id_matrixを今のフレームで並び替える
        id_matrix.sort(key=lambda x: x[1] if x[1] >= 0 else float("inf"))
        for ids in id_matrix:
            if ids[0] == -1 or ids[1] == -1:
                cur_ids.append(global_counter)
                global_counter += 1
            else:
                cur_ids.append(pre_ids[ids[0]])

        # トラッキングの描写
        if pre_ids and cur_ids:
            if tracking_id1 in cur_ids and pre_ids:
                pre_index1 = pre_ids.index(tracking_id1)
                cur_index1 = cur_ids.index(tracking_id1)
                center_x = int(cur_center_pos[cur_index1][0])
                center_y = int(cur_center_pos[cur_index1][1])
                crop_img = cam_image[
                    center_y - 20 : center_y + 20, center_x - 20 : center_x + 20
                ]

                try:
                    # ハエの体軸検出
                    ellipse = fly_direction_detect(crop_img)
                    A_pos, B_pos = ellipse_long_radius(ellipse)
                    # cv2.ellipse(crop_img, ellipse, (255, 0, 0), 2)
                    cv2.line(crop_img, A_pos, B_pos, (255, 0, 0), 1)
                except UnboundLocalError:
                    pass
                cv2.rectangle(
                    crop_img,
                    (0, 0),
                    (crop_img.shape[1] - 1, crop_img.shape[0] - 1),
                    color=(0, 255, 0),
                    thickness=2,
                )
                crop_img_draw = cv2.resize(crop_img, (500, 500))
                crop_image_add = cv2.resize(crop_img, (200, 200))

                # 画像の重ね合わせ
                try:
                    cam_image[
                        center_y - 100 : center_y + 100, center_x - 100 : center_x + 100
                    ] = crop_image_add
                except ValueError:
                    pass

                # 過去5個のプロットを元にした次の点の予測
                pre_position.append((center_x, center_y))
                if len(pre_position) > 5:
                    pre_position.pop(0)

                try:
                    next_position_pre = predict_next_point(pre_position)
                    next_position_pre = (
                        int(next_position_pre[0]),
                        int(next_position_pre[1]),
                    )

                    # 点の補正
                    center_pos_now = cam2proj_point_coord((center_x, center_y))
                    next_position_pre = cam2proj_point_coord(next_position_pre)

                    # 実際の値の点(赤)
                    cv2.circle(
                        prediction_image, center_pos_now, 5, (0, 0, 255), thickness=-1
                    )
                    # 予測の値の点(青)
                    cv2.circle(
                        prediction_image,
                        next_position_pre,
                        5,
                        (255, 0, 0),
                        thickness=-1,
                    )
                except ZeroDivisionError:
                    pass
                # print(pre_position)
                # print(next_position_pre)

            else:
                # print(cur_ids)
                tracking_id1 = cur_ids[0]

        pre_ids = cur_ids
        pre_center_pos = cur_center_pos

    t4 = time.perf_counter()

    # 　ハエに投射する
    # projection_image(proj_image)
    # projection_image(detection_image)
    t5 = time.perf_counter()
    cv2.imshow("Window", cam_image)
    cv2.imshow("prj_view", proj_image)
    # cv2.imshow('det_view_2', detection_image)
    cv2.imshow("crop_view", crop_img_draw)
    cv2.imshow("prediction", prediction_image)
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
