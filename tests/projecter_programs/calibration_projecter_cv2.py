import csv
import ctypes
import os
import time

# import tisgrabber as tis
import cv2
import numpy as np
import pandas as pd
import torch
import win32gui
from PIL import Image

# プロジェクタの設定
proj_width = 800
proj_height = 600
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
# def capture_image(num_trial = -1):
#     while True:
#         if ic.IC_SnapImage(hGrabber, 2000) == tis.IC_SUCCESS:
#             # 成功で抜ける
#             break
#         if num_trial == 0:
#             # 試行回数の上限
#             return None
#         num_trial = num_trial-1
#         continue

#     # 画像記述の変数を宣言する
#     Width = ctypes.c_long()
#     Height = ctypes.c_long()
#     BitsPerPixel = ctypes.c_int()
#     colorformat = ctypes.c_int()

#     # 画像の説明の値を取得する
#     ic.IC_GetImageDescription(hGrabber, Width, Height,
#                                 BitsPerPixel, colorformat)

#     # バッファサイズを計算
#     bpp = int(BitsPerPixel.value / 8.0)
#     buffer_size = Width.value * Height.value * BitsPerPixel.value

#     # イメージデータを取得する
#     imagePtr = ic.IC_GetImagePtr(hGrabber)

#     imagedata = ctypes.cast(imagePtr,
#                             ctypes.POINTER(ctypes.c_ubyte *
#                                             buffer_size))

#     # numpyの配列を作成する
#     image = np.ndarray(buffer=imagedata.contents,
#                         dtype=np.uint8,
#                         shape=(Height.value,
#                                 Width.value,
#                                 bpp))
#     return image

camera_height = 480
camera_width = 640
camera_fps = 30
camera_id = 0

print("CV-capture start...")
capture = cv2.VideoCapture(camera_id)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
print(camera_fps)
capture.set(cv2.CAP_PROP_FPS, camera_fps)
capture.set(cv2.CAP_PROP_SETTINGS, 1)
capture.set(cv2.CAP_PROP_BUFFERSIZE, 2000)
(status, frame) = capture.read()
print(status)
# halfImg = cv2.resize(frame, resized_resolution)
print("CV-capture start success")


def capture_image(capture):
    if capture.isOpened():
        (status, frame) = capture.read()
    return frame


# プロジェクター出力
def projection_image(image, wait=False):
    cv2.imshow("screen", image)
    if wait:
        cv2.waitKey(1)


# サークルグリッドパターンの生成
def generate_circle_grid(
    proj_width=800,
    proj_height=600,
    circle_num_h=5,
    circle_num_w=8,
    circle_radius=16,
    circle_interval=96,
):
    proj_img = np.ones((proj_height, proj_width, 3), np.uint8) * 200

    points = []

    for y in range(circle_num_h):
        # y = circle_num_h - y - 1
        for x in range(circle_num_w):
            # カメラ・プロジェクタ配置の都合でxを逆順に
            # x = circle_num_w - x - 1

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
# circle_grid_image, proj_centers = generate_circle_grid(proj_width=proj_width, proj_height=proj_height)
circle_grid_image, proj_centers = generate_circle_grid(
    proj_width=proj_width, proj_height=proj_height, circle_interval=20, circle_radius=8
)
projection_image(circle_grid_image)

while True:
    cam_image = capture_image(capture)
    resized_img = cv2.resize(cam_image, (640, 480))
    cv2.imshow("Window", resized_img)
    found, corners = cv2.findCirclesGrid(
        cam_image, (8, 5), cv2.CALIB_CB_ASYMMETRIC_GRID
    )
    # cv2.drawChessboardCorners(cam_image, (8, 5), corners, found)
    # resized_img = cv2.resize(cam_image,(640, 480))
    # cv2.imshow('Window', resized_img)

    key = cv2.waitKey(1)
    if found:
        # 検出できた
        break

# プロジェクタ・カメラ間のホモグラフィー変換を計算
corners = corners.reshape(proj_centers.shape)
cam2proj_mat, _ = cv2.findHomography(
    corners.astype(np.float32), proj_centers.astype(np.float32)
)
print(cam2proj_mat)

np.save("./projecter_programs/camera_calibration.npy", cam2proj_mat)
