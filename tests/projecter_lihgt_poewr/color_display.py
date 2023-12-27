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

proj_height = 1080
proj_width = 1920
proj_pos_x = -proj_width
proj_pos_y = -proj_height
back_BGR = (255, 0, 100)


# 円描写関数
def draw_green_circle(img, BGR, calibration_pos_center=(500, 500), circle_radiius=100):
    cv2.circle(
        img,
        center=calibration_pos_center,
        radius=circle_radiius,
        color=BGR,
        thickness=-1,
        lineType=cv2.LINE_4,
        shift=0,
    )
    return img


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


# プロジェクター出力
def projection_image(image, wait=False):
    cv2.imshow("screen", image)
    if wait:
        cv2.waitKey(1)


# 　黒画像
black_img = np.ones((proj_height, proj_width, 3), np.uint8) * 0

# RGB値を指定
R = 0
G = 255
B = 0
# RGB指定画像
color_img = np.ones((proj_height, proj_width, 3), np.uint8)
color_img[:, :, 0] = B  # Redチャンネル
color_img[:, :, 1] = G  # Greenチャンネル
color_img[:, :, 2] = R  # Blueチャンネル

# 円の描写
# proj_img = draw_green_circle(black_img, back_BGR)
proj_image = color_img

# 表示
while True:
    projection_image(proj_image)
    cv2.imshow("Window", proj_image)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
