import csv
import ctypes
import os
import time

# import tisgrabber as tis
import cv2
import numpy as np
import pandas as pd
import win32gui
from PIL import Image


class trigger_condition:
    def __init__(self):
        print("loading....")
        # プロジェクタの設定

        proj_width = 1280
        proj_height = 720
        proj_pos_x = -proj_width
        proj_pos_y = -proj_height
        dummy_img = np.zeros((proj_height, proj_width, 3), np.uint8)
        # 全画面表示でプロジェクター出力
        self.proj_window_name = "screen"
        cv2.namedWindow(self.proj_window_name, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(proj_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(self.proj_window_name, dummy_img)
        cv2.waitKey(1)
        # ウインドウを移動
        proj_win = win32gui.FindWindow(None, self.proj_window_name)
        win32gui.MoveWindow(
            proj_win, proj_pos_x, proj_pos_y, proj_width, proj_height, True
        )

        # 全画面表示に
        cv2.setWindowProperty(
            self.proj_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.imshow(self.proj_window_name, dummy_img)
        cv2.waitKey(1)

        self.black_proj_img = np.ones((proj_height, proj_width, 3), np.uint8) * 0
        # cv2.imshow('prj_view', self.black_proj_img)
        # cv2.waitKey(1)

        self.cam2proj_mat = np.load("./camera_calibration.npy")
        print(self.cam2proj_mat)

        self.past_time = 0
        # cv2.imshow('prj_view', self.black_proj_img)

        print("load trigger plugin!")
        # pass

    # プロジェクター出力
    def projection_image(self, image, wait=False):
        cv2.imshow(self.proj_window_name, image)
        if wait:
            cv2.waitKey(1)

    # カメラ画像上の座標値をプロジェクタ画像上の座標値に変換
    def cam2proj_point_coord(self, p):
        p = np.array([p[0], p[1], 1.0])
        p = self.cam2proj_mat @ p
        p = (int(p[0] / p[2]), int(p[1] / p[2]))
        return p

    def trigger(self, tri_cl, in_cl, ser, results, now):
        # if now != self.past_time:
        proj_image = self.black_proj_img.copy()
        if len(results) > 0:
            # print("a")
            if any(result[5] == tri_cl for result in results):
                for *box, confidence, class_id in results:
                    # print(box)
                    if tri_cl != class_id:
                        # #長方形にする
                        # if (box[0]-box[2]) <= (box[1]-box[3]):
                        #     box[1] = box[1] - ((box[3]-box[1]) /3)
                        #     box[3] = box[3] + ((box[3]-box[1]) /3)
                        #     # box[0] = box[0] - ((box[2]-box[0]) /10)
                        #     # box[2] = box[2] + ((box[2]-box[0]) /10)
                        # else:
                        #     box[0] = box[0] - ((box[2]-box[0]) /3)
                        #     box[2] = box[2] + ((box[2]-box[0]) /3)
                        #     # box[1] = box[1] - ((box[3]-box[1]) /10)
                        #     # box[3] = box[3] + ((box[3]-box[1]) /10)

                        pos_lt = np.array([box[0], box[1]])
                        pos_rd = np.array([box[2], box[3]])
                        pos_center = np.array(
                            [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                        )

                        calibration_pos_lt = self.cam2proj_point_coord(pos_lt)
                        calibration_pos_rd = self.cam2proj_point_coord(pos_rd)
                        calibration_pos_center = self.cam2proj_point_coord(pos_center)

                        cv2.rectangle(proj_image,
                            pt1=calibration_pos_lt,
                            pt2=calibration_pos_rd,
                            color=(0, 255, 0),
                            thickness=-1,
                            lineType=cv2.LINE_4,
                            shift=0)
                        # cv2.circle(
                        #     proj_image,
                        #     center=calibration_pos_center,
                        #     radius=25,
                        #     color=(0, 255, 0),
                        #     thickness=-1,
                        #     lineType=cv2.LINE_4,
                        #     shift=0,
                        # )

        # print(proj_image)
        self.projection_image(proj_image)
        # cv2.imshow('prj_view', proj_image)
        key = cv2.waitKey(1)
        if key == ord("q"):
            return None
        self.past_time = now
