import csv
import ctypes
import os
import time

import cv2
import nidaq as daq
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports
import tisgrabber as tis
import torch
from munkres import Munkres
from PIL import Image


class trigger_condition:
    def __init__(self):
        print("trigger_command")
        self.mydaqDO = daq.dio(
            devID="Dev1", taskType="do", port="port0", lineCh="line0:1"
        )

    def trigger(self, tri_cl, results):
        class_ids = results[:, 5].astype(int)
        # print(class_ids)

        if tri_cl in class_ids:
            self.mydaqDO.writeDO([True, True])
        else:
            self.mydaqDO.writeDO([False, False])


# YOLOモデルのロード
def getModel():
    model = torch.hub.load(
        "./yolov5",
        "custom",
        path="C:/Users/nokai/Desktop/LED_projects/models/640_projects/640_yolov5n_best.pt",
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

daq_trgger = trigger_condition()


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


with torch.no_grad():
    while True:
        start_time = time.perf_counter()

        cam_image = capture_image()

        # YOLOの認識

        # サイズ縮小1/4
        # cam_image = cv2.resize(cam_image, (cam_image.shape[1] // 4, cam_image.shape[0] // 4))
        detection = model(cam_image)
        results = detection.xyxy[0].detach().cpu().numpy()

        daq_trgger.trigger(1, results)
        detection.render()

        # detection.render()
        cv2.imshow("Window", cam_image)

        key = cv2.waitKey(1)
        if key == ord(" "):
            print("cap")
            continue
        if key == ord("q"):
            break
        if key == ord("r"):
            cv2.waitKey(10)
            break

# 後処理
cv2.destroyAllWindows()
ic.IC_StopLive(hGrabber)
ic.IC_ReleaseGrabber(hGrabber)
ic.IC_CloseLibrary()
