import time

import cv2
import numpy as np


class trigger_condition:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        window_width = 200
        window_height = 200
        self.window_name = "Color Changer"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_height, window_width)

        self.black_img = np.ones((window_height, window_width, 3), np.uint8) * 0

        self.red_img = np.zeros((window_height, window_width, 3), np.uint8)
        self.red_img[:] = (0, 0, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        textsize = cv2.getTextSize("Detected", font, 1, 2)[0]
        textX = (self.red_img.shape[1] - textsize[0]) // 2
        textY = (self.red_img.shape[0] + textsize[1]) // 2

        cv2.putText(self.red_img, "Detected", (textX, textY), font, 1, (0, 0, 0), 2)

        cv2.imshow(self.window_name, self.black_img)
        cv2.waitKey(1)
        print("trigger_command")

    def trigger(self, tri_cl, in_cl, Arduino, results, now):
        if tri_cl in in_cl:
            proj_img = self.red_img
        else:
            proj_img = self.black_img

        cv2.imshow(self.window_name, proj_img)
        key = cv2.waitKey(1)

        # If 'q' is pressed, exit the loop
        if key == ord("q"):
            return None
