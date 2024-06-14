import datetime
import time
import tkinter as tk
from threading import Thread
import csv

import cv2
import mss
import numpy as np
from PIL import Image, ImageTk
from pynput import mouse


class capture_streamCV2:
    def __init__(self, srcCam=1, m_dict={}):
        print("CV-initialization...")
        self.m_dict = m_dict
        self.src = srcCam
        self.t0 = self.m_dict["t0"]
        self.default_FPS = self.m_dict["camera_fps"]
        self.resized_resolution = (
            int(self.m_dict["camera_width"] * self.m_dict["camera_scale"]),
            int(self.m_dict["camera_height"] * self.m_dict["camera_scale"]),
        )

        self.frameBufLen = 200
        self.frameBuffer = np.zeros(
            (self.resized_resolution[1], self.resized_resolution[0], self.frameBufLen),
            dtype="uint8",
        )
        self.fmt = cv2.VideoWriter_fourcc("D", "I", "V", "X")
        print("CV-initialization Finished")

    def startCapture(self):
        print("CV-capture start...")
        self.capture = cv2.VideoCapture(self.src + cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.m_dict["camera_width"])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.m_dict["camera_height"])
        print(self.m_dict["camera_fps"])
        self.capture.set(cv2.CAP_PROP_FPS, self.m_dict["camera_fps"])
        self.capture.set(cv2.CAP_PROP_SETTINGS, 1)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2000)
        (status, frame) = self.capture.read()
        print(status)
        halfImg = cv2.resize(frame, self.resized_resolution)
        print("CV-capture start success")

    def run(self):
        self.startCapture()
        t0 = time.perf_counter()
        stream_flag = False
        self.frame_count = 0
        while True:
            now = datetime.datetime.now()
            if (not stream_flag) & self.m_dict["stream"]:
                file_name_base = (
                    self.m_dict["export"] + "/" + self.m_dict["curLog"]
                )
                curVidName = (
                    file_name_base  + "_vid.avi"
                )
                self.vwriter = cv2.VideoWriter(
                    curVidName,
                    self.fmt,
                    self.m_dict["camera_fps"],
                    self.resized_resolution,
                    1,
                )
                # self.currentLogFile = open(curVidName + ".log", "a+")
                # self.currentLogFile.write(
                #     "# Streaming start: " + str(datetime.datetime.now()) + "\r"
                # )
                # self.currentLogFile.write("Date, Time" + "\r")

                self.LogFile = open(file_name_base + "_log.csv", "a+", newline='')
                self.log_writer = csv.writer(self.LogFile)
                self.log_writer.writerows([[
                    "frame",
                    "total_time",]])
                
                
                self.detectionlogfile = open(file_name_base + "_detect.csv", "a+", newline='')
                self.rtesult_writer = csv.writer(self.detectionlogfile)
                self.rtesult_writer.writerows([[
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "confidence",
                    "class",
                    "class_name",
                    "total_time",]])
                stream_flag = True
                print("Start: Video-streaming")

            elif stream_flag & (not self.m_dict["stream"]):
                self.frame_count = 0
                # Streaming Done.
                stream_flag = False
                self.vwriter.release()
                print(curVidName)
                # self.currentLogFile.close()
                self.detectionlogfile.close()
                self.LogFile.close()
                print("Finished: Video-streaming")

            # Ensure camera is connected
            if self.capture.isOpened():
                (status, frame) = self.capture.read()
                t1 = time.perf_counter()
                self.m_dict["total_time"] = t1 - self.m_dict["t0"]
                halfImg = cv2.resize(frame, self.resized_resolution)

                if status:
                    if self.m_dict["camera_imshow"]:
                        cv2.imshow("frame", halfImg)
                    if self.m_dict["stream"] & stream_flag:
                        self.vwriter.write(halfImg)
                        "# Date, total time, Count, Speed, Position, Dark, Z-stage, di"
                        # self.currentLogFile.write(
                        #     str(now) + ", " + str(self.m_dict["total_time"]) + "\r"
                        # )
                        self.log_writer.writerows([[self.frame_count, str(self.m_dict["total_time"])]])
                        if self.m_dict["yolo_process_state"]:
                            self.rtesult_writer.writerows(self.m_dict["yolo_results"])
                        self.frame_count += 1
                else:
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                elif self.m_dict["quit"]:
                    break
                self.m_dict["current_camera_frame"] = halfImg
            else:
                t1 = time.perf_counter()
            # while (t1-t0) < 1/(self.default_FPS+1.0):
            #     t1 = time.perf_counter()
            self.m_dict["camera_fps"] = int(1 / (t1 - t0))
            t0 = t1 * 1
            if self.m_dict["quit"]:
                break

    def run_Buffering(self):
        # TODO: faster buffering and exporting
        self.startCapture()
        k = 0
        self.frameTimeStamp = np.zeros((self.frameBufLen))
        while not self.m_dict["quit"]:
            self.frameTimeStamp[k % self.frameBufLen] = time.perf_counter() - self.t0
            (status, frame) = self.capture.read()
            self.m_dict["currentFrame"] = frame
            self.frameBuffer[:, :, :, k % self.frameBufLen] = frame
            self.fps = self.frameBufLen / (
                np.max(self.frameTimeStamp) - np.min(self.frameTimeStamp)
            )
            k = k + 1

    def __del__(self):
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()
        try:
            cv2.destroyAllWindows()
        except ImportError:
            pass


class capture_streamMSS:
    def __init__(self, m_dict={}):
        print("CV-initialization...")
        self.m_dict = m_dict
        self.initialized = False  # Add this line

        self.disp = m_dict["capture_area"]
        self.t0 = self.m_dict["t0"]
        self.resized_resolution = (
            int(self.m_dict["camera_width"] * self.m_dict["camera_scale"]),
            int(self.m_dict["camera_height"] * self.m_dict["camera_scale"]),
        )

        self.frameBufLen = 200
        self.frameBuffer = np.zeros(
            (self.resized_resolution[1], self.resized_resolution[0], self.frameBufLen),
            dtype="uint8",
        )
        self.fmt = cv2.VideoWriter_fourcc("D", "I", "V", "X")
        print("CV-initialization Finished")

    def startCapture(self):
        self.src = mss.mss()
        frame = np.array(self.src.grab(self.disp))
        halfImg = cv2.resize(frame, self.resized_resolution)
        print("CV-capture start success")

    def run(self):
        self.startCapture()
        t0 = time.perf_counter()
        stream_flag = False
        self.frame_count = 0
        while True:
            now = datetime.datetime.now()
            if (not stream_flag) & self.m_dict["stream"]:
                file_name_base = (
                    self.m_dict["export"] + "/" + self.m_dict["curLog"]
                )
                curVidName = (
                    file_name_base  + "_vid.avi"
                )

                self.vwriter = cv2.VideoWriter(
                    curVidName,
                    self.fmt,
                    self.m_dict["camera_fps"],
                    self.resized_resolution,
                    1,
                )
                # self.currentLogFile = open(curVidName + ".log", "a+")
                
                # self.currentLogFile.write(
                #     "# Streaming start: " + str(datetime.datetime.now()) + "\r"
                # )
                # self.currentLogFile.write("Date, Time" + "\r")

                self.LogFile = open(file_name_base + "_log.csv", "a+", newline='')
                self.log_writer = csv.writer(self.LogFile)
                self.log_writer.writerows([[
                    "frame",
                    "total_time",]])
                
                
                self.detectionlogfile = open(file_name_base + "_detect.csv", "a+", newline='')
                self.rtesult_writer = csv.writer(self.detectionlogfile)
                self.rtesult_writer.writerows([[
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "confidence",
                    "class",
                    "class_name",
                    "total_time",]])
                stream_flag = True
                print("Start: Video-streaming")
            elif stream_flag & (not self.m_dict["stream"]):
                self.frame_count = 0
                # Streaming Done.
                stream_flag = False
                self.vwriter.release()
                print(curVidName)
                # self.currentLogFile.close()
                self.detectionlogfile.close()
                self.LogFile.close()
                print("Finished: Video-streaming")

            # Ensure camera is connected
            if True:  # self.capture.isOpened():
                # (status, frame) = mss.capture.read()
                frame = np.array(self.src.grab(self.disp)) * 1
                t1 = time.perf_counter()
                self.m_dict["total_time"] = t1 - self.m_dict["t0"]
                halfImg = cv2.resize(frame, self.resized_resolution)

                if True:
                    if self.m_dict["camera_imshow"]:
                        cv2.imshow("frame", halfImg)
                    if self.m_dict["stream"] & stream_flag:
                        self.vwriter.write(halfImg)
                        "# Date, total time, Count, Speed, Position, Dark, Z-stage, di"
                        # self.currentLogFile.write(
                        #     str(now) + ", " + str(self.m_dict["total_time"]) + "\r"
                        # )
                        self.log_writer.writerows([[self.frame_count, str(self.m_dict["total_time"])]])
                        if self.m_dict["yolo_process_state"]:
                            self.rtesult_writer.writerows(self.m_dict["yolo_results"])
                        self.frame_count += 1

                else:
                    break
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                elif self.m_dict["quit"]:
                    break
                self.m_dict["current_camera_frame"] = halfImg
            else:
                t1 = time.perf_counter()
            while (t1 - t0) < 1 / 16:  # TODO
                t1 = time.perf_counter()
            self.m_dict["camera_fps"] = int(1 / (t1 - t0))
            t0 = t1 * 1

            if self.m_dict["quit"]:
                break

    def run_Buffering(self):
        # TODO: faster buffering and exporting
        self.startCapture()
        k = 0
        self.frameTimeStamp = np.zeros((self.frameBufLen))
        while not self.m_dict["quit"]:
            self.frameTimeStamp[k % self.frameBufLen] = time.perf_counter() - self.t0
            frame = self.src.grab(self.disp)
            self.m_dict["currentFrame"] = np.array(frame)
            self.frameBuffer[:, :, :, k % self.frameBufLen] = frame
            self.fps = self.frameBufLen / (
                np.max(self.frameTimeStamp) - np.min(self.frameTimeStamp)
            )
            k = k + 1

    def __del__(self):
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()
        try:
            cv2.destroyAllWindows()
        except ImportError:
            pass


class SelectCaptureArea:
    def __init__(self, root, opacity=0.5, m_dict={}):
        print("select area")
        self.m_dict = m_dict
        self.m_dict["capture_area"] = {}

        self.root = root
        self.color = (219, 77, 109)  # Added color parameter
        self.opacity = opacity
        self.canvas = tk.Canvas(
            root, width=root.winfo_screenwidth(), height=root.winfo_screenheight()
        )
        self.canvas.pack()

        self.start_x = None
        self.start_y = None
        self.rectangle = None

        self.root.attributes("-alpha", 0.2)  # Start fully transparent
        self.root.attributes("-fullscreen", True)  # Fullscreen
        self.root.update()  # Make sure the window is shown

        self.listener = mouse.Listener(on_click=self.on_click, on_move=self.on_move)
        self.listener.start()

    def draw_rectangle(self, start_x, start_y, end_x, end_y):
        if start_x > end_x:
            start_x, end_x = end_x, start_x
        if start_y > end_y:
            start_y, end_y = end_y, start_y
        image = Image.new(
            "RGBA",
            (end_x - start_x, end_y - start_y),
            (*self.color, int(255 * self.opacity)),
        )
        self.photo = ImageTk.PhotoImage(image)
        self.rectangle = self.canvas.create_image(
            start_x, start_y, image=self.photo, anchor="nw"
        )

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left:
            if pressed:
                self.start_x = x
                self.start_y = y
                self.top = y
                self.left = x
                self.root.attributes(
                    "-alpha", 0.2
                )  # Make visible when we start the drag
            else:
                self.canvas.delete(self.rectangle)
                if y > self.top and x > self.left:
                    self.width = x - self.left
                    self.height = y - self.top
                else:
                    self.width = self.left - x
                    self.height = self.top - y
                    self.top = y
                    self.left = x
                self.m_dict["capture_area"]["top"] = self.top
                self.m_dict["capture_area"]["left"] = self.left
                self.m_dict["capture_area"]["width"] = self.width
                self.m_dict["capture_area"]["height"] = self.height
                # self.m_dict["camera_width"] = self.width
                # self.m_dict["camera_height"] = self.height
                self.m_dict["camera_width"] = 640
                self.m_dict["camera_height"] = 480
                self.m_dict["capture_area"] = {
                    "top": self.top,
                    "left": self.left,
                    "width": self.width,
                    "height": self.height,
                }
                print(self.m_dict["capture_area"])
                self.root.quit()  # Close the window when we release the mouse button

    def on_move(self, x, y):
        if self.start_x is not None and self.start_y is not None:
            if self.rectangle is not None:
                self.canvas.delete(self.rectangle)
            self.draw_rectangle(self.start_x, self.start_y, x, y)


class select_run:
    def __init__(self, m_dict):
        self.m_dict = m_dict

    def main(self):
        # select capture area
        if self.m_dict["capture_area_select"]:  # Modify this line
            self.root = tk.Tk()
            self.area = SelectCaptureArea(self.root, m_dict=self.m_dict)
            self.root.mainloop()
            self.area.listener.stop()
            self.root.destroy()
            self.initialized = True  # Add this line


if __name__ == "__main__":
    d = {}
    d["t0"] = time.perf_counter()
    d["capture_area"] = {"top": 0, "left": 0, "width": 640, "height": 480}
    d["capture_area_select"] = True
    d["camera_id"] = 1
    d["camera_width"] = 1280
    d["camera_height"] = 960
    d["camera_scale"] = 1
    d["camera_fps"] = 20
    d["export"] = "test\\"
    d["curLog"] = "hoge.txt"
    d["camera_imshow"] = True
    d["stream"] = False
    d["quit"] = False
    d["stream_MSS"] = False
    if d["stream_MSS"]:
        SR = select_run(m_dict=d)
        SR.main()
        imgWin = capture_streamMSS(m_dict=d)

    else:
        imgWin = capture_streamCV2(m_dict=d)
    imgWin.run()
