# SPDX-License-Identifier: AGPL-3.0-or-later
"""Camera and screen acquisition with ordered, buffered recording."""

import sys
import time
import tkinter as tk
from pathlib import Path

import cv2
import mss
import numpy as np

from yoru.libs.recording import BufferedRecorder
from yoru.libs.realtime_state import clear_detection, fresh_results
from yoru.libs.user_paths import log_exception


class _CaptureStream:
    def __init__(self, m_dict=None):
        self.m_dict = m_dict if m_dict is not None else {}
        self.t0 = self.m_dict["t0"]
        self.default_FPS = max(1.0, float(self.m_dict["camera_fps"]))
        self.resized_resolution = (
            max(1, int(self.m_dict["camera_width"] * self.m_dict["camera_scale"])),
            max(1, int(self.m_dict["camera_height"] * self.m_dict["camera_scale"])),
        )
        self.recorder = None

    def _recording(self, frame, timestamp):
        if self.m_dict.get("stream", False):
            base = Path(self.m_dict["export"]) / self.m_dict["curLog"]
            if self.recorder is not None and self.recorder.base_path != str(base):
                recorder, self.recorder = self.recorder, None
                recorder.close()
            if self.recorder is None:
                self.recorder = BufferedRecorder(
                    base, self.default_FPS, self.resized_resolution,
                    self.m_dict.get("config_path"),
                )
            self.recorder.write(frame, timestamp, fresh_results(self.m_dict))
        elif self.recorder is not None:
            recorder, self.recorder = self.recorder, None
            recorder.close()

    def run(self):
        previous = time.perf_counter()
        frame_id = 0
        try:
            self.startCapture()
            self.m_dict["capture_running"] = True
            while not self.m_dict.get("quit", False):
                loop_start = time.perf_counter()
                frame = self._read()
                if frame is None:
                    raise RuntimeError("Capture returned no frame; check the camera connection")
                captured_at = time.perf_counter()
                frame = cv2.resize(frame, self.resized_resolution)
                elapsed = captured_at - self.t0
                frame_id += 1
                # A single manager assignment keeps image and acquisition time paired.
                self.m_dict["camera_snapshot"] = (frame_id, captured_at, frame)
                self.m_dict["camera_frame_id"] = frame_id
                self.m_dict["current_camera_frame"] = frame
                self.m_dict["total_time"] = elapsed
                interval = captured_at - previous
                self.m_dict["camera_fps"] = 1.0 / interval if interval > 0 else 0.0
                previous = captured_at
                self._recording(frame, elapsed)
                if self.m_dict.get("camera_imshow", False):
                    cv2.imshow("frame", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                self._pace(loop_start)
        except Exception as exc:
            log_exception("Capture failed", exc)
            self.m_dict["capture_error"] = str(exc)
            raise
        finally:
            self.m_dict["capture_running"] = False
            self.m_dict["quit"] = True
            self.m_dict["stream"] = False
            clear_detection(self.m_dict)
            try:
                if self.recorder is not None:
                    recorder, self.recorder = self.recorder, None
                    recorder.close()
            finally:
                self._release()
                if self.m_dict.get("camera_imshow", False):
                    cv2.destroyAllWindows()

    def run_Buffering(self):
        """Compatibility entry point: all capture now uses bounded recording."""
        self.run()

    def _pace(self, loop_start):
        pass


class capture_streamCV2(_CaptureStream):
    def __init__(self, srcCam=1, m_dict=None):
        super().__init__(m_dict)
        self.src = srcCam
        self.capture = None

    def startCapture(self):
        preferred = {"win32": cv2.CAP_DSHOW, "darwin": cv2.CAP_AVFOUNDATION,
                     "linux": cv2.CAP_V4L2}.get(sys.platform, cv2.CAP_ANY)
        for backend in dict.fromkeys((preferred, cv2.CAP_ANY)):
            self.capture = cv2.VideoCapture(self.src, backend)
            if self.capture.isOpened():
                break
            self.capture.release()
            self.capture = None
        if self.capture is None:
            raise RuntimeError(f"Could not open camera id {self.src}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.m_dict["camera_width"])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.m_dict["camera_height"])
        self.capture.set(cv2.CAP_PROP_FPS, self.default_FPS)
        # Avoid a long queue of old camera frames in a closed-loop experiment.
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if sys.platform == "win32" and self.m_dict.get("camera_settings_dialog", False):
            self.capture.set(cv2.CAP_PROP_SETTINGS, 1)

    def _read(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def _release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class capture_streamMSS(_CaptureStream):
    def __init__(self, m_dict=None):
        super().__init__(m_dict)
        self.disp = self.m_dict["capture_area"]
        self.src = None

    @staticmethod
    def _to_bgr(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame

    def startCapture(self):
        if self.disp["width"] <= 0 or self.disp["height"] <= 0:
            raise ValueError("Select a screen area with positive width and height")
        self.src = mss.mss()

    def _read(self):
        return self._to_bgr(np.asarray(self.src.grab(self.disp), dtype=np.uint8))

    def _release(self):
        if self.src is not None:
            self.src.close()
            self.src = None

    def _pace(self, loop_start):
        # Honour the configured rate without burning a CPU core between frames.
        remaining = 1.0 / self.default_FPS - (time.perf_counter() - loop_start)
        deadline = time.perf_counter() + max(0.0, remaining)
        while remaining > 0 and not self.m_dict.get("quit", False):
            time.sleep(min(remaining, 0.05))
            remaining = deadline - time.perf_counter()


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

        # Tk operations stay on Tk's thread; a global mouse listener used to
        # modify widgets from its own thread.
        self.canvas.bind("<ButtonPress-1>", lambda e: self.on_click(e.x_root, e.y_root, "left", True))
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.on_click(e.x_root, e.y_root, "left", False))
        self.canvas.bind("<B1-Motion>", lambda e: self.on_move(e.x_root, e.y_root))
        self.selected = False
        self.root.bind("<Escape>", lambda e: self.root.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)

    def draw_rectangle(self, start_x, start_y, end_x, end_y):
        if start_x > end_x:
            start_x, end_x = end_x, start_x
        if start_y > end_y:
            start_y, end_y = end_y, start_y
        ox, oy = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
        self.rectangle = self.canvas.create_rectangle(
            start_x - ox, start_y - oy, end_x - ox, end_y - oy,
            outline="#db4d6d", fill="#db4d6d", stipple="gray50",
        )

    def on_click(self, x, y, button, pressed):
        if button == "left":
            if pressed:
                self.start_x = x
                self.start_y = y
                self.top = y
                self.left = x
                self.root.attributes(
                    "-alpha", 0.2
                )  # Make visible when we start the drag
            else:
                if self.start_x is None or self.start_y is None:
                    return
                self.canvas.delete(self.rectangle)
                self.left, self.top = min(self.start_x, x), min(self.start_y, y)
                self.width, self.height = abs(x - self.start_x), abs(y - self.start_y)
                self.start_x = self.start_y = None
                if self.width == 0 or self.height == 0:
                    return
                self.m_dict["capture_area"] = {
                    "top": self.top,
                    "left": self.left,
                    "width": self.width,
                    "height": self.height,
                }
                print(self.m_dict["capture_area"])
                self.selected = True
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
            try:
                self.area = SelectCaptureArea(self.root, m_dict=self.m_dict)
                self.root.mainloop()
                self.m_dict["quit"] = not self.area.selected
            finally:
                self.root.destroy()


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
