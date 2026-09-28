"""Bounded, ordered video/CSV writing outside the capture loop."""

import csv
import queue
import shutil
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import cv2

from yoru.libs.detector_base import DETECTION_COLUMNS
from yoru.libs.user_paths import log_exception


class BufferedRecorder:
    """Keep video and timestamps paired; apply backpressure instead of dropping.

    Only the writer thread owns the files and codec. The queue holds at most
    64 frames / roughly 64 MiB (or one larger frame), allocated on demand.
    Worker errors propagate to the capture process, including on close.
    """

    def __init__(self, base_path, fps, resolution, config_path=None):
        self.base_path = str(base_path)
        self.fps = fps
        self.resolution = resolution
        self.config_path = config_path
        frame_bytes = max(1, resolution[0] * resolution[1] * 3)
        self.queue = queue.Queue(maxsize=max(1, min(64, 64 * 1024**2 // frame_bytes)))
        self.error = None
        self.closed = False
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name="yoru-recorder", daemon=True)
        self.thread.start()
        self.ready.wait()
        self.check()

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Recording failed: {self.base_path}") from self.error

    def _put(self, value):
        while True:
            self.check()
            try:
                self.queue.put(value, timeout=0.1)
                return
            except queue.Full:
                continue

    def write(self, frame, timestamp, results):
        if self.closed:
            raise RuntimeError("Recording already closed")
        self._put((frame.copy(), timestamp, [list(row) for row in results]))

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                self._put(None)
            finally:
                self.thread.join()
        self.check()

    def _run(self):
        try:
            with ExitStack() as stack:
                Path(self.base_path).parent.mkdir(parents=True, exist_ok=True)
                video = cv2.VideoWriter(
                    self.base_path + "_vid.avi", cv2.VideoWriter_fourcc(*"DIVX"),
                    self.fps, self.resolution, True,
                )
                stack.callback(video.release)
                if not video.isOpened():
                    raise RuntimeError("Could not open video writer (path or codec unavailable)")
                timing_file = stack.enter_context(open(self.base_path + "_log.csv", "w", newline=""))
                detection_file = stack.enter_context(open(self.base_path + "_detect.csv", "w", newline=""))
                timing, detection = csv.writer(timing_file), csv.writer(detection_file)
                timing.writerow(["frame", "total_time"])
                detection.writerow(DETECTION_COLUMNS)
                if self.config_path:
                    shutil.copyfile(self.config_path, self.base_path + ".yaml")
                self.ready.set()
                frame_id, last_flush = 0, time.monotonic()
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    frame, timestamp, rows = item
                    video.write(frame)
                    timing.writerow([frame_id, timestamp])
                    detection.writerows(rows)
                    frame_id += 1
                    if time.monotonic() - last_flush >= 1.0:
                        timing_file.flush()
                        detection_file.flush()
                        last_flush = time.monotonic()
        except Exception as exc:
            self.error = exc
            log_exception(f"Recording failed: {self.base_path}", exc)
        finally:
            self.ready.set()
