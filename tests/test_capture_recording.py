import csv
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from yoru.libs import imager, recording


def state():
    return dict(t0=0, camera_fps=30, camera_width=8, camera_height=6,
                camera_scale=1, quit=False, stream=True, export="unused", curLog="run",
                camera_imshow=False, capture_area=dict(left=0, top=0, width=8, height=6))


class FakeVideo:
    def __init__(self, opened=True, fail=False):
        self.frames = []
        self.opened, self.fail, self.released = opened, fail, False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        if self.fail:
            raise OSError("disk write failed")
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


def test_buffered_recording_drains_frames_and_closes_csv(tmp_path, monkeypatch):
    video = FakeVideo()
    monkeypatch.setattr(recording.cv2, "VideoWriter", lambda *a: video)
    config = tmp_path / "condition.yaml"
    config.write_text("test: true\n")
    writer = recording.BufferedRecorder(tmp_path / "run", 30, (8, 6), config)
    frame = np.zeros((6, 8, 3), np.uint8)
    for i in range(20):
        frame[:] = i
        writer.write(frame, i / 30, [])
    writer.close()
    writer.close()
    assert video.released
    assert [int(f[0, 0, 0]) for f in video.frames] == list(range(20))
    with (tmp_path / "run_log.csv").open(newline="") as f:
        rows = list(csv.reader(f))
    assert [int(r[0]) for r in rows[1:]] == list(range(20))
    assert float(rows[-1][1]) == pytest.approx(19/30)
    assert (tmp_path / "run.yaml").read_text() == config.read_text()
    assert len((tmp_path / "run_detect.csv").read_text().splitlines()) == 1


def test_capture_can_enqueue_while_codec_is_busy(tmp_path, monkeypatch):
    video = FakeVideo()
    entered, release = threading.Event(), threading.Event()
    def slow_write(frame):
        entered.set()
        assert release.wait(3)
        video.frames.append(frame)
    video.write = slow_write
    monkeypatch.setattr(recording.cv2, "VideoWriter", lambda *a: video)
    writer = recording.BufferedRecorder(tmp_path / "run", 30, (8, 6))
    try:
        writer.write(np.zeros((6, 8, 3), np.uint8), 0, [])
        assert entered.wait(3)
        writer.write(np.zeros((6, 8, 3), np.uint8), 1, [])
        assert writer.queue.qsize() == 1
        assert writer.queue.maxsize <= 64
    finally:
        release.set()
        writer.close()
    assert len(video.frames) == 2


def test_codec_open_failure_is_reported_and_released(tmp_path, monkeypatch):
    video = FakeVideo(opened=False)
    monkeypatch.setattr(recording.cv2, "VideoWriter", lambda *a: video)
    with pytest.raises(RuntimeError, match="Recording failed"):
        recording.BufferedRecorder(tmp_path / "run", 30, (8, 6))
    assert video.released


def test_write_failure_surfaces_on_close_without_hanging(tmp_path, monkeypatch):
    video = FakeVideo(fail=True)
    monkeypatch.setattr(recording.cv2, "VideoWriter", lambda *a: video)
    writer = recording.BufferedRecorder(tmp_path / "run", 30, (8, 6))
    writer.write(np.zeros((6, 8, 3), np.uint8), 0, [])
    with pytest.raises(RuntimeError, match="Recording failed"):
        writer.close()
    assert video.released and not writer.thread.is_alive()


@pytest.mark.parametrize("disconnect", [False, True])
def test_capture_finalizes_recording_on_quit_and_disconnect(tmp_path, monkeypatch, disconnect):
    d = state()
    d["export"] = str(tmp_path)
    capture = imager.capture_streamCV2(m_dict=d)
    video = FakeVideo()
    monkeypatch.setattr(recording.cv2, "VideoWriter", lambda *a: video)
    monkeypatch.setattr(capture, "startCapture", lambda: None)
    released = []
    monkeypatch.setattr(capture, "_release", lambda: released.append(True))
    calls = []
    def read():
        calls.append(1)
        if len(calls) == 2:
            if disconnect:
                return None
            d["quit"] = True
        return np.zeros((6, 8, 3), np.uint8)
    monkeypatch.setattr(capture, "_read", read)
    if disconnect:
        with pytest.raises(RuntimeError, match="no frame"):
            capture.run()
    else:
        capture.run()
    assert video.released and released
    assert d["quit"] and not d["capture_running"]
    assert d["detection_snapshot"] is None
    expected = 1 if disconnect else 2
    assert len(video.frames) == expected
    assert len((tmp_path / "run_log.csv").read_text().splitlines()) == expected + 1


@pytest.mark.parametrize("platform,backend", [("win32", imager.cv2.CAP_DSHOW),
    ("darwin", imager.cv2.CAP_AVFOUNDATION), ("linux", imager.cv2.CAP_V4L2)])
def test_camera_backend_selection_and_fallback(monkeypatch, platform, backend):
    monkeypatch.setattr(imager, "sys", SimpleNamespace(platform=platform))
    calls, released = [], []
    def open_camera(index, api):
        calls.append((index, api))
        return SimpleNamespace(isOpened=lambda: api == imager.cv2.CAP_ANY,
                               release=lambda: released.append(api), set=lambda *a: None)
    monkeypatch.setattr(imager.cv2, "VideoCapture", open_camera)
    capture = imager.capture_streamCV2(srcCam=2, m_dict=state())
    capture.startCapture()
    assert calls == [(2, backend), (2, imager.cv2.CAP_ANY)]
    capture._release()
    assert released == [backend, imager.cv2.CAP_ANY]


def test_screen_capture_paces_at_configured_rate_and_releases(monkeypatch):
    capture = imager.capture_streamMSS(state())
    clock, sleeps, closed = [0.0], [], []
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(imager, "time", SimpleNamespace(perf_counter=lambda: clock[0], sleep=sleep))
    capture._pace(0)
    assert sum(sleeps) == pytest.approx(1/30)
    capture.src = SimpleNamespace(close=lambda: closed.append(True))
    capture._release()
    assert closed == [True]
    assert not hasattr(capture, "frameBuffer")


@pytest.mark.parametrize("end", [(150, 150), (50, 50), (150, 50), (50, 150)])
def test_screen_selection_in_all_four_directions(end):
    obj = imager.SelectCaptureArea.__new__(imager.SelectCaptureArea)
    obj.m_dict = {}
    obj.rectangle = None
    obj.canvas = SimpleNamespace(delete=lambda *a: None)
    obj.root = SimpleNamespace(attributes=lambda *a: None, quit=lambda: None)
    obj.on_click(100, 100, "left", True)
    obj.on_click(*end, "left", False)
    assert obj.m_dict["capture_area"] == dict(left=min(100, end[0]), top=min(100, end[1]), width=50, height=50)


def test_empty_screen_selection_keeps_selection_open():
    obj = imager.SelectCaptureArea.__new__(imager.SelectCaptureArea)
    obj.m_dict, obj.rectangle, obj.selected = {}, None, False
    obj.canvas = SimpleNamespace(delete=lambda *a: None)
    obj.root = SimpleNamespace(attributes=lambda *a: None, quit=lambda: pytest.fail("empty selection accepted"))
    obj.on_click(100, 100, "left", True)
    obj.on_click(100, 100, "left", False)
    assert not obj.selected and "capture_area" not in obj.m_dict


def test_recorded_avi_is_readable_after_close(tmp_path):
    writer = recording.BufferedRecorder(tmp_path / "real", 30, (64, 48))
    try:
        for i in range(12):
            writer.write(np.full((48, 64, 3), i * 10, np.uint8), i/30, [])
    finally:
        writer.close()
    reader = recording.cv2.VideoCapture(str(tmp_path / "real_vid.avi"))
    frames = []
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        reader.release()
    assert len(frames) == 12
    assert frames[-1].shape == (48, 64, 3)
    assert float(frames[-1].mean()) == pytest.approx(110, abs=10)


def test_quick_restart_opens_a_new_recording(tmp_path, monkeypatch):
    videos = []
    def new_video(*a):
        videos.append(FakeVideo())
        return videos[-1]
    monkeypatch.setattr(recording.cv2, "VideoWriter", new_video)
    d = state()
    d["export"] = str(tmp_path)
    capture = imager.capture_streamCV2(m_dict=d)
    frame = np.zeros((6, 8, 3), np.uint8)
    capture._recording(frame, 0)
    d["curLog"] = "second"
    capture._recording(frame, 1)
    d["stream"] = False
    capture._recording(frame, 2)
    assert [len(v.frames) for v in videos] == [1, 1]
    assert all(v.released for v in videos)
