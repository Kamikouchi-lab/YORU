"""The Frame Capture GUI's half of automatic extraction.

The work happens on a thread, so the parts that can go wrong are not the
algorithms -- those are covered in ``test_frame_extraction.py`` -- but the
handover: the buttons have to say whether a run is going, the progress the
worker writes has to reach the widgets, Stop has to be heard, and the frame
counter has to end up telling the truth.  DearPyGui is replaced by a recorder
so all of that can be driven without a window.
"""

import time
import types

import pytest

from yoru.libs import frame_extraction


class _FakeDPG(types.ModuleType):
    """Stands in for dearpygui: remembers values and item configuration."""

    def __init__(self):
        super().__init__("dearpygui.dearpygui")
        self.values = {
            "extract_count": 8,
            "extract_algo": "uniform",
            "extract_start": 0.0,
            "extract_stop": 1.0,
            "save_name": "fly",
            "extract_status": "",
            "extract_progress": 0.0,
            "count_frames": "0 frames grabbed",
        }
        self.config = {}

    def set_value(self, tag, value):
        self.values[tag] = value

    def get_value(self, tag):
        return self.values.get(tag)

    def configure_item(self, tag, **kwargs):
        self.config.setdefault(tag, {}).update(kwargs)

    def does_item_exist(self, tag):
        return tag in self.values

    def is_item_active(self, tag):
        return False

    def enabled(self, tag):
        return self.config.get(tag, {}).get("enabled")


@pytest.fixture
def gui(monkeypatch, repo_root):
    """A grab_gui with a recorder in place of DearPyGui, and a video loaded."""
    monkeypatch.chdir(repo_root)  # __init__ reads the logo through a relative path
    from yoru import grab_GUI

    fake = _FakeDPG()
    monkeypatch.setattr(grab_GUI, "dpg", fake)

    g = grab_GUI.grab_gui({"quit": False})
    g.has_video = True
    g.file_path = "video.mp4"
    g.grab_dir = "out"
    return g, fake


def _pump(g, until, timeout=5.0):
    """Run the render loop's share of the work until ``until`` or a timeout."""
    end = time.time() + timeout
    while time.time() < end:
        g._pump_extraction()
        if until():
            return True
        time.sleep(0.01)
    g._pump_extraction()
    return until()


# ---------------------------------------------------------------------------
# Refusals, before a thread is ever started
# ---------------------------------------------------------------------------


def test_extraction_without_a_video_says_so_and_starts_nothing(gui):
    g, fake = gui
    g.has_video = False
    g.extract_btn_cb()
    assert "video" in fake.values["extract_status"].lower()
    assert g._extract_thread is None


def test_extraction_without_a_save_directory_says_so(gui):
    g, fake = gui
    g.grab_dir = ""
    g.extract_btn_cb()
    assert "directory" in fake.values["extract_status"].lower()
    assert g._extract_thread is None


def test_a_backwards_range_is_refused_before_the_video_is_touched(gui, monkeypatch):
    g, fake = gui
    started = []
    monkeypatch.setattr(
        frame_extraction, "extract_frames", lambda *a, **k: started.append(a)
    )
    fake.values["extract_start"] = 0.9
    fake.values["extract_stop"] = 0.1
    g.extract_btn_cb()
    assert "start < stop" in fake.values["extract_status"]
    assert not started


def test_an_empty_frame_name_falls_back_to_the_video_name(gui, monkeypatch):
    """Otherwise the automatic mode is unusable without also typing a name."""
    g, fake = gui
    seen = {}

    def stub(video, out_dir, name, **kw):
        seen["name"] = name
        return frame_extraction.ExtractionResult(True, "done", (), ())

    monkeypatch.setattr(frame_extraction, "extract_frames", stub)
    fake.values["save_name"] = ""
    g.file_path = r"C:\videos\arena7.mp4"
    g.extract_btn_cb()
    _pump(g, lambda: not g._extract_state["running"])
    assert seen["name"] == "arena7"


# ---------------------------------------------------------------------------
# A run from start to finish
# ---------------------------------------------------------------------------


def test_a_run_reports_progress_and_hands_the_buttons_back(gui, monkeypatch):
    g, fake = gui

    def stub(video, out_dir, name, count=0, progress=None, should_stop=None, **kw):
        for i in range(1, 5):
            progress(i, 4, "scanning")
        for i in range(1, count + 1):
            progress(i, count, "saving")
        paths = tuple(f"{out_dir}/{name}_{i}.png" for i in range(count))
        return frame_extraction.ExtractionResult(True, f"Saved {count} frame(s)", (), paths)

    monkeypatch.setattr(frame_extraction, "extract_frames", stub)

    g.extract_btn_cb()
    assert fake.enabled("extract_btn") is False, "Extract must not be clickable twice"
    assert fake.enabled("extract_stop_btn") is True

    assert _pump(g, lambda: g._extract_state["applied"])
    assert fake.values["extract_progress"] == 1.0
    assert fake.values["extract_status"] == "Saved 8 frame(s)"
    assert fake.enabled("extract_btn") is True
    assert fake.enabled("extract_stop_btn") is False
    assert fake.values["count_frames"] == "8 frames grabbed"


def test_the_counter_adds_automatic_frames_to_the_ones_grabbed_by_hand(gui, monkeypatch):
    g, fake = gui
    g.grab_count = 3
    monkeypatch.setattr(
        frame_extraction,
        "extract_frames",
        lambda *a, **k: frame_extraction.ExtractionResult(True, "ok", (), ("a", "b")),
    )
    g.extract_btn_cb()
    assert _pump(g, lambda: g._extract_state["applied"])
    assert fake.values["count_frames"] == "5 frames grabbed"


def test_a_failure_in_the_worker_reaches_the_status_line(gui, monkeypatch):
    """A thread that dies quietly would leave the buttons stuck for good."""
    g, fake = gui

    def boom(*a, **k):
        raise RuntimeError("codec exploded")

    monkeypatch.setattr(frame_extraction, "extract_frames", boom)
    g.extract_btn_cb()
    assert _pump(g, lambda: g._extract_state["applied"])
    assert "codec exploded" in fake.values["extract_status"]
    assert fake.enabled("extract_btn") is True
    assert fake.enabled("extract_stop_btn") is False


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------


def test_stop_is_heard_by_the_worker_and_reported(gui, monkeypatch):
    g, fake = gui

    def stub(video, out_dir, name, count=0, progress=None, should_stop=None, **kw):
        for i in range(10000):
            if should_stop():
                return frame_extraction.ExtractionResult(
                    False, "Stopped after saving 2 frame(s).", (), ("a", "b"),
                    cancelled=True,
                )
            progress(i, 10000, "saving")
            time.sleep(0.001)
        return frame_extraction.ExtractionResult(True, "finished", (), ())

    monkeypatch.setattr(frame_extraction, "extract_frames", stub)
    g.extract_btn_cb()
    assert _pump(g, lambda: g._extract_state["fraction"] > 0)

    g.extract_stop_cb()
    assert _pump(g, lambda: g._extract_state["applied"])
    assert "Stopped" in fake.values["extract_status"]
    assert fake.enabled("extract_btn") is True
    assert fake.values["count_frames"] == "2 frames grabbed"


def test_a_second_extract_click_while_one_runs_is_ignored(gui, monkeypatch):
    g, fake = gui
    runs = []

    def stub(video, out_dir, name, count=0, progress=None, should_stop=None, **kw):
        runs.append(1)
        while not should_stop():
            time.sleep(0.005)
        return frame_extraction.ExtractionResult(False, "Extraction stopped.", (), (),
                                                 cancelled=True)

    monkeypatch.setattr(frame_extraction, "extract_frames", stub)
    g.extract_btn_cb()
    _pump(g, lambda: bool(runs))
    g.extract_btn_cb()
    g.extract_stop_cb()
    assert _pump(g, lambda: g._extract_state["applied"])
    assert len(runs) == 1


# ---------------------------------------------------------------------------
# The rest of the window, while extraction is somebody else's problem
# ---------------------------------------------------------------------------


def test_the_preview_capture_is_never_handed_to_the_worker(gui, monkeypatch):
    """Two threads seeking one VideoCapture garble frames or crash outright."""
    g, fake = gui
    seen = {}

    def stub(video, out_dir, name, **kw):
        seen["video"] = video
        seen["capture_kw"] = [k for k in kw if "capture" in k]
        return frame_extraction.ExtractionResult(True, "ok", (), ())

    monkeypatch.setattr(frame_extraction, "extract_frames", stub)
    g.file_path = "some/movie.mp4"
    g.extract_btn_cb()
    assert _pump(g, lambda: g._extract_state["applied"])
    assert seen["video"] == "some/movie.mp4"
    assert seen["capture_kw"] == []


def test_the_arrow_keys_leave_the_video_alone_while_a_name_is_being_typed(gui, monkeypatch):
    g, fake = gui
    monkeypatch.setattr(fake, "is_item_active", lambda tag: tag == "save_name")
    g.framecount = 500
    g.current_frame_num = 10
    g.advance_frame_bt()
    g.reverse_frame_bt()
    assert g.current_frame_num == 10
