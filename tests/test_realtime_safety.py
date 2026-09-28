import threading
import time
import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest

from yoru.libs.realtime_state import fresh_results
from yoru.libs import detection
from yoru.libs.realtime_process import run_workers


def state():
    now = time.perf_counter()
    rows = [[0, 0, 10, 10, .95, 0, "fly", 0]]
    return dict(quit=False, yolo_detection=True, yolo_process_state=True,
                capture_running=True, detection_generation=2, result_max_age=1,
                detection_snapshot=(1, now, now, 2, rows), yolo_results=rows,
                camera_snapshot=(1, now, np.zeros((2, 2, 3), np.uint8)),
                yolo_model="fake", t0=now, Trigger=True)


@pytest.mark.parametrize("key,value", [("quit", True), ("yolo_detection", False),
    ("yolo_process_state", False), ("capture_running", False), ("detection_generation", 3)])
def test_old_results_cannot_trigger_after_stop_or_reload(key, value):
    d = state()
    d[key] = value
    assert fresh_results(d) == []


def test_expiry_uses_capture_time_not_drawing_or_inference_time():
    d = state()
    stamp = d["detection_snapshot"][1]
    d["now"] = stamp + 100  # Drawing used to keep this timestamp fresh.
    assert fresh_results(d, stamp + .5)
    assert fresh_results(d, stamp + 1.01) == []


def test_plugin_receives_no_old_names_or_rows(fake_serial):
    from yoru.libs.trigger import trigger_python
    obj = trigger_python.__new__(trigger_python)
    obj.m_dict = state()
    obj.tri_class, obj.tri_th_conf, obj.myArduino = "fly", .5, None
    calls = []
    obj.trigger_instance = SimpleNamespace(trigger=lambda *a: calls.append(a))
    obj.trigger()
    assert calls[-1][1] == ["fly"]
    obj.m_dict["yolo_detection"] = False
    obj.trigger()
    assert calls[-1][1] == []
    assert calls[-1][3] == []


def test_trigger_cleanup_continues_after_output_error(fake_serial):
    from yoru.libs.trigger import trigger_python
    obj = trigger_python.__new__(trigger_python)
    events = []
    def fail(*a):
        raise OSError("device unplugged")
    obj.trigger_instance = SimpleNamespace(trigger=fail, close=lambda: events.append("plugin closed"))
    obj.tri_class = "fly"
    obj.myArduino = SimpleNamespace(writeDO_all=lambda v: events.append(v), close=lambda: events.append("board closed"))
    obj.close()
    obj.close()
    assert events == ["plugin closed", 0, "board closed"]


def test_stop_during_inference_does_not_publish_old_result(monkeypatch):
    d = state()
    def infer(image):
        d["yolo_detection"] = False
        d["quit"] = True
        return []
    monkeypatch.setattr(detection, "get_detector", lambda *a: SimpleNamespace(names={}, detect=infer))
    detection.yolo_detection(d).detect(d)
    assert d["detection_snapshot"] is None
    assert d["yolo_results"] == []


def test_inference_processes_each_captured_frame_once(monkeypatch):
    d, calls = state(), []
    ready = threading.Event()
    def infer(image):
        calls.append(1)
        ready.set()
        return []
    monkeypatch.setattr(detection, "get_detector", lambda *a: SimpleNamespace(names={}, detect=infer))
    worker = threading.Thread(target=detection.yolo_detection(d).detect, args=(d,))
    worker.start()
    try:
        assert ready.wait(3)
        time.sleep(.03)
        assert len(calls) == 1
    finally:
        d["quit"] = True
        worker.join(3)
    assert not worker.is_alive()


class FakeProcess:
    def __init__(self, name, state, exitcode=None, stuck=False):
        self.name, self.state, self.exitcode, self.stuck = name, state, exitcode, stuck
        self.started = self.terminated = False

    def start(self):
        self.started = True

    def join(self, timeout):
        if not self.stuck and self.state.get("quit"):
            self.exitcode = self.exitcode or 0

    def is_alive(self):
        return self.exitcode is None

    def terminate(self):
        self.terminated = True
        self.exitcode = -1


def test_worker_exit_stops_siblings_and_allows_recording_to_drain():
    d = state()
    processes = [FakeProcess("GUI", d, exitcode=0), FakeProcess("capture", d)]
    run_workers(processes, d, shutdown_timeout=.01)
    assert d["quit"] and not d["Trigger"] and not d["stream"]
    assert all(p.exitcode == 0 and not p.terminated for p in processes)


def test_hung_driver_is_terminated_and_reported():
    d = state()
    processes = [FakeProcess("GUI", d, exitcode=0), FakeProcess("capture", d, stuck=True)]
    with pytest.raises(RuntimeError, match="forced shutdown: capture"):
        run_workers(processes, d, shutdown_timeout=0)
    assert processes[1].terminated


def test_native_gui_close_signals_shutdown(monkeypatch):
    from yoru import realtime_yoru_GUI as gui
    obj = gui.camGUI.__new__(gui.camGUI)
    obj.m_dict = state()
    events = []
    obj.startDPG = lambda: None
    obj.session = SimpleNamespace(save=lambda: events.append("saved"))
    monkeypatch.setattr(gui.dpg, "is_dearpygui_running", lambda: False)
    monkeypatch.setattr(gui.dpg, "destroy_context", lambda: events.append("destroyed"))
    obj.run()
    assert obj.m_dict["quit"]
    assert events == ["saved", "destroyed"]


def _cooperative_worker(shared, request_stop):
    if request_stop:
        shared["quit"] = True
    while not shared.get("quit", False):
        time.sleep(.01)
    shared["closed_" + str(request_stop)] = True


def test_spawned_workers_shutdown_before_manager_exits():
    context = multiprocessing.get_context("spawn")
    with context.Manager() as manager:
        shared = manager.dict(quit=False)
        processes = [context.Process(target=_cooperative_worker, args=(shared, value))
                     for value in (False, True)]
        run_workers(processes, shared, shutdown_timeout=10)
        assert shared["closed_True"] and shared["closed_False"]
        assert all(p.exitcode == 0 and not p.is_alive() for p in processes)
