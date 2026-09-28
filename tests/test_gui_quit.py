"""Exercise each GUI's real Quit handler through its owning render loop."""

import importlib
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest


GUIS = [
    ("analysis_GUI", "analyze_GUI", "startDPG"),
    ("train_GUI", "yoru_train", "startDPG"),
    ("evaluation_GUI", "model_eval_gui", "gui_configure"),
    ("create_labels_GUI", "model_eval_gui", "gui_configure"),
    ("grab_GUI", "grab_gui", "gui_configure"),
    ("refine_GUI", "grab_gui", "gui_configure"),
    ("config_creator_GUI", "ConfigCreatorGUI", "startDPG"),
    ("realtime_yoru_GUI", "camGUI", "startDPG"),
]


@pytest.mark.parametrize("module_name,class_name,setup", GUIS)
@pytest.mark.parametrize("exit_mode", ["quit", "native_close", "exception"])
def test_every_gui_closes_once_after_rendering(monkeypatch, module_name, class_name, setup, exit_mode):
    module = importlib.import_module("yoru." + module_name)
    gui = getattr(module, class_name).__new__(getattr(module, class_name))
    gui.m_dict = {"quit": False}
    gui._extract_thread = None
    gui._class_result = None
    events = []
    gui.session = SimpleNamespace(save=lambda: events.append("save"))
    gui.vid = SimpleNamespace(release=lambda: events.append("release"))
    setattr(gui, setup, lambda: events.append("setup"))
    gui.plot_callback = lambda: None

    def render():
        events.append("render start")
        if exit_mode == "exception":
            raise RuntimeError("render failed")
        gui.quit_cb()
        gui.quit_cb()  # A queued second click must be harmless.
        assert "destroy" not in events
        events.append("render end")

    monkeypatch.setattr(module.dpg, "is_dearpygui_running", lambda: exit_mode != "native_close")
    monkeypatch.setattr(module.dpg, "render_dearpygui_frame", render)
    monkeypatch.setattr(module.dpg, "destroy_context", lambda: events.append("destroy"))
    if exit_mode == "exception":
        with pytest.raises(RuntimeError, match="render failed"):
            gui.run()
    else:
        gui.run()
    assert gui.m_dict["quit"]
    assert events[-3:] == ["release", "save", "destroy"]
    assert events.count("destroy") == 1
    if module_name == "train_GUI":
        assert gui._gpu_stop
    if module_name == "grab_GUI":
        assert gui._extract_stop
    if module_name == "realtime_yoru_GUI":
        assert not gui.m_dict["Trigger"] and not gui.m_dict["stream"]


def test_cleanup_failure_does_not_skip_remaining_cleanup(monkeypatch):
    from yoru.gui_lifecycle import run_gui
    events = []
    def fail():
        raise OSError("capture was disconnected")
    gui = SimpleNamespace(m_dict={}, vid=SimpleNamespace(release=fail),
                          session=SimpleNamespace(save=lambda: events.append("saved")))
    dpg = SimpleNamespace(is_dearpygui_running=lambda: False,
                          destroy_context=lambda: events.append("destroyed"))
    run_gui(gui, dpg, lambda: None)
    assert events == ["saved", "destroyed"]


def test_home_opens_after_the_old_window_is_destroyed(monkeypatch):
    from yoru import evaluation_GUI, app
    gui = evaluation_GUI.model_eval_gui.__new__(evaluation_GUI.model_eval_gui)
    gui.m_dict = {}
    events = []
    gui.gui_configure = lambda: None
    gui.plot_callback = lambda: None
    monkeypatch.setattr(evaluation_GUI.dpg, "is_dearpygui_running", lambda: True)
    monkeypatch.setattr(evaluation_GUI.dpg, "render_dearpygui_frame", gui.home_cb)
    monkeypatch.setattr(evaluation_GUI.dpg, "destroy_context", lambda: events.append("destroy"))
    monkeypatch.setattr(app, "main", lambda: events.append("home"))
    gui.run()
    assert events == ["destroy", "home"]


def test_training_quit_stops_the_subprocess_tree_and_monitor(monkeypatch, tmp_path):
    from yoru import train_GUI
    gui = train_GUI.yoru_train({})
    gui._stop_file = tmp_path / "stop"
    events = []
    class Process:
        stopped = False
        def poll(self):
            return 0 if self.stopped else None
        def wait(self, timeout):
            assert gui._stop_file.exists()
            if not self.stopped:
                raise subprocess.TimeoutExpired("train", timeout)
            events.append("reaped")
    proc = Process()
    gui._train_proc = proc
    def terminate(p):
        assert p is proc
        proc.stopped = True
        events.append("terminated")
    monkeypatch.setattr(train_GUI, "terminate_process_tree", terminate)
    gui._train_monitor = SimpleNamespace(is_alive=lambda: True, join=lambda timeout: events.append("joined"))
    gui.quit_cb()
    gui._shutdown()
    assert events == ["terminated", "reaped", "joined"]
    assert not gui._stop_file.exists()
    assert gui._gpu_stop and gui.m_dict["train_stop_mode"] == "force"


def test_config_model_loader_never_touches_a_closed_context(monkeypatch):
    from yoru import config_creator_GUI
    from yoru.libs import plugins
    gui = config_creator_GUI.ConfigCreatorGUI.__new__(config_creator_GUI.ConfigCreatorGUI)
    gui.m_dict = {"quit": True}
    def forbidden(*a, **kw):
        pytest.fail("worker touched DearPyGui")
    monkeypatch.setattr(config_creator_GUI.dpg, "configure_item", forbidden)
    monkeypatch.setattr(config_creator_GUI.dpg, "set_value", forbidden)
    monkeypatch.setattr(plugins, "get_detector", lambda *a: SimpleNamespace(names={0: "fly"}))
    gui._load_classes("fake", "auto")
    assert gui._class_result == (["fly"], None)


@pytest.mark.parametrize("module_name,analyzer_name", [
    ("evaluation_GUI", "EvaluationImageAnalyzer"),
    ("create_labels_GUI", "yolo_analysis_image"),
])
def test_prediction_does_not_block_quit(monkeypatch, module_name, analyzer_name):
    module = importlib.import_module("yoru." + module_name)
    gui = module.model_eval_gui.__new__(module.model_eval_gui)
    gui.m_dict = {"quit": False}
    started, stopped = threading.Event(), threading.Event()
    def analyze():
        started.set()
        while not gui.m_dict.get("quit", False):
            time.sleep(.001)
        stopped.set()
    monkeypatch.setattr(module, analyzer_name, lambda state: SimpleNamespace(analyze_image=analyze))
    monkeypatch.setattr(module.dpg, "set_value", lambda *a: None)
    gui.yolo_detection()
    assert started.wait(3)
    gui.gui_configure = lambda: None
    gui.plot_callback = lambda: None
    monkeypatch.setattr(module.dpg, "is_dearpygui_running", lambda: True)
    monkeypatch.setattr(module.dpg, "render_dearpygui_frame", gui.quit_cb)
    monkeypatch.setattr(module.dpg, "destroy_context", lambda: None)
    gui.run()
    assert stopped.is_set()
    assert not gui._gui_task["thread"].is_alive()


def test_evaluation_observes_shutdown_before_calculating():
    from yoru.libs.evaluation_calculation import Evaluator
    obj = Evaluator({"quit": True})
    with pytest.raises(InterruptedError, match="cancelled"):
        obj.evaluate([[]], [[]], {0: "fly"}, "test")
