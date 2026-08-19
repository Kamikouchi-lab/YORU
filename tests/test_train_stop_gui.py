"""The training GUI's half of the stop protocol: buttons, state, and reporting.

The controls have to say what is true at every point of a run -- a stop that is
pending is not a stop that has happened, and a run that was cut short did not
"Complete!!".  DearPyGui is replaced by a recorder here so the state machine can
be driven without a window.
"""

import sys
import types

import pytest

from yoru.libs import train_stop


class _FakeDPG(types.ModuleType):
    """Records what the GUI would have drawn."""

    def __init__(self):
        super().__init__("dearpygui.dearpygui")
        self.values = {}
        self.config = {}
        self.themes = {}

    def set_value(self, tag, value):
        self.values[tag] = value

    def get_value(self, tag):
        return self.values.get(tag)

    def configure_item(self, tag, **kwargs):
        self.config.setdefault(tag, {}).update(kwargs)

    def bind_item_theme(self, tag, theme):
        self.themes[tag] = theme

    def __getattr__(self, name):
        raise AttributeError(f"the stop controls should not call dpg.{name}")


class _FakeProc:
    """A finished training subprocess: a few lines of stdout, then exit."""

    def __init__(self, lines=()):
        self.stdout = iter(lines)
        self.pid = -1
        self.terminated = False

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True


@pytest.fixture
def gui(tmp_path, monkeypatch):
    """A yoru_train wired to a recording dpg, as if a project were loaded."""
    pytest.importorskip("yaml")
    pytest.importorskip("tkinter", reason="file_operation_train needs tkinter")

    fake = _FakeDPG()
    package = types.ModuleType("dearpygui")
    package.__path__ = []
    package.dearpygui = fake
    monkeypatch.setitem(sys.modules, "dearpygui", package)
    monkeypatch.setitem(sys.modules, "dearpygui.dearpygui", fake)
    # Re-imported per test so the module binds this test's recorder.
    monkeypatch.delitem(sys.modules, "yoru.train_GUI", raising=False)

    train_GUI = pytest.importorskip("yoru.train_GUI")
    instance = train_GUI.yoru_train(m_dict={})
    # Normally built in startDPG(), which needs a real viewport.
    for theme in ("_complete_theme", "_warn_theme", "_error_theme", "_yet_theme"):
        setattr(instance, theme, theme)
    instance._stop_file = train_stop.stop_file_for(tmp_path)
    instance.dpg = fake
    yield instance
    # The imported module holds a reference to this test's recorder; leaving it
    # in sys.modules would hand the stub to whoever imports the GUI next.
    sys.modules.pop("yoru.train_GUI", None)


def _start(gui, epoch=0, total=300):
    gui.m_dict["training_active"] = True
    gui.m_dict["train_stop_mode"] = ""
    gui.m_dict["train_epoch"] = epoch
    gui.m_dict["train_total_epoch"] = total
    gui._sync_stop_controls()


def test_stopping_is_offered_only_while_a_run_is_going(gui):
    gui._sync_stop_controls()
    assert gui.dpg.config["str_btn"]["enabled"] is True
    assert gui.dpg.config["stop_btn"]["enabled"] is False
    assert gui.dpg.config["force_stop_btn"]["show"] is False

    _start(gui)
    assert gui.dpg.config["str_btn"]["enabled"] is False
    assert gui.dpg.config["stop_btn"]["enabled"] is True


def test_a_pending_stop_replaces_the_stop_button_with_the_forced_one(gui):
    _start(gui, epoch=12)
    gui.stop_after_epoch()
    gui._sync_stop_controls()

    assert gui._stop_file.exists()
    assert gui.m_dict["train_stop_mode"] == "graceful"
    # Asking twice would do nothing, and the run is not over yet.
    assert gui.dpg.config["stop_btn"]["enabled"] is False
    assert gui.dpg.config["force_stop_btn"]["show"] is True
    assert "epoch in progress" in gui.dpg.values["train_stop_text"]


def test_there_is_nothing_to_stop_before_the_first_run(gui):
    gui._stop_file = None
    gui.stop_after_epoch()
    assert not gui.m_dict.get("train_stop_mode")


def test_a_stopped_run_is_reported_as_stopped_not_complete(gui):
    _start(gui, epoch=12)
    gui.stop_after_epoch()
    # The trainer takes the request, finishes epoch 12 and exits.
    train_stop.clear_stop(gui._stop_file)
    gui._monitor_training(_FakeProc(["  12/300   0.4  0.5  640: 100%"]), 300)

    assert gui.m_dict["training_active"] is False
    assert gui.m_dict["training_done"] is True
    # Not rounded up to 300/300: the run really did end at epoch 12.
    assert gui.m_dict["train_epoch"] == 12

    gui._show_training_outcome()
    assert gui.dpg.values["step6_state"] == "Stopped"
    assert gui.dpg.values["train_progress_text"] == "Stopped at epoch 12 / 300"
    assert "epoch 12" in gui.dpg.values["train_stop_text"]


def test_the_controls_come_back_once_the_run_is_over(gui):
    _start(gui, epoch=12)
    gui.stop_after_epoch()
    gui._monitor_training(_FakeProc(["  12/300   0.4  0.5  640: 100%"]), 300)
    gui._sync_stop_controls()

    assert gui.dpg.config["str_btn"]["enabled"] is True
    assert gui.dpg.config["stop_btn"]["enabled"] is False
    assert gui.dpg.config["force_stop_btn"]["show"] is False


def test_a_request_the_trainer_never_took_does_not_outlive_the_run(gui):
    """A forced kill can beat the trainer to its own stop file."""
    _start(gui, epoch=12)
    gui.stop_after_epoch()
    assert gui._stop_file.exists()

    gui._monitor_training(_FakeProc([]), 300)
    assert not gui._stop_file.exists()


def test_a_run_that_finishes_normally_still_reports_complete(gui):
    _start(gui)
    gui._monitor_training(_FakeProc(["  300/300   0.1  0.2  640: 100%"]), 300)
    gui._show_training_outcome()

    assert gui.dpg.values["step6_state"] == "Complete!!"
    assert gui.dpg.values["train_progress_text"] == "Done!"
    assert gui.dpg.values["train_stop_text"] == ""


def test_a_stop_asked_for_during_the_last_epoch_is_not_a_stopped_run(gui):
    _start(gui, epoch=300)
    gui.stop_after_epoch()
    gui._monitor_training(_FakeProc(["  300/300   0.1  0.2  640: 100%"]), 300)
    gui._show_training_outcome()

    assert gui.dpg.values["step6_state"] == "Complete!!"


def test_force_stop_closes_the_prompt_and_records_the_kill(gui):
    _start(gui, epoch=12)
    gui.stop_after_epoch()
    gui._train_proc = _FakeProc()
    gui.force_stop()

    assert gui.dpg.config["force_stop_modal"]["show"] is False
    assert gui.m_dict["train_stop_mode"] == "force"


def test_force_stop_after_the_run_ended_is_a_no_op(gui):
    _start(gui)
    gui._train_proc = None
    gui.force_stop()
    assert gui.m_dict["train_stop_mode"] == ""
