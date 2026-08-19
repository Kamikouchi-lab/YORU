"""A training run must be stoppable at a clean point, not only killable.

Killing the trainer in the middle of an epoch throws that epoch away, and for
ultralytics it also throws away the final validation pass and the ``best.pt``
copy, which only happen once the epoch loop ends by itself.  The GUI therefore
*asks* for a stop, through a file the trainer checks at epoch boundaries.
These tests pin down both ends of that protocol.
"""

import importlib
import subprocess
import sys
import types

from yoru.libs import train_stop

ul = importlib.import_module("yoru.libs.train_ultralytics")


# ── the protocol ──────────────────────────────────────────────────────────

def test_the_stop_file_lives_in_the_project_directory(tmp_path):
    assert train_stop.stop_file_for(tmp_path) == tmp_path / train_stop.STOP_FILENAME
    # Dotted: the project directory is also the user's dataset folder.
    assert train_stop.STOP_FILENAME.startswith(".")


def test_a_request_is_visible_until_it_is_taken(tmp_path):
    path = train_stop.stop_file_for(tmp_path)
    assert not train_stop.stop_requested(path)

    train_stop.request_stop(path)
    assert train_stop.stop_requested(path)

    train_stop.clear_stop(path)
    assert not train_stop.stop_requested(path)


def test_asking_twice_is_not_an_error(tmp_path):
    path = train_stop.stop_file_for(tmp_path)
    train_stop.request_stop(path)
    train_stop.request_stop(path)
    assert train_stop.stop_requested(path)


def test_clearing_a_request_nobody_made_is_not_an_error(tmp_path):
    train_stop.clear_stop(tmp_path / "no-such-file")
    train_stop.clear_stop(None)


def test_a_run_without_a_stop_file_never_stops_itself():
    assert train_stop.stop_requested(None) is False


def test_force_stop_kills_the_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        train_stop.terminate_process_tree(proc, timeout=15)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_force_stop_on_a_process_that_already_exited_is_not_an_error():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    train_stop.terminate_process_tree(proc)


# ── ultralytics: the request becomes a clean end of training ──────────────

class _FakeTrainer:
    """Stands in for ultralytics' BaseTrainer, whose epoch loop reads .stop."""

    def __init__(self, epoch=0):
        self.epoch = epoch
        self.stop = False


class _FakeModel:
    def __init__(self, weights=None):
        self.weights = weights
        self.callbacks = {}
        self.train_kwargs = None

    def add_callback(self, event, fn):
        self.callbacks.setdefault(event, []).append(fn)

    def train(self, **kwargs):
        self.train_kwargs = kwargs


def _installed_callback(tmp_path):
    model = _FakeModel()
    path = train_stop.stop_file_for(tmp_path)
    ul.install_stop_callback(model, path)
    (callback,) = model.callbacks["on_train_epoch_end"]
    return callback, path


def test_no_stop_file_means_no_callback():
    model = _FakeModel()
    ul.install_stop_callback(model, None)
    assert model.callbacks == {}


def test_the_callback_stops_the_trainer_only_once_asked(tmp_path):
    callback, path = _installed_callback(tmp_path)
    trainer = _FakeTrainer(epoch=2)

    callback(trainer)
    assert trainer.stop is False

    train_stop.request_stop(path)
    callback(trainer)
    assert trainer.stop is True


def test_the_callback_takes_the_request_off_the_disk(tmp_path):
    """Otherwise the file would still be there to stop the *next* run."""
    callback, path = _installed_callback(tmp_path)
    train_stop.request_stop(path)
    callback(_FakeTrainer())
    assert not path.exists()


def test_the_stop_hook_reaches_the_model_the_cli_builds(monkeypatch, tmp_path):
    created = {}

    def _make(weights):
        created["model"] = _FakeModel(weights)
        return created["model"]

    module = types.ModuleType("ultralytics")
    module.YOLO = _make
    module.RTDETR = _make
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(sys, "argv", [
        "train_ultralytics.py",
        "--weights", "yolo11s.pt",
        "--data", str(tmp_path / "config.yaml"),
        "--project", str(tmp_path),
        "--stop-file", str(train_stop.stop_file_for(tmp_path)),
    ])

    ul.main()
    assert "on_train_epoch_end" in created["model"].callbacks


# ── the plugins hand the stop file to the training script ─────────────────

def _recorded_cmd(monkeypatch, module, config):
    recorded = {}

    def _fake_popen(cmd, **kwargs):
        recorded["cmd"] = cmd
        return object()

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    module_trainer = (
        module.UltralyticsTrainer if hasattr(module, "UltralyticsTrainer")
        else module.TorchvisionTrainer
    )
    module_trainer().train(config)
    return recorded["cmd"]


def _base_config(tmp_path):
    return {
        "img_size": 320,
        "batch_size": 1,
        "epochs": 1,
        "data_yaml": str(tmp_path / "config.yaml"),
        "weights": "yolo11s.pt",
        "project_dir": str(tmp_path),
        "model_family": "YOLO",
    }


def test_ultralytics_plugin_passes_the_stop_file(monkeypatch, tmp_path):
    from yoru.libs.plugins import ultralytics_trainer as plugin

    stop_file = str(train_stop.stop_file_for(tmp_path))
    config = dict(_base_config(tmp_path), stop_file=stop_file)
    cmd = _recorded_cmd(monkeypatch, plugin, config)
    assert "--stop-file" in cmd
    assert cmd[cmd.index("--stop-file") + 1] == stop_file


def test_ultralytics_plugin_omits_the_flag_when_there_is_no_stop_file(
    monkeypatch, tmp_path
):
    from yoru.libs.plugins import ultralytics_trainer as plugin

    cmd = _recorded_cmd(monkeypatch, plugin, _base_config(tmp_path))
    assert "--stop-file" not in cmd


def test_torchvision_plugin_passes_the_stop_file(monkeypatch, tmp_path):
    from yoru.libs.plugins import torchvision_trainer as plugin

    stop_file = str(train_stop.stop_file_for(tmp_path))
    config = dict(
        _base_config(tmp_path), model_family="Faster R-CNN", stop_file=stop_file
    )
    cmd = _recorded_cmd(monkeypatch, plugin, config)
    assert "--stop-file" in cmd
    assert cmd[cmd.index("--stop-file") + 1] == stop_file
