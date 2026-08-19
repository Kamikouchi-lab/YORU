"""Training results must not be written into the dataset's own train/ folder.

``<project>/train/`` holds the training images (see create_yaml_train.py), so
the ultralytics default run name -- ``train`` -- collided with it and every run
landed in ``train2``, ``train3``, ... beside the dataset.  Runs are now named
after the model: ``exp_yolo11s``, ``exp_rtdetr-l``, ``exp_fasterrcnn``, ...
"""

import importlib
import sys
import types
from pathlib import Path

import pytest

ul = importlib.import_module("yoru.libs.train_ultralytics")


def test_ultralytics_run_name_is_named_after_the_model():
    assert ul.run_name("yolo11s.pt") == "exp_yolo11s"
    assert ul.run_name("rtdetr-l.pt") == "exp_rtdetr-l"
    assert ul.run_name(str(Path("models") / "yolov8m.pt")) == "exp_yolov8m"


def test_ultralytics_run_name_never_matches_a_dataset_split():
    for weights in ("yolo11s.pt", "train.pt", "val.pt", ""):
        assert ul.run_name(weights) not in {"train", "val"}


def test_ultralytics_run_name_falls_back_on_a_nameless_weight():
    assert ul.run_name("") == "exp_model"
    assert ul.run_name("my model.pt") == "exp_my_model"


def _fake_ultralytics(recorded):
    class FakeModel:
        def __init__(self, weights):
            recorded["weights"] = weights

        def train(self, **kwargs):
            recorded.update(kwargs)

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeModel
    module.RTDETR = FakeModel
    return module


def _run_main(monkeypatch, tmp_path, *extra):
    recorded = {}
    monkeypatch.setitem(sys.modules, "ultralytics", _fake_ultralytics(recorded))
    monkeypatch.setattr(sys, "argv", [
        "train_ultralytics.py",
        "--weights", "yolo11s.pt",
        "--data", str(tmp_path / "config.yaml"),
        "--project", str(tmp_path),
        *extra,
    ])
    ul.main()
    return recorded


def test_main_hands_the_run_name_to_ultralytics(monkeypatch, tmp_path):
    recorded = _run_main(monkeypatch, tmp_path)
    assert recorded["name"] == "exp_yolo11s"
    assert recorded["project"] == str(tmp_path)


def test_main_honours_an_explicit_name(monkeypatch, tmp_path):
    recorded = _run_main(monkeypatch, tmp_path, "--name", "exp_custom")
    assert recorded["name"] == "exp_custom"


tv = pytest.importorskip(
    "yoru.libs.train_torchvision", reason="torch/torchvision not installed"
)


def test_torchvision_run_name_is_named_after_the_model():
    assert tv.run_name("fasterrcnn") == "exp_fasterrcnn"
    assert tv.run_name("maskrcnn") == "exp_maskrcnn"
    assert tv.run_name("ssd") == "exp_ssd"


def test_unique_run_dir_increments_instead_of_overwriting(tmp_path):
    first = tv.unique_run_dir(tmp_path, "exp_fasterrcnn")
    assert first == tmp_path / "exp_fasterrcnn"
    first.mkdir()

    second = tv.unique_run_dir(tmp_path, "exp_fasterrcnn")
    assert second == tmp_path / "exp_fasterrcnn2"
    second.mkdir()

    assert tv.unique_run_dir(tmp_path, "exp_fasterrcnn") == tmp_path / "exp_fasterrcnn3"


def test_unique_run_dir_ignores_the_dataset_split(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "val").mkdir()
    assert tv.unique_run_dir(tmp_path, tv.run_name("ssd")).name == "exp_ssd"
