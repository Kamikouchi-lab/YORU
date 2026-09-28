"""The Evaluation GUI's "Run LabelImg" button opens the bundled labelImg.

It used to run the ``labelImg`` on PATH.  That copy shares
``~/.labelImgSettings.pkl`` with the bundled one, and once the bundled one has
saved a ``yoru.labelimg`` format there, the upstream one fails to start
wherever ``yoru`` is importable.  It also cannot read YOLO-OBB labels.  The
command is checked without launching anything: ``Popen`` and DearPyGui are
replaced by recorders.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
import yaml

import yoru.evaluation_GUI as evaluation_GUI


class _FakeDpg:
    def __init__(self):
        self.values = {}

    def set_value(self, tag, value):
        self.values[tag] = value


@pytest.fixture
def gui(monkeypatch):
    fake = _FakeDpg()
    monkeypatch.setattr(evaluation_GUI, "dpg", fake)
    launched = []

    def fake_popen(cmd, **kwargs):
        launched.append((cmd, kwargs))
        return types.SimpleNamespace()

    monkeypatch.setattr(evaluation_GUI.subprocess, "Popen", fake_popen)
    g = evaluation_GUI.model_eval_gui.__new__(evaluation_GUI.model_eval_gui)
    g.m_dict = {}
    evaluation_GUI.init_evaluater(g.m_dict)
    return g, fake, launched


def _write_config(path, project_dir, **extra):
    data = {"project_dir": str(project_dir), **extra}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_the_bundled_labelimg_is_launched_not_the_one_on_path(gui):
    g, fake, launched = gui
    g.labelImg_bt()
    (cmd, _kwargs), = launched
    assert cmd[:3] == [sys.executable, "-m", "yoru.labelimg.labelimg"]
    assert fake.values["step3_state"] == "Complete!!"


def test_with_no_project_loaded_it_opens_empty_as_axis_aligned(gui):
    g, _fake, launched = gui
    g.labelImg_bt()
    (cmd, _kwargs), = launched
    assert cmd[3:] == ["--no-obb"]


def test_a_detection_project_opens_its_evaluation_data(gui, tmp_path):
    g, _fake, launched = gui
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, tmp_path)
    g.m_dict["config_file_path"] = str(cfg)
    g.load_pr_dir()

    data_dir = g.m_dict["data_dir"]
    g.labelImg_bt()
    (cmd, _kwargs), = launched
    # No classes.txt yet: the empty argument means the bundled classes.
    assert cmd[3:] == [data_dir, "", data_dir, "--no-obb"]


def test_an_obb_project_opens_labelimg_in_obb_mode(gui, tmp_path):
    g, _fake, launched = gui
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, tmp_path, task="obb")
    g.m_dict["config_file_path"] = str(cfg)
    g.load_pr_dir()
    assert g.m_dict["obb"] is True

    data_dir = g.m_dict["data_dir"]
    # A classes.txt beside the evaluation images is passed on.
    classes_txt = os.path.join(data_dir, "classes.txt")
    with open(classes_txt, "w") as f:
        f.write("fly\n")
    g.labelImg_bt()
    (cmd, _kwargs), = launched
    assert cmd[3:] == [data_dir, classes_txt, data_dir, "--obb"]


def test_the_command_is_one_the_bundled_labelimg_accepts(gui, tmp_path):
    pytest.importorskip("PyQt5", reason="PyQt5 not installed")
    from yoru.labelimg.labelimg import build_arg_parser

    g, _fake, launched = gui
    cfg = tmp_path / "config.yaml"
    _write_config(cfg, tmp_path, task="obb")
    g.m_dict["config_file_path"] = str(cfg)
    g.load_pr_dir()
    g.labelImg_bt()
    (cmd, _kwargs), = launched
    args = build_arg_parser().parse_args(cmd[3:])
    assert args.image_dir == g.m_dict["data_dir"]
    assert args.save_dir == g.m_dict["data_dir"]
    assert args.obb is True


def test_a_launch_failure_is_reported(gui, monkeypatch):
    g, fake, _launched = gui
    reported = []

    def failing_popen(cmd, **kwargs):
        raise OSError("no such interpreter")

    monkeypatch.setattr(evaluation_GUI.subprocess, "Popen", failing_popen)
    monkeypatch.setattr(g, "_report_error", lambda msg, e: reported.append(msg),
                        raising=False)
    g.labelImg_bt()
    assert reported == ["Failed to launch LabelImg"]
    assert fake.values["step3_state"] == "Error"
