"""The upstream-YOLOv5 backend, and the v1 checkpoints only it can read.

A checkpoint written by YORU v1 pickles ``models.yolo.DetectionModel``, which
exists in the vendored tree and nowhere else -- ultralytics rejects those files
by name.  These tests build a checkpoint in that same format and take it all
the way through the plugin, because the failure mode being guarded against
(``ModuleNotFoundError: No module named 'models'``) only shows up at unpickle
time and only for a file laid out this way.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
np = pytest.importorskip("numpy", reason="numpy not installed")

from yoru.libs import plugins  # noqa: E402
from yoru.libs.detector_base import DETECTION_COLUMNS, detection_row  # noqa: E402
from yoru.libs.yolov5 import YOLOV5_DIR, ensure_importable  # noqa: E402

# The smallest architecture upstream ships: these tests build one for real, so
# the difference between yolov5n and yolov5x is the difference between a test
# suite that runs and one nobody waits for.
_ARCH = "yolov5n.yaml"
_NAMES = {0: "fly", 1: "wing"}


@pytest.fixture(scope="module")
def legacy_checkpoint(tmp_path_factory):
    """A .pt in v1's format: the model object itself, pickled by class path."""
    ensure_importable()
    from models.yolo import DetectionModel

    with contextlib.redirect_stdout(io.StringIO()):
        model = DetectionModel(str(YOLOV5_DIR / "models" / _ARCH), ch=3, nc=len(_NAMES))
    model.names = dict(_NAMES)

    path = tmp_path_factory.mktemp("yolov5") / "best.pt"
    torch.save({"model": model, "epoch": -1}, path)
    return path


class TestVendoredImport:
    def test_binds_the_names_a_legacy_pickle_asks_for(self):
        ensure_importable()
        assert "models" in sys.modules and "utils" in sys.modules
        import models.yolo  # noqa: F401  -- the path the unpickler walks

        assert str(YOLOV5_DIR) in models.yolo.__file__

    def test_is_idempotent(self):
        first = ensure_importable()
        models_before = sys.modules["models"]
        assert ensure_importable() == first
        assert sys.modules["models"] is models_before

    def test_does_not_prepend_itself_to_sys_path(self):
        """v1 did, via torch.hub, and shadowed every later 'import utils'."""
        ensure_importable()
        assert str(YOLOV5_DIR) != sys.path[0]

    def test_refuses_to_overwrite_somebody_else_s_module(self, monkeypatch):
        import types

        import yoru.libs.yolov5 as vendored

        monkeypatch.setitem(sys.modules, "models", types.ModuleType("models"))
        with pytest.raises(RuntimeError, match="already taken"):
            vendored._bind("models")


class TestBackendRouting:
    def test_a_v1_checkpoint_is_recognised_by_its_contents(self, legacy_checkpoint):
        """Not by its name: v1 projects call their weights best.pt too."""
        assert plugins._sniff_checkpoint(str(legacy_checkpoint)) == "yolov5"

    def test_auto_sends_it_to_the_yolov5_backend(self, legacy_checkpoint):
        assert plugins._auto_detect_backend(str(legacy_checkpoint)) == "yolov5"

    def test_the_backend_is_registered(self):
        plugins._ensure_plugins_loaded()
        assert "yolov5" in plugins._DETECTOR_REGISTRY
        assert "yolov5" in plugins._TRAINER_REGISTRY
        assert not plugins._PLUGIN_IMPORT_ERRORS, plugins._PLUGIN_IMPORT_ERRORS

    @pytest.mark.parametrize("weight,expected", [
        ("yolov5s.pt", "yolov5"),
        ("yolov5x.pt", "yolov5"),
        ("yolov5su.pt", "ultralytics"),
        ("yolo11s.pt", "ultralytics"),
    ])
    def test_the_trainer_follows_the_weight(self, weight, expected):
        m = {"model_family": "YOLO", "weight": weight}
        assert plugins.detect_trainer_backend(m) == expected


class TestDetection:
    @pytest.fixture(scope="class")
    def detector(self, legacy_checkpoint):
        with contextlib.redirect_stdout(io.StringIO()):
            return plugins.get_detector(
                "auto", str(legacy_checkpoint), conf_thresh=0.001, device="cpu"
            )

    def test_class_names_survive_the_round_trip(self, detector):
        assert detector.names == _NAMES

    def test_detections_have_the_shape_the_rest_of_yoru_expects(self, detector):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for det in detector.detect(frame):
            assert {"x1", "y1", "x2", "y2", "conf", "class_id", "class_name"} <= set(det)
            # Upright boxes only: obb_of() derives the rest.
            assert "angle" not in det
            assert len(detection_row(det, 0.01)) == len(DETECTION_COLUMNS)

    def test_boxes_come_back_in_frame_coordinates_not_letterboxed_ones(self, detector):
        """A 640x640 model on a wide frame: scale_boxes has to undo the pad."""
        h, w = 480, 1280
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        for det in detector.detect(frame):
            assert -1 <= det["x1"] <= w + 1 and -1 <= det["x2"] <= w + 1
            assert -1 <= det["y1"] <= h + 1 and -1 <= det["y2"] <= h + 1

    def test_the_confidence_threshold_is_honoured(self, legacy_checkpoint):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        with contextlib.redirect_stdout(io.StringIO()):
            strict = plugins.get_detector(
                "yolov5", str(legacy_checkpoint), conf_thresh=0.99, device="cpu"
            )
        assert strict.detect(frame) == []


class TestTrainerCommand:
    def _cmd(self, **overrides):
        config = {
            "weights": "yolov5s.pt",
            "data_yaml": "proj/config.yaml",
            "epochs": 50,
            "img_size": 640,
            "batch_size": 8,
            "project_dir": "proj",
        }
        config.update(overrides)
        trainer = plugins.get_trainer("yolov5")
        captured = {}

        class _FakePopen:
            def __init__(self, cmd, **kwargs):
                captured["cmd"] = cmd
                captured["kwargs"] = kwargs

        import yoru.libs.plugins.yolov5_trainer as mod

        real, mod.subprocess.Popen = mod.subprocess.Popen, _FakePopen
        try:
            trainer.train(config)
        finally:
            mod.subprocess.Popen = real
        return captured

    def test_it_runs_yorus_launcher_not_upstream_train_py(self):
        """train.main() would git-fetch and pip-install; the launcher does not."""
        cmd = self._cmd()["cmd"]
        assert cmd[1].endswith("train_yolov5.py")

    def test_the_settings_reach_the_command_line(self):
        cmd = self._cmd()["cmd"]
        for flag, value in [("--weights", "yolov5s.pt"), ("--epochs", "50"),
                            ("--imgsz", "640"), ("--batch", "8")]:
            assert cmd[cmd.index(flag) + 1] == value

    def test_the_stop_file_is_passed_only_when_there_is_one(self):
        assert "--stop-file" not in self._cmd()["cmd"]
        cmd = self._cmd(stop_file="proj/.yoru_stop_request")["cmd"]
        assert cmd[cmd.index("--stop-file") + 1] == "proj/.yoru_stop_request"

    def test_it_runs_from_the_repository_root(self):
        """Upstream relpath()s its own directory against the cwd, which raises
        on Windows when the project lives on another drive."""
        kwargs = self._cmd()["kwargs"]
        assert (Path(kwargs["cwd"]) / "yoru" / "libs" / "yolov5").is_dir()


class TestStopFileHook:
    """The patch YORU carries in the vendored train.py."""

    def test_no_env_var_means_no_stop(self, monkeypatch):
        ensure_importable()
        import train as yolov5_train

        monkeypatch.delenv("YORU_STOP_FILE", raising=False)
        assert yolov5_train.yoru_stop_requested() is False

    def test_the_request_is_seen_and_then_taken(self, monkeypatch, tmp_path):
        ensure_importable()
        import train as yolov5_train

        stop_file = tmp_path / ".yoru_stop_request"
        monkeypatch.setenv("YORU_STOP_FILE", str(stop_file))
        assert yolov5_train.yoru_stop_requested() is False

        stop_file.touch()
        assert yolov5_train.yoru_stop_requested() is True
        # Taken, so it cannot also end the next run after one epoch.
        assert not stop_file.exists()
        assert yolov5_train.yoru_stop_requested() is False


class TestObbIsNotOffered:
    """YOLOv5 has no rotated-box head, so an OBB project must not show it."""

    def test_the_version_list_excludes_it_for_obb(self):
        from yoru.libs.init_train import (
            MODEL_FAMILY_CONFIG,
            OBB_CAPABLE_YOLO_VERSIONS,
        )

        assert "YOLOv5" in MODEL_FAMILY_CONFIG["YOLO"]["versions"]
        assert "YOLOv5" not in OBB_CAPABLE_YOLO_VERSIONS

    def test_no_obb_weight_is_offered_for_it(self):
        from yoru.libs.init_train import init_train

        m = {}
        init_train(m)
        assert not [w for w in m["weight_list"] if "yolov5" in w and "obb" in w]

    def test_every_plain_yolov5_weight_it_can_build_is_offered(self):
        from yoru.libs.init_train import init_train

        m = {}
        init_train(m)
        for size in m["yolo_size_list"]:
            assert f"yolov5{size}.pt" in m["weight_list"]


class TestVramEstimate:
    def test_yolov5_is_covered(self):
        from yoru.libs.vram_estimate import profile_for

        for size in "nsmlx":
            assert profile_for(f"yolov5{size}.pt") is not None

    def test_the_assigner_workspace_is_not_charged_to_it(self):
        """Anchor matching costs (anchors, labels), not (batch, labels, points)."""
        from yoru.libs.vram_estimate import profile_for

        assert profile_for("yolov5s.pt").anchors == 0
        assert profile_for("yolov8s.pt").anchors > 0

    def test_a_yolov5u_name_does_not_borrow_the_yolov5_row(self):
        """Different parameter count and a different loss; no estimate beats a
        wrong one."""
        from yoru.libs.vram_estimate import profile_for

        assert profile_for("yolov5su.pt") is None


# ---------------------------------------------------------------------------
# The training GUI's model selector
# ---------------------------------------------------------------------------

class _FakeDpg:
    """Enough of DearPyGui to exercise the selector logic without a context."""

    def __init__(self):
        self.values = {}
        self.items = {}

    def set_value(self, tag, value):
        self.values[tag] = value

    def get_value(self, tag):
        return self.values.get(tag)

    def configure_item(self, tag, **kw):
        self.items.setdefault(tag, {}).update(kw)

    def enable_item(self, tag):
        pass

    def disable_item(self, tag):
        pass


@pytest.fixture
def gui(monkeypatch):
    from yoru import train_GUI
    from yoru.libs.init_train import init_train

    fake = _FakeDpg()
    monkeypatch.setattr(train_GUI, "dpg", fake)
    g = train_GUI.yoru_train.__new__(train_GUI.yoru_train)
    g.m_dict = {}
    init_train(g.m_dict)
    return g, fake


class TestTrainingGuiSelector:
    @pytest.mark.parametrize("size", list("nsmlx"))
    def test_it_builds_a_yolov5_weight(self, gui, size):
        g, _ = gui
        g.m_dict.update(model_family="YOLO", yolo_version="YOLOv5", yolo_size=size)
        assert g._build_weight() == f"yolov5{size}.pt"

    def test_a_v1_project_is_restored_as_yolov5_not_remapped(self, gui):
        """v2.0 swapped these onto YOLO11 because nothing could train them."""
        g, fake = gui
        g._restore_model_ui("yolov5m.pt")
        assert g.m_dict["yolo_version"] == "YOLOv5"
        assert g.m_dict["yolo_size"] == "m"
        assert g.m_dict["weight"] == "yolov5m.pt"
        assert fake.values["weight_display_text"] == "yolov5m.pt"

    def test_a_bare_yolov5_in_a_v1_config_gains_its_size_and_extension(self, gui):
        g, _ = gui
        g._restore_model_ui("yolov5.pt")
        assert g.m_dict["weight"] == "yolov5s.pt"

    def test_a_yolov5u_weight_is_left_exactly_as_it_is(self, gui):
        """Rewriting it to yolov5s.pt would quietly swap the model."""
        g, _ = gui
        g.m_dict["weight"] = "yolov5su.pt"
        g._restore_model_ui("yolov5su.pt")
        assert g.m_dict["weight"] == "yolov5su.pt"

    def test_an_obb_project_does_not_offer_yolov5(self, gui):
        g, fake = gui
        g.m_dict["obb"] = True
        g.m_dict["model_family"] = "YOLO"
        g._sync_obb_ui()
        assert "YOLOv5" not in fake.items["yolo_version_combo"]["items"]

    def test_a_detection_project_does_offer_it(self, gui):
        g, fake = gui
        g.m_dict["obb"] = False
        g._sync_obb_ui()
        assert "YOLOv5" in fake.items["yolo_version_combo"]["items"]

    def test_ticking_obb_moves_a_yolov5_selection_off_it(self, gui):
        g, fake = gui
        g.m_dict.update(model_family="YOLO", yolo_version="YOLOv5", yolo_size="s")
        g.m_dict["obb"] = True
        g._sync_obb_ui()
        assert g.m_dict["yolo_version"] == "YOLO11"
        assert g.m_dict["weight"] == "yolo11s-obb.pt"

    def test_selecting_yolov5_in_an_obb_project_is_refused(self, gui):
        g, fake = gui
        g.m_dict.update(model_family="YOLO", yolo_version="YOLO11", yolo_size="s",
                        obb=True)
        fake.set_value("yolo_version_combo", "YOLOv5")
        g.select_version()
        assert g.m_dict["yolo_version"] == "YOLO11"
        assert fake.values["yolo_version_combo"] == "YOLO11"
