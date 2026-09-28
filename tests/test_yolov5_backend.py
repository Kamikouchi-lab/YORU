"""YOLOv5, run by the bundled ultralytics/yolov5 code: recognition, training, loading.

YORU v1 trained every model with YOLOv5, and v2.0 beta could load none of them:
the "yolov5" model type was routed to ultralytics, which refuses a YOLOv5
checkpoint.  These tests pin down each piece of the way back -- that such a
checkpoint is recognised whatever it is called, that training reaches yolov5's
own train.py with the arguments v1 gave it, and that a loaded model gives the
detections v1's ``torch.hub.load`` gave.
"""

import os
import pickle
import sys
import types
import zipfile
from pathlib import Path

import pytest

from yoru.libs import plugins, yolov5_support
from yoru.libs import train_yolov5 as tv5
from yoru.libs.plugins import yolov5_detector as v5det


# ---------------------------------------------------------------------------
# Recognising a YOLOv5 checkpoint without unpickling it
# ---------------------------------------------------------------------------


def _pickled_instance(monkeypatch, module_name, protocol, extra=None):
    """Bytes of a pickle naming a class of *module_name*, as torch.save writes."""
    # pickle imports the class's module, parents included, to check it.
    parts = module_name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    module = types.ModuleType(module_name)

    class DetectionModel:
        pass

    DetectionModel.__module__ = module_name
    DetectionModel.__qualname__ = "DetectionModel"
    module.DetectionModel = DetectionModel
    monkeypatch.setitem(sys.modules, module_name, module)
    payload = {"model": DetectionModel(), "epoch": 3}
    if extra:
        payload.update(extra)
    return pickle.dumps(payload, protocol=protocol)


def _checkpoint(path: Path, blob: bytes) -> Path:
    """A zip-format torch checkpoint whose data.pkl is *blob*."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("archive/data.pkl", blob)
        zf.writestr("archive/data/0", b"\0" * 16)
    return path


@pytest.mark.parametrize("protocol", [2, 4])
def test_a_yolov5_checkpoint_is_recognised_by_its_contents(tmp_path, monkeypatch, protocol):
    blob = _pickled_instance(monkeypatch, "models.yolo", protocol)
    ckpt = _checkpoint(tmp_path / "best.pt", blob)
    assert plugins._sniff_checkpoint(str(ckpt)) == "yolov5"
    assert plugins._auto_detect_backend(str(ckpt)) == "yolov5"


def test_the_contents_outrank_a_misleading_name(tmp_path, monkeypatch):
    """v1 users named their models freely; the file decides, not the name."""
    blob = _pickled_instance(monkeypatch, "models.yolo", 2)
    ckpt = _checkpoint(tmp_path / "yolov8_compare_best.pt", blob)
    assert plugins._auto_detect_backend(str(ckpt)) == "yolov5"


def test_a_clone_of_the_yolov5_repo_is_not_mistaken_for_ultralytics(tmp_path, monkeypatch):
    """yolov5 saves its git remote -- github.com/ultralytics/yolov5 -- in every run."""
    blob = _pickled_instance(
        monkeypatch, "models.yolo", 2,
        extra={"git": {"remote": "https://github.com/ultralytics/yolov5"}},
    )
    ckpt = _checkpoint(tmp_path / "best.pt", blob)
    assert b"ultralytics" in blob
    assert plugins._sniff_checkpoint(str(ckpt)) == "yolov5"


@pytest.mark.parametrize("module_name", ["ultralytics.nn.tasks", "ultralytics.models.yolo.detect"])
def test_an_ultralytics_checkpoint_stays_ultralytics(tmp_path, monkeypatch, module_name):
    """Including one naming ultralytics' own ``models.yolo`` subpackage."""
    blob = _pickled_instance(monkeypatch, module_name, 2)
    ckpt = _checkpoint(tmp_path / "best.pt", blob)
    assert plugins._sniff_checkpoint(str(ckpt)) == "ultralytics"
    assert plugins._auto_detect_backend(str(ckpt)) == "ultralytics"


def test_a_pre_zip_yolov5_checkpoint_is_recognised(tmp_path, monkeypatch):
    """torch < 1.6 wrote a bare pickle stream; early YOLOv5 releases predate the zip."""
    blob = _pickled_instance(monkeypatch, "models.yolo", 2)
    ckpt = tmp_path / "last.pt"
    ckpt.write_bytes(b"\x80\x02\x8a\x0al\xfc\x9cF\xf9 j\xa8P\x19." + blob + b"\0" * 64)
    assert plugins._sniff_checkpoint(str(ckpt)) == "yolov5"


def test_yolov5_weights_train_with_the_bundled_yolov5():
    for weight in ("yolov5s.pt", "yolov5x.pt", r"C:\models\yolov5m.pt"):
        assert plugins.detect_trainer_backend(
            {"model_family": "YOLO", "weight": weight}
        ) == "yolov5"


def test_the_yolov5_backends_are_registered():
    """Registered -- that is, their modules import -- for both directions."""
    pytest.importorskip("torch")
    plugins._ensure_plugins_loaded()
    assert "yolov5" in plugins._DETECTOR_REGISTRY, plugins._PLUGIN_IMPORT_ERRORS
    assert "yolov5" in plugins._TRAINER_REGISTRY, plugins._PLUGIN_IMPORT_ERRORS


# ---------------------------------------------------------------------------
# Making the bundled code importable
# ---------------------------------------------------------------------------


def test_the_bundled_yolov5_is_found():
    assert yolov5_support.require_bundled_yolov5() == yolov5_support.YOLOV5_DIR


def test_the_path_entry_is_removed_again(monkeypatch):
    monkeypatch.delitem(sys.modules, "models", raising=False)
    monkeypatch.delitem(sys.modules, "utils", raising=False)
    before = list(sys.path)
    with yolov5_support.yolov5_importable() as root:
        assert sys.path[0] == str(root)
    assert sys.path == before


def test_a_foreign_utils_module_is_reported_not_used(monkeypatch, tmp_path):
    """yolov5's ``from utils.general import ...`` would silently get the wrong one."""
    foreign = types.ModuleType("utils")
    foreign.__file__ = str(tmp_path / "utils.py")
    monkeypatch.setitem(sys.modules, "utils", foreign)
    with pytest.raises(ImportError, match="utils"):
        with yolov5_support.yolov5_importable():
            pass


# ---------------------------------------------------------------------------
# Training: the command, the device, the run folder, the stop request
# ---------------------------------------------------------------------------


def test_the_run_is_named_after_the_model():
    assert tv5.run_name("yolov5s.pt") == "exp_yolov5s"
    assert tv5.run_name(r"C:\w\yolov5m.pt") == "exp_yolov5m"


@pytest.mark.parametrize(
    "resolved,expected",
    [
        # yolov5's select_device() does not know "cuda"; its default is CUDA:0.
        ("cuda", ""),
        ("cpu", "cpu"),
        ("mps", "mps"),
        ("1", "1"),
    ],
)
def test_the_device_is_named_only_when_it_is_not_the_default(monkeypatch, resolved, expected):
    monkeypatch.setattr(tv5, "resolve_device", lambda _pref: resolved)
    assert tv5.yolov5_device("auto") == expected


def test_train_py_gets_the_arguments_yoru_v1_gave_it(monkeypatch):
    monkeypatch.setattr(tv5, "resolve_device", lambda _pref: "cuda")
    args = types.SimpleNamespace(
        imgsz=640, batch=16, epochs=300, data="d/config.yaml",
        weights="yolov5s.pt", project="d", name=None, device="auto",
    )
    argv = tv5.yolov5_argv(args)
    assert argv == [
        "--imgsz", "640", "--batch-size", "16", "--epochs", "300",
        "--data", "d/config.yaml", "--weights", "yolov5s.pt", "--project", "d",
        "--name", "exp_yolov5s",
    ]


def test_a_working_directory_on_another_drive_is_left_before_importing(monkeypatch, tmp_path):
    """yolov5's train.py calls os.path.relpath at import; Windows raises across drives."""
    moved = []

    def _relpath(*_a):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")

    monkeypatch.setattr(tv5.os.path, "relpath", _relpath)
    monkeypatch.setattr(tv5.os, "chdir", moved.append)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"")
    args = types.SimpleNamespace(
        data="config.yaml", project="proj", weights=str(weights),
    )
    tv5._cwd_on_the_drive_of(yolov5_support.YOLOV5_DIR, args)

    assert moved == [tv5._REPO_ROOT]
    # Made absolute first, so they still name what they named.
    assert Path(args.data).is_absolute() and Path(args.project).is_absolute()
    assert Path(args.weights) == weights.resolve()


def test_a_working_directory_on_the_same_drive_is_kept(monkeypatch):
    moved = []
    monkeypatch.setattr(tv5.os, "chdir", moved.append)
    args = types.SimpleNamespace(data="config.yaml", project="proj", weights="yolov5s.pt")
    tv5._cwd_on_the_drive_of(Path.cwd(), args)
    assert moved == []
    assert args.data == "config.yaml"


def _fake_train_module():
    class EarlyStopping:
        def __init__(self, patience=30):
            self.patience = patience

        def __call__(self, epoch, fitness):
            return False

    return types.SimpleNamespace(EarlyStopping=EarlyStopping)


def test_a_stop_request_ends_the_run_after_the_epoch(tmp_path, capsys):
    stop_file = tmp_path / ".yoru_stop_request"
    module = _fake_train_module()
    tv5.install_stop_request(module, str(stop_file))

    stopper = module.EarlyStopping(patience=100)
    assert stopper.patience == 100
    assert stopper(epoch=4, fitness=0.5) is False

    stop_file.touch()
    assert stopper(epoch=5, fitness=0.5) is True
    # Taken, so it cannot end the next run too; reported 1-based, as the GUI shows it.
    assert not stop_file.exists()
    assert "ending after epoch 6" in capsys.readouterr().out


def test_no_stop_file_leaves_train_py_alone():
    module = _fake_train_module()
    original = module.EarlyStopping
    tv5.install_stop_request(module, None)
    assert module.EarlyStopping is original


def test_the_trainer_launches_the_yolov5_script(monkeypatch):
    pytest.importorskip("torch")
    plugins._ensure_plugins_loaded()
    trainer = plugins.get_trainer("yolov5")
    # yolov5 prints the first of 300 epochs as "0/299".
    assert trainer.epoch_base == 0

    launched = {}
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: launched.setdefault("cmd", cmd))
    trainer.train({
        "img_size": 320, "batch_size": 4, "epochs": 3, "data_yaml": "p/config.yaml",
        "weights": "yolov5n.pt", "project_dir": "p", "stop_file": "p/.stop", "device": "cpu",
    })
    cmd = launched["cmd"]
    assert cmd[0] == sys.executable
    assert Path(cmd[1]).name == "train_yolov5.py"
    assert cmd[cmd.index("--weights") + 1] == "yolov5n.pt"
    assert cmd[cmd.index("--stop-file") + 1] == "p/.stop"
    assert cmd[cmd.index("--device") + 1] == "cpu"


def test_yolov5_is_offered_for_training_but_not_for_obb():
    from yoru.libs.init_train import (
        MODEL_FAMILY_CONFIG,
        OBB_CAPABLE_YOLO_VERSIONS,
        init_train,
    )

    assert "YOLOv5" in MODEL_FAMILY_CONFIG["YOLO"]["versions"]
    assert "YOLOv5" not in OBB_CAPABLE_YOLO_VERSIONS
    m_dict = {}
    init_train(m_dict=m_dict)
    assert "yolov5s.pt" in m_dict["weight_list"]


# ---------------------------------------------------------------------------
# Channel order: BGR as in v1, RGB on request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env,expected",
    [(None, False), ("", False), ("0", False), ("off", False),
     ("1", True), ("true", True), ("Yes", True), ("ON", True),
     # A typo keeps v1's behaviour rather than changing every detection.
     ("rbg", False)],
)
def test_the_environment_variable_decides_the_channel_order(monkeypatch, env, expected):
    if env is None:
        monkeypatch.delenv(v5det.RGB_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(v5det.RGB_ENV_VAR, env)
    assert v5det.rgb_input_requested() is expected


def test_an_explicit_setting_outranks_the_environment(monkeypatch):
    monkeypatch.setenv(v5det.RGB_ENV_VAR, "1")
    assert v5det.rgb_input_requested(False) is False
    monkeypatch.setenv(v5det.RGB_ENV_VAR, "0")
    assert v5det.rgb_input_requested(True) is True


class _Recorder:
    """Stands in for AutoShape: records the frame it is given, detects nothing."""

    class _NoBoxes:
        def cpu(self):
            return self

        def tolist(self):
            return []

    def __call__(self, image):
        self.seen = image
        return types.SimpleNamespace(xyxy=[self._NoBoxes()])


def _detector_with(rgb):
    detector = v5det.YOLOv5Detector()
    detector._model, detector._names, detector._rgb = _Recorder(), {}, rgb
    return detector


def test_frames_go_in_as_bgr_by_default():
    np = pytest.importorskip("numpy")
    frame = np.dstack([np.full((4, 6), c, np.uint8) for c in (10, 20, 30)])  # B, G, R
    detector = _detector_with(rgb=False)
    detector.detect(frame)
    assert detector._model.seen is frame


def test_frames_go_in_as_rgb_when_asked():
    np = pytest.importorskip("numpy")
    bgra = np.dstack([np.full((4, 6), c, np.uint8) for c in (10, 20, 30, 255)])
    detector = _detector_with(rgb=True)
    detector.detect(bgra)
    assert detector._model.seen.shape == (4, 6, 3)
    assert detector._model.seen[0, 0].tolist() == [30, 20, 10]  # R, G, B; alpha dropped


def test_a_grey_frame_is_left_alone_in_rgb_mode():
    np = pytest.importorskip("numpy")
    grey = np.zeros((4, 6), np.uint8)
    detector = _detector_with(rgb=True)
    detector.detect(grey)
    assert detector._model.seen is grey


# ---------------------------------------------------------------------------
# A real YOLOv5 model: the same detections as YORU v1
# ---------------------------------------------------------------------------


def _yolov5_weights(repo_root: Path):
    """A YOLOv5 checkpoint to test with: $YORU_YOLOV5_WEIGHTS, else ./yolov5s.pt."""
    env = os.environ.get("YORU_YOLOV5_WEIGHTS", "")
    for candidate in (env, repo_root / "yolov5s.pt"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


@pytest.mark.slow
def test_detections_match_yoru_v1(repo_root, tmp_path, monkeypatch):
    """The plugin must give exactly what v1's ``torch.hub.load`` path gave."""
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    weights = _yolov5_weights(repo_root)
    if weights is None:
        pytest.skip("set YORU_YOLOV5_WEIGHTS to a YOLOv5 checkpoint to run this")
    from yoru.libs.device import resolve_device

    # Where the plugin will run (YORU_DEVICE is honoured).  v1 ran on yolov5's
    # default -- CUDA:0, else the CPU -- so compare there, or on the CPU.
    device = resolve_device("auto")
    if device not in ("cuda", "cpu"):
        pytest.skip(f"compared on CUDA:0 or the CPU, not {device}")
    # Renamed so that nothing can be read off the name, as for a v1 best.pt.
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(weights.read_bytes())
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    # The default must be v1's BGR whatever the machine has set.
    monkeypatch.delenv(v5det.RGB_ENV_VAR, raising=False)
    image = cv2.imread(str(yolov5_support.YOLOV5_DIR / "data" / "images" / "zidane.jpg"))

    def boxes(detector, frame):
        return [
            [d["x1"], d["y1"], d["x2"], d["y2"], d["conf"], d["class_id"]]
            for d in detector.detect(frame)
        ]

    detector = plugins.get_detector("auto", str(ckpt))
    assert type(detector).__name__ == "YOLOv5Detector"
    got = boxes(detector, image)
    got_rgb = boxes(plugins.get_detector("auto", str(ckpt), rgb_input=True), image)
    # Loading must not have pinned the process to a device, nor left the
    # yolov5 directory on the import path.
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == cuda_env
    assert str(yolov5_support.YOLOV5_DIR) not in sys.path

    # What YORU v1's YOLOv5Wrapper did, BGR frame included.  Naming the CPU
    # makes yolov5's select_device() hide CUDA from this process; undo that.
    hub_device = {"device": "cpu"} if device == "cpu" else {}
    try:
        v1 = torch.hub.load(
            str(yolov5_support.YOLOV5_DIR), "custom", path=str(ckpt), source="local",
            **hub_device,
        )
    finally:
        if cuda_env is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_env
    expected = v1(image).xyxy[0].cpu().tolist()
    # RGB mode is v1's model handed the channel-swapped frame, nothing more.
    expected_rgb = v1(image[..., ::-1]).xyxy[0].cpu().tolist()

    assert detector.names == {int(k): v for k, v in dict(v1.names).items()}
    for mine, v1s in ((got, expected), (got_rgb, expected_rgb)):
        assert len(mine) == len(v1s)
        for g, e in zip(mine, v1s):
            assert g == pytest.approx(e, abs=1e-4)
    # And the two orders really do differ, or this would prove nothing.
    assert got != got_rgb
