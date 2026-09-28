from pathlib import Path
import pytest

def test_required_directories_exist(repo_root: Path):
    for p in ("yoru", "config", "trigger_plugins"):
        assert (repo_root / p).exists(), f"missing: {p}/"


def test_labelimg_bundled_directory_exists(repo_root: Path):
    labelimg_dir = repo_root / "yoru" / "labelimg"
    assert labelimg_dir.exists(), "yoru/labelimg/ not found (bundled annotation tool missing)"
    assert (labelimg_dir / "labelimg.py").exists(), "yoru/labelimg/labelimg.py not found"
    assert (labelimg_dir / "libs").exists(), "yoru/labelimg/libs/ not found"
    assert (labelimg_dir / "libs" / "yolo_io.py").exists(), "yoru/labelimg/libs/yolo_io.py not found"

def test_config_template_exists(repo_root: Path):
    cfg = repo_root / "config" / "template.yaml"
    assert cfg.exists(), "config/template.yaml not found"

def test_bundled_yolov5_is_present(repo_root: Path):
    """The bundled ultralytics/yolov5 code must stay.

    It is the only code that can load a YOLOv5 checkpoint -- every model YORU
    v1 trained -- and the code YOLOv5 training runs.  The ultralytics package
    is no replacement: it refuses these checkpoints, and its yolov5*u models
    are a different network.  v2.0 beta shipped without this copy and could
    not use a single v1 model.
    """
    yv5 = repo_root / "yoru" / "libs" / "yolov5"
    for rel in (
        "train.py",
        "val.py",
        "hubconf.py",
        "requirements.txt",
        "LICENSE",
        "models/common.py",
        "models/yolo.py",
        "models/experimental.py",
        "utils/general.py",
        "utils/dataloaders.py",
        "data/hyps/hyp.scratch-low.yaml",
        "models/yolov5s.yaml",
    ):
        assert (yv5 / rel).is_file(), f"yoru/libs/yolov5/{rel} missing (bundled YOLOv5)"
