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

def test_yolov5_is_vendored(repo_root: Path):
    """Upstream YOLOv5 is carried in-tree, not installed.

    It cannot come from a pip package: a checkpoint trained with YORU v1
    unpickles into these exact classes, under these exact module names, and
    ultralytics refuses the file outright.  The __init__.py is YORU's, and is
    what binds 'models' and 'utils' for that unpickling.
    """
    yv5 = repo_root / "yoru" / "libs" / "yolov5"
    assert yv5.exists(), "yoru/libs/yolov5/ not found (vendored YOLOv5 missing)"
    for rel in ("__init__.py", "train.py", "models/yolo.py", "models/common.py",
                "models/experimental.py", "utils/general.py", "utils/augmentations.py"):
        assert (yv5 / rel).exists(), f"yoru/libs/yolov5/{rel} not found"
    # The pretrained architectures the training GUI offers.
    for size in "nsmlx":
        assert (yv5 / "models" / f"yolov5{size}.yaml").exists(), size
