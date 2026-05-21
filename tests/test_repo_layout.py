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

def test_optional_yolov5_submodule(repo_root: Path):
    yv5 = repo_root / "yoru" / "libs" / "yolov5"
    if not yv5.exists():
        pytest.skip("yoru/libs/yolov5 not present (submodule not checked out)")
