from pathlib import Path
import pytest

def test_required_directories_exist(repo_root: Path):
    for p in ("yoru", "config", "trigger_plugins"):
        assert (repo_root / p).exists(), f"missing: {p}/"

def test_config_template_exists(repo_root: Path):
    cfg = repo_root / "config" / "template.yaml"
    assert cfg.exists(), "config/template.yaml not found"

def test_optional_yolov5_submodule(repo_root: Path):
    yv5 = repo_root / "libs" / "yolov5"
    if not yv5.exists():
        pytest.skip("libs/yolov5 not present (submodule not checked out)")
