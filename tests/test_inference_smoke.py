import os
import importlib
import subprocess
import sys
from pathlib import Path
import pytest

try:
    from PIL import Image
except Exception:
    Image = None

WEIGHTS = os.environ.get("YORU_SMOKE_WEIGHTS", "")
CMD_TPL = os.environ.get("YORU_SMOKE_CMD", "")

def _make_dummy_images(root: Path, n: int = 5, size=(64, 64)):
    root.mkdir(parents=True, exist_ok=True)
    if Image is None:
        pytest.skip("pillow not installed; pip install pillow")
    for i in range(n):
        img = Image.new("RGB", size, (i * 10 % 255, 0, 128))
        img.save(root / f"dummy_{i:02d}.png")
    return root

def _maybe_get_python_entry():
    try:
        mod = importlib.import_module("yoru.testing")
        fn = getattr(mod, "run_inference", None)
        if callable(fn): return fn
    except Exception:
        pass
    try:
        mod = importlib.import_module("yoru")
        for name in ("run_inference", "infer"):
            fn = getattr(mod, name, None)
            if callable(fn): return fn
    except Exception:
        pass
    return None

@pytest.mark.slow
def test_inference_smoke(tmp_path):
    if not WEIGHTS or not Path(WEIGHTS).exists():
        pytest.skip("Set YORU_SMOKE_WEIGHTS to a tiny .pt file to enable this test")
    images_dir = _make_dummy_images(tmp_path / "imgs")
    out_dir = tmp_path / "out"; out_dir.mkdir(parents=True, exist_ok=True)

    py_entry = _maybe_get_python_entry()
    if callable(py_entry):
        py_entry(str(images_dir), str(WEIGHTS), str(out_dir))
        assert True; return

    if CMD_TPL:
        cmd = CMD_TPL.format(images=images_dir, weights=WEIGHTS, out=out_dir)
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        assert result.returncode == 0, f"CLI failed: {result.returncode}\nSTDERR:\n{result.stderr}"
        return

    pytest.skip("No Python inference entry found and YORU_SMOKE_CMD not provided")
