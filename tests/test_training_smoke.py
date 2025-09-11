import os
import importlib
import subprocess
from pathlib import Path
import pytest

DATA = os.environ.get("YORU_SMOKE_DATA", "")
CMD_TPL = os.environ.get("YORU_TRAIN_CMD", "")

def _maybe_get_python_entry():
    try:
        mod = importlib.import_module("yoru.testing")
        fn = getattr(mod, "run_training", None)
        if callable(fn): return fn
    except Exception:
        pass
    try:
        mod = importlib.import_module("yoru")
        for name in ("run_training", "train"):
            fn = getattr(mod, name, None)
            if callable(fn): return fn
    except Exception:
        pass
    return None

@pytest.mark.slow
def test_training_smoke(tmp_path):
    if not DATA or not Path(DATA).exists():
        pytest.skip("Set YORU_SMOKE_DATA to a small training dataset to enable this test")
    out_dir = tmp_path / "train_out"; out_dir.mkdir(parents=True, exist_ok=True)

    py_entry = _maybe_get_python_entry()
    if callable(py_entry):
        py_entry(str(DATA), str(out_dir), epochs=1, device="cpu")
        artifacts = list(out_dir.glob("**/*"))
        assert artifacts, "No training artifacts found in out_dir"
        return

    if CMD_TPL:
        cmd = CMD_TPL.format(data=DATA, out=out_dir, epochs=1)
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=1800)
        assert result.returncode == 0, f"CLI training failed\nSTDERR:\n{result.stderr}"
        artifacts = list(out_dir.glob("**/*"))
        assert artifacts, "No training artifacts found in out_dir"
        return

    pytest.skip("No Python training entry found and YORU_TRAIN_CMD not provided")
