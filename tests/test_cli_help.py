import subprocess
import sys
import pytest

@pytest.mark.gui
def test_cli_help_runs():
    proc = subprocess.run([sys.executable, "-m", "yoru", "--help"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=60)
    assert proc.returncode == 0, f"--help failed\nSTDERR:\n{proc.stderr[:500]}"
