import os
import re
import sys
import subprocess
import pytest

def _run(args, timeout=60):
    env = dict(os.environ)
    # Make stdout decoding stable on Windows
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run([sys.executable, "-m", "yoru", *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, timeout=timeout, env=env)

def test_top_level_help_exits_zero_and_mentions_yoru():
    proc = _run(["--help"])
    assert proc.returncode == 0, f"--help failed\nSTDERR:\n{proc.stderr}"
    assert "YORU" in proc.stdout

def test_version_flag_prints_semver_like_tag():
    proc = _run(["-V"])
    assert proc.returncode == 0, f"-V failed\nSTDERR:\n{proc.stderr}"
    # Accept semantic-ish tags like '1.2.3' or '0+unknown' fallback
    assert re.search(r"(\d+\.\d+\.\d+|0\+unknown)", proc.stdout), proc.stdout

def test_build_parser_defaults_and_help_without_gui_import(monkeypatch):
    # Import cli in-process and ensure parsing help does not import yoru.app
    import sys as _sys
    assert "yoru.app" not in _sys.modules
    from yoru.cli import build_parser
    parser = build_parser()
    # Default command should be GUI (but help shouldn't launch anything)
    ns = parser.parse_args([])
    assert getattr(ns, "command", None) == "gui"
    with pytest.raises(SystemExit):  # argparse exits on --help
        parser.parse_args(["--help"])
    # Still no GUI import happened as a side-effect
    assert "yoru.app" not in _sys.modules