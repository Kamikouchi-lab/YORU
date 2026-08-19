"""Guards against source-level breakage that no other test would notice.

``yoru/app.py`` once shipped with a syntax error introduced by a bad merge.
Nothing in the suite imported or parsed it, so the whole GUI entry point was
broken while the tests stayed green.
"""

import importlib
from pathlib import Path

import pytest

SOURCE_DIRS = ("yoru", "trigger_plugins")


def _python_files(repo_root: Path):
    for d in SOURCE_DIRS:
        yield from sorted((repo_root / d).rglob("*.py"))


def test_every_source_file_parses(repo_root: Path):
    failures = []
    for path in _python_files(repo_root):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as e:
            failures.append(f"{path.relative_to(repo_root)}:{e.lineno}: {e.msg}")
    assert not failures, "syntax errors found:\n" + "\n".join(failures)


def test_app_entry_point_is_importable():
    """The launcher module must import and expose main()."""
    pytest.importorskip("webview", reason="pywebview not installed")
    mod = importlib.import_module("yoru.app")
    assert callable(mod.main)


def test_cli_entry_point_is_importable():
    """The CLI must import without pulling in the GUI stack."""
    import sys

    mod = importlib.import_module("yoru.cli")
    assert callable(mod.main)
    parser = mod.build_parser()
    ns = parser.parse_args([])
    assert ns.command == "gui"
    # --config defaults to None so the remembered condition file is preserved.
    assert ns.config is None
    assert "yoru.app" not in sys.modules


def test_version_is_consistent(repo_root: Path):
    """yoru.__version__ and the pyproject version must not drift apart."""
    import re

    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    assert m, "no version in pyproject.toml"
    from yoru import __version__

    assert __version__ == m.group(1), (
        f"yoru.__version__ ({__version__}) != pyproject version ({m.group(1)})"
    )
