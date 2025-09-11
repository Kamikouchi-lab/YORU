from pathlib import Path
import importlib
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGINS = [
    f"trigger_plugins.{p.stem}"
    for p in (ROOT / "trigger_plugins").glob("*.py")
    if p.name not in ("__init__.py",) and not p.name.startswith("_")
]

@pytest.mark.parametrize("modname", PLUGINS)
def test_plugins_are_importable(modname, fake_serial):
    mod = importlib.import_module(modname)
    # basic sanity: module object exists
    assert hasattr(mod, "__doc__")
