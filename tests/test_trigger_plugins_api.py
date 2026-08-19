import inspect
from pathlib import Path
import importlib
import pytest

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_PATHS = [
    p for p in (ROOT / "trigger_plugins").glob("*.py")
    if p.name not in ("__init__.py",) and not p.name.startswith("_")
]

@pytest.mark.parametrize("plugin_path", PLUGIN_PATHS)
def test_trigger_plugin_api_shape(plugin_path, fake_serial):
    mod = importlib.import_module(f"trigger_plugins.{plugin_path.stem}")
    # Expect a class named 'trigger_condition' with a 'trigger' method
    assert hasattr(mod, "trigger_condition"), "missing class 'trigger_condition'"
    cls = getattr(mod, "trigger_condition")
    assert inspect.isclass(cls), "trigger_condition must be a class"
    assert hasattr(cls, "trigger") and callable(getattr(cls, "trigger")), "trigger_condition.trigger must be callable"
    # The trigger signature in current plugins is (self, tri_cl, in_cl, arduino, results, now)
    sig = inspect.signature(cls.trigger)
    assert len(sig.parameters) >= 6, f"trigger(...) should accept >= 6 params, got: {sig}"

@pytest.mark.parametrize("plugin_path", PLUGIN_PATHS)
def test_trigger_plugin_constructor_takes_m_dict(plugin_path, fake_serial):
    """yoru.libs.trigger builds every plugin as ``trigger_condition(m_dict)``.

    Three bundled plugins previously declared ``__init__(self)`` and therefore
    could not be instantiated at all; the API check above did not catch it
    because it only inspected ``trigger()``.
    """
    mod = importlib.import_module(f"trigger_plugins.{plugin_path.stem}")
    sig = inspect.signature(mod.trigger_condition.__init__)
    params = list(sig.parameters.values())[1:]  # drop self
    required = [p.name for p in params if p.default is inspect.Parameter.empty]
    assert len(required) <= 1, (
        f"trigger_condition(m_dict) must be constructible, but "
        f"{plugin_path.name} requires {required}"
    )
    if params:
        assert params[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ), f"{plugin_path.name}: first parameter must accept m_dict positionally"
