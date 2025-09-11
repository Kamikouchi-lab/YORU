import importlib
import importlib.util
import pytest

def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None

def test_import_yoru():
    mod = importlib.import_module("yoru")
    assert mod is not None

def test_has_main_module_or_skip():
    if not module_exists("yoru.__main__"):
        pytest.skip("__main__ not provided")
    mod = importlib.import_module("yoru.__main__")
    assert mod is not None
