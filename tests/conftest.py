from __future__ import annotations
import sys, types
from pathlib import Path
import pytest

# ---------------------------
# Repo root & import path
# ---------------------------
@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True, scope="session")
def add_repo_to_syspath(repo_root: Path):
    sys.path.insert(0, str(repo_root))

# ---------------------------
# Hardware dependency stubs (pyserial, libs.arduino, nidaqmx)
# Implemented directly as a fixture (no contextlib.contextmanager) to avoid
# 'generator didn't stop' issues. We save/restore previous sys.modules entries.
# ---------------------------
@pytest.fixture
def fake_serial():
    prev = {}  # name -> previous module or None if absent

    def _put(name: str, mod):
        prev[name] = sys.modules.get(name)
        sys.modules[name] = mod

    # ---- serial (+ tools.list_ports) as a package ----
    serial_mod = types.ModuleType("serial"); serial_mod.__path__ = []
    class SerialException(Exception): ...
    class Serial:
        def __init__(self, *a, **kw): pass
        def write(self, *a, **kw): return 0
        def close(self): pass
    serial_mod.Serial = Serial
    serial_mod.SerialException = SerialException
    tools_mod = types.ModuleType("serial.tools"); tools_mod.__path__ = []
    list_ports_mod = types.ModuleType("serial.tools.list_ports")
    list_ports_mod.comports = lambda: []

    # ---- libs.arduino package stub ----
    libs_pkg = types.ModuleType("libs"); libs_pkg.__path__ = []
    libs_arduino_mod = types.ModuleType("libs.arduino")

    # ---- nidaqmx package + submodules ----
    nidaqmx_mod = types.ModuleType("nidaqmx"); nidaqmx_mod.__path__ = []
    nidaqmx_constants = types.ModuleType("nidaqmx.constants")
    class AcquisitionType:
        FINITE_SAMPLES = 0
        CONTINUOUS_SAMPLES = 1
    class Edge:
        RISING = 0
        FALLING = 1
    class LineGrouping:
        CHAN_PER_LINE = 0
        ONE_CHANNEL_FOR_ALL_LINES = 1
    nidaqmx_constants.AcquisitionType = AcquisitionType
    nidaqmx_constants.Edge = Edge
    nidaqmx_constants.LineGrouping = LineGrouping

    nidaqmx_errors = types.ModuleType("nidaqmx.errors")
    class DaqError(Exception): ...
    nidaqmx_errors.DaqError = DaqError

    # install stubs
    _put("serial", serial_mod)
    _put("serial.tools", tools_mod)
    _put("serial.tools.list_ports", list_ports_mod)
    _put("libs", libs_pkg)
    _put("libs.arduino", libs_arduino_mod)
    _put("nidaqmx", nidaqmx_mod)
    _put("nidaqmx.constants", nidaqmx_constants)
    _put("nidaqmx.errors", nidaqmx_errors)

    try:
        yield
    finally:
        # restore previous modules
        for name, old in prev.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old

# ---------------------------
# Optional compact progress line (safe)
# ---------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--progress",
        action="store",
        default="auto",
        choices=("on", "off", "auto"),
        help="Show a compact progress line (on/off/auto). Default: auto (TTY only).",
    )

def _progress_enabled(config) -> bool:
    mode = config.getoption("--progress")
    if mode == "off":
        return False
    if mode == "on":
        return True
    tr = config.pluginmanager.getplugin("terminalreporter")
    if not tr:
        return False
    is_tty = getattr(getattr(tr, "_tw", None), "isatty", False)
    if config.pluginmanager.hasplugin("xdist"):
        return False
    return bool(is_tty)

def pytest_configure(config):
    if _progress_enabled(config):
        plugin = _YoruProgressPlugin(config)
        config.pluginmanager.register(plugin, "yoru-progress")

class _YoruProgressPlugin:
    def __init__(self, config):
        self.config = config
        self.tr = config.pluginmanager.getplugin("terminalreporter")
        self.total = 0
        self.count = 0

    def pytest_collection_finish(self, session):
        self.total = len(getattr(session, "items", []))
        self._draw(force=True)

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        self.count += 1
        self._draw()
        if self.count >= self.total > 0:
            self._emit(None, final=True)

    def pytest_sessionfinish(self, session, exitstatus):
        self._emit(None, final=True)

    def _emit(self, line: str | None, final: bool = False):
        # try terminalreporter first
        if self.tr:
            try:
                if line is not None and hasattr(self.tr, "rewrite"):
                    self.tr.rewrite(line, soft=True)
                elif line is not None:
                    self.tr.write("\\r" + line)
                if final:
                    self.tr.write("\\n")
                return
            except Exception:
                pass
        # fallback to stderr
        if line is not None:
            sys.stderr.write("\\r" + line)
            sys.stderr.flush()
        if final:
            sys.stderr.write("\\n")
            sys.stderr.flush()

    def _draw(self, force: bool = False):
        if self.total <= 0 and not force:
            return
        total = max(self.total, 1)
        width = 30
        filled = int(width * self.count / total)
        bar = "#" * filled + "-" * (width - filled)
        percent = int(100 * self.count / total)

        p = f = s = 0
        stats = getattr(getattr(self, "tr", None), "stats", None)
        if isinstance(stats, dict):
            p = len(stats.get("passed", []))
            f = len(stats.get("failed", [])) + len(stats.get("error", [])) + len(stats.get("errors", []))
            s = len(stats.get("skipped", []))
        line = f"[progress] [{bar}] {percent:3d}%  {self.count}/{self.total}  (✓{p} ✗{f} ~{s})"
        self._emit(line, final=False)
