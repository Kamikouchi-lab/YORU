# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

r"""Ask a running training subprocess to stop at the end of the current epoch.

Training runs in a subprocess (see ``libs/plugins/*_trainer.py``) and its only
channel back to the GUI is the stdout pipe, so there is no way to *tell* it
anything once it has started.  Killing it is not the same as stopping it: a
process killed mid-epoch loses that epoch, and for ultralytics it also loses
the final validation pass and the ``best.pt`` copy, which only happen once the
epoch loop exits on its own.

The GUI therefore *requests* a stop instead of forcing one.  It creates an
empty file -- the stop file -- and the trainer looks for it at the one moment
where quitting costs nothing: the end of an epoch, once that epoch's
checkpoint has been written.  A file works where a signal does not.  It needs
no shared handle, it is unambiguous on Windows (there is no SIGINT to send to
a child there), and it can be created by hand for a run started from a shell::

    type nul > path\to\project\.yoru_stop_request     (Windows)
    touch      path/to/project/.yoru_stop_request        (macOS / Linux)

Whoever acts on the request deletes the file, so a stop never carries over
into the next run.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "STOP_FILENAME",
    "stop_file_for",
    "request_stop",
    "clear_stop",
    "stop_requested",
    "terminate_process_tree",
]

STOP_FILENAME = ".yoru_stop_request"


def stop_file_for(project_dir) -> Path:
    """Path of the stop file belonging to *project_dir*."""
    return Path(project_dir) / STOP_FILENAME


def request_stop(path) -> None:
    """Create the stop file: end the run after the epoch now in progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def clear_stop(path) -> None:
    """Delete the stop file if there is one.  Never raises.

    Called both before a run starts and after one ends: a request left behind
    by a crash -- or by a stop that a forced kill got to first -- would
    otherwise end the *next* run after a single epoch.
    """
    if path is None:
        return
    try:
        Path(path).unlink()
    except OSError:
        pass


def stop_requested(path) -> bool:
    """Has a stop been requested?  False for ``None`` (no stop file in use)."""
    if path is None:
        return False
    try:
        return Path(path).exists()
    except OSError:
        return False


def terminate_process_tree(proc, timeout: float = 5.0) -> None:
    """Kill *proc* and everything it started, escalating if it does not go.

    The last resort behind the cooperative stop, for when the epoch is long
    enough that waiting for it is not an option.  The trainer spawns
    dataloader workers of its own, and killing only the process YORU launched
    would leave them behind still holding the GPU.
    """
    if proc is None or proc.poll() is not None:
        return

    children = []
    try:
        import psutil

        children = psutil.Process(proc.pid).children(recursive=True)
    except Exception:
        # psutil missing, or the process died between the poll and here:
        # fall back to terminating the process YORU knows about.
        pass

    for child in children:
        try:
            child.terminate()
        except Exception:
            pass
    try:
        proc.terminate()
    except Exception:
        pass

    try:
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    for child in children:
        try:
            if child.is_running():
                child.kill()
        except Exception:
            pass
