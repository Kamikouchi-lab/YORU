# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Locating the project data directories: ``web/``, ``config/``, ``trigger_plugins/``.

These sit next to the ``yoru`` package rather than inside it, because YORU is
run from a checkout - trigger plugins in particular are meant to be read and
edited by the user.  Previously each caller resolved them differently
(``yoru.libs.trigger`` walked up from ``__file__`` while
``yoru.config_creator_GUI`` globbed a path relative to the current working
directory), so the two disagreed whenever YORU was started from another
directory.  Everything now goes through here.
"""

import sys
from pathlib import Path

# .../yoru/libs/paths.py -> .../yoru/libs -> .../yoru -> project root
_PACKAGE_DIR = Path(__file__).resolve().parent.parent
_INSTALL_ROOT = _PACKAGE_DIR.parent

_DATA_DIRS = ("trigger_plugins", "config", "web")


def project_root() -> Path:
    """Directory holding ``web/``, ``config/`` and ``trigger_plugins/``.

    Prefers the current working directory when it looks like a YORU checkout -
    that is how the launcher starts its sub-processes, and it lets a user keep
    their own trigger plugins beside the project.  Falls back to the directory
    containing the package.
    """
    cwd = Path.cwd()
    if all((cwd / d).is_dir() for d in _DATA_DIRS):
        return cwd
    return _INSTALL_ROOT


def web_dir() -> Path:
    return project_root() / "web"


def config_dir() -> Path:
    return project_root() / "config"


def trigger_plugins_dir() -> Path:
    return project_root() / "trigger_plugins"


def list_trigger_plugins() -> list:
    """Sorted module names of the available trigger plugins."""
    directory = trigger_plugins_dir()
    if not directory.is_dir():
        return []
    return sorted(
        p.stem for p in directory.glob("*.py") if not p.stem.startswith("_")
    )


def ensure_importable() -> None:
    """Put the project root on ``sys.path``.

    ``trigger_plugins`` is a top-level package outside ``yoru``, so importing
    ``trigger_plugins.<name>`` only works when the project root is importable.
    """
    root = str(project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
