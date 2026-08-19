# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

# yoru/cli.py
from __future__ import annotations
import argparse
from pathlib import Path

def _get_version() -> str:
    try:
        from . import __version__
        return str(__version__)
    except Exception:
        try:
            from importlib.metadata import version
            return version("yoru")
        except Exception:
            return "0+unknown"

def build_parser() -> argparse.ArgumentParser:
    examples = r"""
Examples:
  # Launch GUI with default config
  python -m yoru

  # Show top-level help (does not launch GUI)
  python -m yoru --help

  # Show GUI subcommand help
  python -m yoru gui --help
"""
    parser = argparse.ArgumentParser(
        prog="yoru",
        description="YORU - Your Optimal Recognition Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=examples,
    )
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s { _get_version() }")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # ---- gui ----
    p_gui = sub.add_parser(
        "gui",
        help="Launch the YORU GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p_gui.add_argument("--config", default=None,
                       help="Condition YAML to load at startup "
                            "(default: the last-used file)")
    p_gui.set_defaults(func=_cmd_gui)

    # ここがポイント：引数なしのときは GUI を既定動作にする
    parser.set_defaults(func=_cmd_gui, command="gui", config=None)
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # argparse が --help を処理した場合はここに来ない（0終了）
    # 既定（引数なし）は func=_cmd_gui が入っている
    return int(bool(args.func(args)))  # 成功は 0/None, 失敗は非0

# -------------------------
# Subcommand impls
# -------------------------

def _cmd_gui(args) -> int:
    """Launch GUI. Keep imports lazy so '--help' stays fast."""
    try:
        from yoru.app import main as app_main   # ←あなたの既存 GUI エントリ
    except Exception as e:
        print(f"[yoru] failed to import yoru.app.main: {e}")
        return 1

    cfg = getattr(args, "config", None)
    try:
        # Decide how to call app_main by inspecting its signature.  Catching
        # TypeError instead would misread an error raised *inside* the GUI as a
        # signature mismatch and launch the GUI a second time.
        if cfg is not None and _accepts_config(app_main):
            return int(bool(app_main(cfg)))
        return int(bool(app_main()))
    except SystemExit as se:
        code = se.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(f"[yoru] {code}")  # SystemExit("message") means failure
        return 1
    except Exception as e:
        print(f"[yoru] GUI crashed: {e}")
        return 1


def _accepts_config(func) -> bool:
    """True if *func* takes a leading positional parameter for the config path."""
    import inspect

    try:
        params = list(inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return False
    return bool(params) and params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
