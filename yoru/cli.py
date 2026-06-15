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
    p_gui.add_argument("--config", default="config/template.yaml",
                       help="Condition YAML to load at startup")
    p_gui.set_defaults(func=_cmd_gui)

    # Key point: when no arguments are given, default to launching the GUI
    parser.set_defaults(func=_cmd_gui, command="gui", config="config/template.yaml")
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # If argparse handled --help, execution never reaches here (exits with 0)
    # The default (no arguments) has func=_cmd_gui set
    return int(bool(args.func(args)))  # success is 0/None, failure is non-zero

# -------------------------
# Subcommand impls
# -------------------------

def _cmd_gui(args) -> int:
    """Launch GUI. Keep imports lazy so '--help' stays fast."""
    try:
        from yoru.app import main as app_main   # <- your existing GUI entry point
    except Exception as e:
        print(f"[yoru] failed to import yoru.app.main: {e}")
        return 1

    cfg = getattr(args, "config", None)
    try:
        # Call safely in case the signature of app_main is unknown
        if cfg is not None:
            # Use this if app_main(config_path=...) is supported
            try:
                return int(bool(app_main(cfg)))  # variant passing it as a positional argument
            except TypeError:
                try:
                    return int(bool(app_main(config=cfg)))  # variant passing it as a keyword argument
                except TypeError:
                    pass
        # No-argument variant (works for most implementations)
        return int(bool(app_main()))
    except SystemExit as se:
        return int(se.code or 0)
    except Exception as e:
        print(f"[yoru] GUI crashed: {e}")
        return 1
