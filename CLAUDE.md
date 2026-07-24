# YORU — Claude Code project guide

YORU is a Windows Python app (DearPyGui GUIs + Ultralytics YOLO) for real-time
animal-behavior detection.

## Runtime logs — check these first when something fails

Runtime errors and work logs are written to a rotating log file under the
user's home directory:

- **`~/.yoru/logs/yoru.log`** — every caught GUI error (via
  `GuiErrorMixin._report_error`), subprocess failures reported by
  `yoru/app.py`, and CLI-level failures.

When investigating **any** YORU failure, read this file first: it contains the
full traceback even when the on-screen popup or console only shows a summary.

- The user directory can be relocated with the `YORU_HOME` environment variable
  (defaults to `~/.yoru`).
- Logs rotate at 5 MB and keep 3 backups (`yoru.log`, `yoru.log.1`, …).

## User state

- **`~/.yoru/condition_file_log.json`** — remembers the last-used condition
  file. Migrated automatically (once) from the old
  `./logs/condition_file_log.json`.

## Paths / logging helpers

`yoru/libs/user_paths.py` centralizes these paths and configures logging:
`get_yoru_home()`, `get_log_dir()`, `get_log_file()`, `get_state_file()`,
`setup_logging()`, `log_exception(context, exc)`, `log_message(msg, level)`.
It has **no** GUI/OpenCV dependency, so it is safe to import early and to
unit-test headlessly.

## Environment

The primary interpreter is the conda `yoru` env
(`miniconda3\envs\yoru\python.exe`). Bare `python` on this machine is a broken
Windows Store stub — always launch subprocesses with `sys.executable`.
