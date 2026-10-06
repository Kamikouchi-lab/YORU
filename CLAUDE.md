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

## Bundled YOLOv5 — do not remove

`yoru/libs/yolov5/` is a vendored copy of ultralytics/yolov5 (the copy YORU
v1.1.2 shipped, with v1's patches). It is the only code that can load a YOLOv5
checkpoint — every model trained with YORU v1, and the models of the YORU
paper — and it runs YOLOv5 training. The `ultralytics` package is **not** a
substitute: it refuses these checkpoints, and its `yolov5*u` models are a
different network. Never route YOLOv5 to ultralytics or offer `yolov5*u`.

- Entry points: `yoru/libs/yolov5_support.py` (path handling),
  `yoru/libs/plugins/yolov5_detector.py`, `yoru/libs/plugins/yolov5_trainer.py`,
  `yoru/libs/train_yolov5.py` (runs the bundled `train.py`).
- A YOLOv5 checkpoint is recognised by its pickle contents
  (`plugins._sniff_checkpoint`), whatever the file is called.
- Inference must stay identical to v1 by default (BGR frame passed straight
  to `AutoShape`; RGB only when `YORU_YOLOV5_RGB=1` or `rgb_input=True`;
  NMS at 0.25 with any higher GUI threshold applied after it, as v1 did —
  fp16 score ties make NMS at the higher threshold keep different boxes);
  `tests/test_yolov5_backend.py::test_detections_match_yoru_v1` checks both
  against v1's `torch.hub.load` path.

## External API and YORU-Tracker

`docs/external_api.md` lists the names sister applications (YORU-Tracker,
`../YORU-Tracker`) import: `get_detector`, `list_detector_backends`,
`DETECTION_COLUMNS`, `obb_of`, `yoru.libs.obb`, `yoru.libs.camera.open_camera`,
the GUI primitives and `user_paths`. `tests/test_public_api.py` pins them —
renaming one is a breaking change, not a refactor.

YORU never imports `yoru_tracker` (the same test enforces it) and its GUI gets
no tracker-specific branches. Tracking, identity, trajectories and the tracking
GUI live in YORU-Tracker; only general detector / OBB / camera / GUI primitives
belong here.

## Environment

The primary interpreter is the conda `yoru` env
(`miniconda3\envs\yoru\python.exe`). Bare `python` on this machine is a broken
Windows Store stub — always launch subprocesses with `sys.executable`.
