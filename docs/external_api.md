# YORU's external API

YORU is also a library. Sister applications — today
[YORU-Tracker](https://github.com/Kamikouchi-lab/YORU-Tracker) — load
detectors, read detections and draw their windows through the names listed on
this page instead of copying YORU's code.

These names are the **supported surface**. Changing one of them is a breaking
change for every application built on YORU: it needs a major version, or a
deprecation period during which the old name keeps working.
`tests/test_public_api.py` pins each of them, so a change shows up as a failing
test rather than as a broken sister application.

Anything not listed here — the GUI classes, `m_dict` keys, the realtime worker
processes — is internal and may change at any time.

## The dependency rule

```text
YORU-Tracker  ──depends on──▶  YORU
YORU          ──never imports──▶  YORU-Tracker
```

YORU must start, run and pass its tests without any sister application
installed. Nothing under `yoru/` imports `yoru_tracker`, and the normal YORU
GUI carries no tracker-specific branches; `tests/test_public_api.py` checks the
first of these. A feature belongs in YORU only when YORU's own closed-loop
detection role needs it, or when it is a genuinely general detector / OBB /
camera / GUI primitive. Tracking policy, identity logic and the tracking GUI
belong to YORU-Tracker.

## Detectors — `yoru.libs.plugins`

| Name | What it is |
|------|------------|
| `get_detector(backend, model_path, conf_thresh=DEFAULT_CONF_THRESH, iou_thresh=DEFAULT_IOU_THRESH, **kwargs)` | Load a model and return a `DetectorBase`. `backend="auto"` identifies the model from the file (a YOLOv5 checkpoint by its contents, whatever it is called). |
| `list_detector_backends()` | `["auto", ...]` — the backend names `get_detector` accepts on this machine. |
| `DEFAULT_CONF_THRESH`, `DEFAULT_IOU_THRESH` | The thresholds every backend applies unless told otherwise. |

## One detection — `yoru.libs.detector_base`

`DetectorBase.detect(image)` takes a BGR `numpy` image and returns a list of
dicts:

| Key | Always present | Meaning |
|-----|----------------|---------|
| `x1`, `y1`, `x2`, `y2` | yes | Upright box, pixels. |
| `conf` | yes | Confidence. |
| `class_id`, `class_name` | yes | Class. |
| `cx`, `cy`, `w`, `h`, `angle` | OBB models only | The oriented box; `angle` in radians. |

| Name | What it is |
|------|------------|
| `DetectorBase` | `load(model_path, **kwargs)`, `detect(image)`, `names` (`{class_id: class_name}`). |
| `obb_of(detection)` | `(cx, cy, w, h, angle)` for any detection dict — the upright box with angle 0 when the model does not predict rotation. Use this instead of reading `cx..angle` directly. |
| `DETECTION_COLUMNS` | Column order of a detection row and of the realtime `*_detect.csv` header. |
| `detection_row(detection, total_time)` | One detection as a row in that order. |

## Oriented boxes — `yoru.libs.obb`

Everything in `__all__`: `normalize_angle`, `obb_corners`, `corners_to_obb`,
`obb_to_aabb`, `aabb_to_obb`, `point_in_obb`, `rotate_points`, `is_rotated`,
`box_axes`, `resize_corner`, `ANGLE_SNAP`. Pure Python, no OpenCV.

An OBB is `(cx, cy, w, h, angle)`: `angle` in radians from the image +x axis
towards +y, folded into `[-pi/2, pi/2)`. A rectangle's angle is only defined
modulo pi, so it is **not** a heading — the two ends of an animal are not told
apart.

## Cameras — `yoru.libs.camera`

| Name | What it is |
|------|------------|
| `open_camera(src, width=None, height=None, fps=None, *, settings_dialog=False)` | A `cv2.VideoCapture` opened the way YORU's realtime capture opens it (preferred backend per platform, then any; one-frame buffer). Raises `RuntimeError` if it cannot be opened. |

## GUI primitives — `yoru.gui_base`, `yoru.gui_layout`, `yoru.libs.gui_error`

| Name | What it is |
|------|------------|
| `gui_base.apply_default_theme()` | Bind the YORU DearPyGui theme; returns the theme. |
| `gui_base.process_frame(frame, preview_size, *, v_flip=False, h_flip=False)` | Letterbox a frame into a square canvas. |
| `gui_base.frame_to_data_rgb(frame)`, `gui_base.frame_to_data_rgba(frame)` | BGR frame → DearPyGui texture data. |
| `gui_layout.GuiSession(name, title, width, height, docking=False, ...)` | Viewport sized to the screen, Japanese-capable font, the Window menu, remembered size. `begin()`, `window_kwargs()`, `add_layout_menu()`, `finish()`, `content_region()`. |
| `gui_error.GuiErrorMixin` | `_report_error(context, exc)`: log the traceback and show it in a popup. |

Share primitives, not screens: a sister application builds its own windows
from these and never subclasses one of YORU's `*_GUI` classes.

## Paths and logs — `yoru.libs.user_paths`

`get_yoru_home()`, `get_log_dir()`, `get_log_file()`, `setup_logging()`,
`log_exception(context, exc)`, `log_message(message, level)`. A sister
application writes its own log file into `get_log_dir()` (YORU-Tracker writes
`yoru_tracker.log`) so both are found in one place, `~/.yoru/logs/`.

## Version

`yoru.__version__`. A sister application declares the YORU versions it supports
in its own dependencies (for example `yoru>=2.0.0b3,<3`).
