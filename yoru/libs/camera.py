# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Opening a camera the way YORU does, for anything that needs a live feed.

The realtime GUI's capture process and sister applications such as
YORU-Tracker open cameras through this one function, so a camera that works in
one works in the other: the same per-platform backend preference, the same
fallback, and the one-frame buffer a closed-loop experiment needs.

Only OpenCV is imported here -- no GUI toolkit, no recorder, no shared state.
"""

import sys

import cv2

__all__ = ["open_camera", "preferred_backend"]


def preferred_backend():
    """The OpenCV capture API YORU tries first on this platform."""
    return {
        "win32": cv2.CAP_DSHOW,
        "darwin": cv2.CAP_AVFOUNDATION,
        "linux": cv2.CAP_V4L2,
    }.get(sys.platform, cv2.CAP_ANY)


def open_camera(src, width=None, height=None, fps=None, *, settings_dialog=False):
    """Open camera *src* and return the ``cv2.VideoCapture``.

    The platform's preferred backend is tried first and ``CAP_ANY`` second.
    *width*, *height* and *fps* are requested when given; a camera is free to
    ignore them, so read the frames to learn what it actually delivers.  The
    driver's buffer is kept to one frame so that what is read is the newest
    frame rather than a queue of stale ones.

    Raises ``RuntimeError`` if no backend can open the camera.
    """
    capture = None
    for backend in dict.fromkeys((preferred_backend(), cv2.CAP_ANY)):
        capture = cv2.VideoCapture(src, backend)
        if capture.isOpened():
            break
        capture.release()
        capture = None
    if capture is None:
        raise RuntimeError(f"Could not open camera id {src}")
    if width:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        capture.set(cv2.CAP_PROP_FPS, fps)
    # Avoid a long queue of old camera frames in a closed-loop experiment.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if sys.platform == "win32" and settings_dialog:
        capture.set(cv2.CAP_PROP_SETTINGS, 1)
    return capture
