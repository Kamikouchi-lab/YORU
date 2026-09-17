# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Base class for detection engine plugins, and the shape of one detection.

This module is part of YORU core and is NOT subject to any plugin's license.
"""

from yoru.libs.obb import aabb_to_obb

#: Column order of one row of ``m_dict["yolo_results"]``, and of the
#: ``*_detect.csv`` header.  The first eight are what YORU has always written,
#: unchanged and in the same places, so trigger plugins that index a row by
#: position keep working; the last five are the box as an oriented rectangle.
#: Defining the order once here is what keeps the CSV header and the rows the
#: writer is handed from ever disagreeing.
DETECTION_COLUMNS = (
    "x1", "y1", "x2", "y2",
    "confidence", "class", "class_name", "total_time",
    "cx", "cy", "w", "h", "angle",
)


def obb_of(detection):
    """``(cx, cy, w, h, angle)`` for a detection dict.

    A backend that predicts rotated boxes supplies these directly; every other
    backend supplies only ``x1..y2``, and the upright box is derived from them
    with a zero angle.  Deriving it here rather than in each plugin means the
    four detector backends stay identical in this respect, and a new one gets
    it for free.
    """
    if detection.get("angle") is not None:
        return (
            float(detection["cx"]), float(detection["cy"]),
            float(detection["w"]), float(detection["h"]),
            float(detection["angle"]),
        )
    return aabb_to_obb(
        detection["x1"], detection["y1"], detection["x2"], detection["y2"]
    )


def detection_row(detection, total_time):
    """One detection as a row in :data:`DETECTION_COLUMNS` order."""
    cx, cy, w, h, angle = obb_of(detection)
    return [
        detection["x1"], detection["y1"], detection["x2"], detection["y2"],
        detection["conf"], detection["class_id"], detection["class_name"],
        total_time,
        cx, cy, w, h, angle,
    ]


class DetectorBase:
    """Abstract interface that all detection plugins must implement."""

    def load(self, model_path: str, **kwargs) -> None:
        """Load a model from the given path."""
        raise NotImplementedError

    def detect(self, image) -> list:
        """Run detection on a BGR numpy image (H, W, 3).

        Returns a list of dicts, each with keys:
            x1, y1, x2, y2 (float): axis-aligned bounding box, always present
            conf (float): confidence score
            class_id (int): class ID
            class_name (str): class name

        A backend that predicts *oriented* boxes adds five more keys —
        ``cx``, ``cy``, ``w``, ``h`` and ``angle`` (radians) — describing the
        rotated box itself, while ``x1..y2`` stays the upright box around it so
        that consumers written before OBB support keep working.  Backends that
        do not predict rotation leave these out; :func:`obb_of` fills them in.
        """
        raise NotImplementedError

    @property
    def names(self) -> dict:
        """Return {class_id: class_name} mapping for the loaded model."""
        raise NotImplementedError
