# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Base class for detection engine plugins.

This module is part of YORU core and is NOT subject to any plugin's license.
"""


class DetectorBase:
    """Abstract interface that all detection plugins must implement."""

    def load(self, model_path: str, **kwargs) -> None:
        """Load a model from the given path."""
        raise NotImplementedError

    def detect(self, image) -> list:
        """Run detection on a BGR numpy image (H, W, 3).

        Returns a list of dicts, each with keys:
            x1, y1, x2, y2 (float): bounding box coordinates
            conf (float): confidence score
            class_id (int): class ID
            class_name (str): class name
        """
        raise NotImplementedError

    @property
    def names(self) -> dict:
        """Return {class_id: class_name} mapping for the loaded model."""
        raise NotImplementedError
