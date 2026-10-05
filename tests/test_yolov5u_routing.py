# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""YOLOv5 and ultralytics' YOLOv5u must not be mistaken for one another.

They are one "u" apart in the file name and two backends apart in what can
read the file: upstream YOLOv5 is anchor-based and loads only with the bundled
copy, while ``yolov5su.pt`` is a YOLOv5 backbone under YOLOv8's anchor-free
head, in ultralytics' format.

A checkpoint that exists on disk is routed by the module names in its pickle,
so these tests use names that are *not* on disk -- the one case where the file
name alone decides, and the one where the two can be confused.
"""

import pytest

from yoru.libs import plugins
from yoru.libs.vram_estimate import profile_for

UPSTREAM = ["yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
ULTRALYTICS_U = ["yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5x6u.pt"]


@pytest.mark.parametrize("weight", UPSTREAM)
def test_an_upstream_name_reaches_the_bundled_backend(weight):
    assert plugins._auto_detect_backend(weight) == "yolov5"
    assert plugins.detect_trainer_backend({"weight": weight}) == "yolov5"


@pytest.mark.parametrize("weight", ULTRALYTICS_U)
def test_a_yolov5u_name_reaches_ultralytics(weight):
    """The bundled YOLOv5 cannot read these, and must not be handed them."""
    assert plugins._auto_detect_backend(weight) == "ultralytics"
    assert plugins.detect_trainer_backend({"weight": weight}) == "ultralytics"


@pytest.mark.parametrize("weight", ULTRALYTICS_U)
def test_a_yolov5u_name_does_not_borrow_a_yolov5_row(weight):
    """A different head costs different memory; no estimate beats a wrong one."""
    assert profile_for(weight) is None


@pytest.mark.parametrize("weight", UPSTREAM)
def test_upstream_names_keep_their_row(weight):
    assert profile_for(weight) is not None
