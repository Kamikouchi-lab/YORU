"""Frame-to-frame ID tracking in movie analysis.

``match_to_previous`` assigns each detection of a frame to one of the previous
frame's by minimum summed centre distance.  These tests pin the two ways the
first version got that wrong: surplus detections were matched to a dummy point
at (-1000, -1000), so a detection's distance from the image corner decided
which one lost its ID; and an ID was spent on every track that ended, leaving
gaps in the numbering.
"""

from __future__ import annotations

import itertools
import math
import random

import cv2
import numpy as np
import pandas as pd
import pytest

from yoru.libs import analysis
from yoru.libs.analysis import match_to_previous


def _cost(pre, cur, matches, max_dist=None):
    """Objective the matcher minimises, evaluated for *matches*.

    Without a cap: summed distance of the matched pairs.  With one: every pair
    of a full matching costs ``min(d, max_dist)``, so a split pair costs the
    cap -- the leftover pairs of a split are added back at that price.
    """
    total = 0.0
    kept = 0
    for j, i in enumerate(matches):
        if i >= 0:
            total += math.dist(pre[i], cur[j])
            kept += 1
    if max_dist is not None:
        total += max_dist * (min(len(pre), len(cur)) - kept)
    return total


def _best_cost(pre, cur, max_dist=None):
    """Brute-force optimum over every full matching of the smaller side."""
    if not pre or not cur:
        return 0.0
    clip = (lambda d: d) if max_dist is None else (lambda d: min(d, max_dist))
    if len(pre) <= len(cur):
        return min(
            sum(clip(math.dist(pre[i], cur[j])) for i, j in enumerate(perm))
            for perm in itertools.permutations(range(len(cur)), len(pre))
        )
    return min(
        sum(clip(math.dist(pre[i], cur[j])) for j, i in enumerate(perm))
        for perm in itertools.permutations(range(len(pre)), len(cur))
    )


class TestMatchToPrevious:
    def test_a_newcomer_far_away_does_not_take_the_existing_id(self):
        # The animal moved ~70 px toward the corner; a new one appeared far
        # off.  The dummy-point padding handed the old ID to the newcomer.
        assert match_to_previous([(100, 100)], [(50, 50), (1900, 1000)]) == [0, -1]

    def test_the_animal_that_stays_keeps_its_id_when_another_leaves(self):
        pre = [(1900, 1000), (100, 100)]
        assert match_to_previous(pre, [(1850, 1000)]) == [0]
        assert match_to_previous(pre, [(50, 50)]) == [1]

    def test_no_previous_frame_starts_every_track(self):
        assert match_to_previous([], [(1, 2), (3, 4)]) == [-1, -1]

    def test_an_empty_frame_matches_nothing(self):
        assert match_to_previous([(1, 2)], []) == []

    def test_a_move_past_the_cap_starts_a_new_track(self):
        assert match_to_previous([(0, 0)], [(80, 0)], max_dist=50) == [-1]
        assert match_to_previous([(0, 0)], [(40, 0)], max_dist=50) == [0]

    def test_the_cap_is_applied_before_the_assignment_not_only_after(self):
        # Uncapped, the best pairing is (0,0)-(60,0) and (100,0)-(200,0), both
        # past a 50 px cap; filtering that afterwards would keep nothing.  The
        # pair within the cap, (100,0)-(60,0), should survive.
        pre = [(0, 0), (100, 0)]
        cur = [(60, 0), (200, 0)]
        assert match_to_previous(pre, cur) == [0, 1]
        assert match_to_previous(pre, cur, max_dist=50) == [1, -1]

    @pytest.mark.parametrize("max_dist", [None, 150.0])
    def test_the_assignment_is_optimal(self, max_dist):
        rng = random.Random(0)
        for _ in range(300):
            pre = [(rng.uniform(0, 1920), rng.uniform(0, 1080))
                   for _ in range(rng.randint(0, 5))]
            cur = [(rng.uniform(0, 1920), rng.uniform(0, 1080))
                   for _ in range(rng.randint(0, 5))]
            if max_dist is not None:
                # Cluster the points so the cap actually decides something.
                pre = [(x / 8, y / 8) for x, y in pre]
                cur = [(x / 8, y / 8) for x, y in cur]
            matches = match_to_previous(pre, cur, max_dist)

            assert len(matches) == len(cur)
            used = [i for i in matches if i >= 0]
            assert len(used) == len(set(used))
            if max_dist is None:
                assert len(used) == min(len(pre), len(cur))
            else:
                assert all(math.dist(pre[i], cur[j]) <= max_dist
                           for j, i in enumerate(matches) if i >= 0)
            assert _cost(pre, cur, matches, max_dist) == pytest.approx(
                _best_cost(pre, cur, max_dist)
            )


class TestTrackingMaxDistSetting:
    @pytest.mark.parametrize("value,expected", [
        (0, None), (0.0, None), (-5, None), ("abc", None), (None, None),
        (40, 40.0), ("12.5", 12.5),
    ])
    def test_values(self, value, expected):
        assert analysis._tracking_max_dist({"tracking_max_dist": value}) == expected

    def test_missing_means_no_cap(self):
        assert analysis._tracking_max_dist({}) is None


def _det(cx, cy, cls=0):
    return {
        "x1": cx - 10.0, "y1": cy - 10.0, "x2": cx + 10.0, "y2": cy + 10.0,
        "conf": 0.9, "class_id": cls, "class_name": f"c{cls}",
    }


class _ScriptedDetector:
    names = {0: "c0"}

    def __init__(self, frames):
        self._frames = iter(frames)

    def detect(self, frame):
        return next(self._frames)


def test_analyze_numbers_new_tracks_without_gaps(tmp_path, monkeypatch):
    frames = [
        [_det(100, 100), _det(300, 100)],   # ids 0, 1
        [_det(102, 100)],                   # second animal leaves
        [_det(104, 100), _det(500, 400)],   # newcomer: next id is 2
    ]
    movie = tmp_path / "movie.avi"
    writer = cv2.VideoWriter(
        str(movie), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480)
    )
    assert writer.isOpened()
    for _ in frames:
        writer.write(np.zeros((480, 640, 3), np.uint8))
    writer.release()

    monkeypatch.setattr(
        analysis, "get_detector", lambda *a, **kw: _ScriptedDetector(frames)
    )
    m_dict = {
        "model_path": "fake.pt", "input_path": [str(movie)],
        "output_path": str(tmp_path), "create_video": False,
        "tracking_state": True, "tracking_exclude_classes": [],
        "tracking_max_dist": 0, "threshold": 0.3,
        "v_flip": False, "h_flip": False,
    }
    analysis.yolo_analysis(m_dict).analyze()

    table = pd.read_csv(tmp_path / "movie.csv")
    assert table["frame"].tolist() == [0, 0, 1, 2, 2]
    assert table["tracking_id"].tolist() == [0, 1, 0, 0, 2]
