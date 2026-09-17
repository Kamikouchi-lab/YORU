"""Automatic frame extraction: what comes out of a video, and what never does.

Everything here runs against a fake capture rather than a real video file.
That is not only for speed: the properties that matter -- the sample stays
inside the range the user asked for, no frame is saved twice, a rare posture
survives clustering, Stop actually stops -- are properties of the selection,
and a real encoder would add codec-dependent seeking noise on top of them
without testing anything extra.
"""

import numpy as np
import pytest

import cv2

from yoru.libs import frame_extraction as fe


class FakeCapture:
    """A video of ``nframes`` frames, each one a flat colour.

    Implements only the handful of ``cv2.VideoCapture`` calls the module uses.
    ``reads`` records every frame handed out, so a test can show that scanning
    walks the video forwards instead of seeking per frame.
    """

    def __init__(self, nframes=300, shape=(48, 64, 3), maker=None, opened=True):
        self.nframes = nframes
        self.shape = shape
        self.maker = maker or (lambda i: np.full(shape, i % 256, dtype=np.uint8))
        self.pos = 0
        self.opened = opened
        self.reads = []
        self.seeks = []
        self.released = False

    # -- capture protocol ------------------------------------------------
    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self.nframes)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.pos)
        return 0.0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value)
            self.seeks.append(int(value))
            return True
        return False

    def grab(self):
        if self.pos >= self.nframes:
            return False
        self.pos += 1
        return True

    def read(self):
        if self.pos >= self.nframes:
            return False, None
        frame = self.maker(self.pos)
        self.reads.append(self.pos)
        self.pos += 1
        return True, frame

    def release(self):
        self.released = True


def _grouped_video(nframes, groups, shape=(48, 64, 3)):
    """A capture that changes appearance ``groups`` times, in equal blocks.

    Blocks rather than an interleave on purpose: a stepped scan of interleaved
    frames aliases onto a subset of the appearances, which would be a property
    of the test video and not of anything under test.
    """

    def maker(i):
        block = min(groups - 1, i * groups // max(1, nframes))
        return np.full(shape, int(255 * block / max(1, groups - 1)), dtype=np.uint8)

    return FakeCapture(nframes=nframes, shape=shape, maker=maker)


# ---------------------------------------------------------------------------
# The window of video a sample is drawn from
# ---------------------------------------------------------------------------


def test_window_covers_the_whole_video_by_default():
    assert fe.frame_window(1000) == (0, 1000)


def test_window_is_a_fraction_of_the_video():
    assert fe.frame_window(1000, 0.25, 0.75) == (250, 750)


def test_window_never_runs_past_the_last_frame():
    first, last = fe.frame_window(37, 0.0, 1.0)
    assert (first, last) == (0, 37)


@pytest.mark.parametrize(
    "start,stop",
    [(0.5, 0.5), (0.8, 0.2), (-0.1, 0.5), (0.0, 1.5)],
)
def test_a_backwards_or_out_of_range_window_is_refused(start, stop):
    """Silently repairing it would extract from a range nobody asked for."""
    with pytest.raises(ValueError):
        fe.frame_window(100, start, stop)


# ---------------------------------------------------------------------------
# uniform
# ---------------------------------------------------------------------------


def test_uniform_returns_the_requested_number_of_distinct_frames():
    picked = fe.uniform_indices(1000, count=25, rng=0)
    assert len(picked) == 25
    assert len(set(picked)) == 25


def test_uniform_stays_inside_the_range():
    picked = fe.uniform_indices(1000, count=50, start=0.4, stop=0.6, rng=1)
    assert picked and all(400 <= i < 600 for i in picked)


def test_uniform_comes_back_sorted():
    """Saving seeks forwards through this list; out of order costs real time."""
    picked = fe.uniform_indices(500, count=30, rng=2)
    assert picked == sorted(picked)


def test_uniform_asked_for_more_frames_than_exist_returns_every_frame():
    assert fe.uniform_indices(12, count=50) == list(range(12))


def test_uniform_is_reproducible_from_a_seed():
    assert fe.uniform_indices(900, 20, rng=7) == fe.uniform_indices(900, 20, rng=7)


def test_uniform_without_a_seed_does_not_repeat_itself():
    """Extracting twice must widen the training set, not re-pick the same frames."""
    first = fe.uniform_indices(5000, 40)
    second = fe.uniform_indices(5000, 40)
    assert first != second


def test_uniform_on_an_empty_range_is_refused():
    with pytest.raises(ValueError):
        fe.uniform_indices(0, count=5)


# ---------------------------------------------------------------------------
# kmeans
# ---------------------------------------------------------------------------


def test_kmeans_takes_one_frame_from_each_visual_group():
    """The point of clustering: three looks in, three looks out.

    The 300 frames here are 3 repeating appearances. Uniform sampling of 3
    frames could easily land twice on the same one; clustering must not.
    """
    features = np.tile(np.eye(3, dtype=np.float32), (100, 1))
    indices = list(range(300))
    picked = fe.kmeans_indices_from_features(features, indices, count=3, rng=0)
    groups = {i % 3 for i in picked}
    assert len(picked) == 3
    assert groups == {0, 1, 2}


def test_kmeans_keeps_a_rare_appearance():
    """One odd frame in a thousand is the frame worth labelling."""
    features = np.zeros((1000, 4), dtype=np.float32)
    features[:, 0] = 1.0
    features[500] = np.array([0.0, 9.0, 9.0, 9.0], dtype=np.float32)
    picked = fe.kmeans_indices_from_features(features, list(range(1000)), count=4, rng=3)
    assert 500 in picked


def test_kmeans_returns_the_number_of_frames_that_was_asked_for():
    """Empty clusters are normal; handing back fewer frames than asked is not."""
    features = np.repeat(np.eye(3, dtype=np.float32), 20, axis=0)
    picked = fe.kmeans_indices_from_features(features, list(range(60)), count=10, rng=5)
    assert len(picked) == 10
    assert len(set(picked)) == 10


def test_kmeans_with_fewer_candidates_than_clusters_returns_them_all():
    features = np.eye(4, dtype=np.float32)
    assert fe.kmeans_indices_from_features(features, [3, 9, 27, 81], count=10) == [3, 9, 27, 81]


def test_kmeans_on_nothing_is_refused():
    with pytest.raises(ValueError):
        fe.kmeans_indices_from_features(np.empty((0, 0)), [], count=5)


# ---------------------------------------------------------------------------
# Reading the video
# ---------------------------------------------------------------------------


def test_thumbnail_is_small_grey_and_normalised():
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    row = fe.thumbnail(frame, resize_width=30)
    assert row.shape == (30 * 22,)  # 30 px wide, height follows the 4:3 frame
    assert row.dtype == np.float32
    assert 0.0 <= row.min() and row.max() <= 1.0


def test_thumbnail_keeps_colour_when_asked():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[..., 2] = 255
    assert fe.thumbnail(frame, 10, color=True).size == 10 * 10 * 3


def test_scanning_walks_the_video_forwards_once():
    """One seek, then sequential grabs -- seeking per frame is the slow way."""
    cap = FakeCapture(nframes=200)
    features, read = fe.read_features(cap, list(range(0, 200, 10)))
    assert read == list(range(0, 200, 10))
    assert features.shape[0] == 20
    assert len(cap.seeks) == 1


def test_scanning_stops_when_the_user_stops_it():
    cap = FakeCapture(nframes=500)
    seen = []

    def progress(done, total, phase):
        seen.append((done, total, phase))

    _, read = fe.read_features(
        cap, list(range(500)), progress=progress, should_stop=lambda: len(seen) >= 5
    )
    assert len(read) < 500
    assert seen and seen[0][1] == 500 and seen[0][2] == "scanning"


def test_selection_refuses_an_unknown_algorithm():
    with pytest.raises(ValueError):
        fe.select_indices(FakeCapture(), algo="magic")


def test_selection_refuses_a_video_with_no_frames():
    with pytest.raises(ValueError):
        fe.select_indices(FakeCapture(nframes=0))


def test_clustering_a_long_video_coarsens_the_step_instead_of_reading_it_all():
    """A three-hour video must not be pulled into memory frame by frame."""
    cap = _grouped_video(fe.MAX_CLUSTER_CANDIDATES * 10, groups=4)
    picked = fe.select_indices(cap, count=4, algo="kmeans", rng=0)
    assert len(cap.reads) <= fe.MAX_CLUSTER_CANDIDATES
    assert len(picked) == 4


def test_clustering_stays_inside_the_requested_range():
    cap = _grouped_video(1000, groups=5)
    picked = fe.select_indices(cap, count=5, algo="kmeans", start=0.5, stop=0.8, rng=0)
    assert picked and all(500 <= i < 800 for i in picked)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def test_saved_frames_are_named_after_the_frame_they_came_from(tmp_path):
    """Manual grabs use the same name, so both kinds land as one series."""
    cap = FakeCapture(nframes=100)
    paths = fe.save_frames(cap, [4, 40, 7], str(tmp_path), "arena1")
    assert [p.split("_")[-1] for p in paths] == ["4.png", "7.png", "40.png"]
    assert all(tmp_path.joinpath(p).exists() for p in (
        "arena1_4.png", "arena1_7.png", "arena1_40.png"))


def test_saving_survives_a_non_ascii_directory(tmp_path):
    """cv2.imwrite writes nothing at all under a Japanese path, and says so
    only through a return value nobody reads."""
    out = tmp_path / "実験データ"
    paths = fe.save_frames(FakeCapture(nframes=10), [1, 2], str(out), "実験")
    assert len(paths) == 2
    assert all(out.joinpath(f"実験_{i}.png").exists() for i in (1, 2))


def test_saving_stops_when_the_user_stops_it(tmp_path):
    cap = FakeCapture(nframes=100)
    stop = {"now": False}

    def should_stop():
        return stop["now"]

    def progress(done, total, phase):
        if done >= 2:
            stop["now"] = True

    paths = fe.save_frames(
        cap, list(range(20)), str(tmp_path), "x", progress=progress, should_stop=should_stop
    )
    assert len(paths) == 2


# ---------------------------------------------------------------------------
# End to end, as the GUI calls it
# ---------------------------------------------------------------------------


def test_extraction_writes_the_frames_and_reports_them(tmp_path):
    cap = _grouped_video(400, groups=4)
    result = fe.extract_frames(
        "fake.mp4", str(tmp_path), "fly", count=8, algo="uniform", seed=0,
        open_capture=lambda path: cap,
    )
    assert result.ok
    assert len(result.paths) == 8
    assert len(list(tmp_path.glob("fly_*.png"))) == 8
    assert cap.released


def test_extraction_reports_a_video_it_cannot_open(tmp_path):
    result = fe.extract_frames(
        "missing.mp4", str(tmp_path), "fly",
        open_capture=lambda path: FakeCapture(opened=False),
    )
    assert not result.ok
    assert "missing.mp4" in result.message
    assert not list(tmp_path.iterdir())


def test_extraction_reports_bad_settings_without_touching_the_video(tmp_path):
    opened = []

    def opener(path):
        opened.append(path)
        return FakeCapture()

    result = fe.extract_frames(
        "v.mp4", str(tmp_path), "fly", count=0, open_capture=opener
    )
    assert not result.ok and not opened


def test_a_stopped_extraction_says_so_and_keeps_what_it_saved(tmp_path):
    cap = FakeCapture(nframes=100)
    stop = {"now": False}

    def progress(done, total, phase):
        if phase == "saving" and done >= 3:
            stop["now"] = True

    result = fe.extract_frames(
        "v.mp4", str(tmp_path), "fly", count=10, algo="uniform", seed=0,
        progress=progress, should_stop=lambda: stop["now"], open_capture=lambda p: cap,
    )
    assert result.cancelled and not result.ok
    assert len(result.paths) == 3
    assert len(list(tmp_path.glob("*.png"))) == 3


@pytest.mark.parametrize(
    "count,start,stop,algo",
    [(0, 0.0, 1.0, "uniform"), ("x", 0.0, 1.0, "uniform"),
     (10, 0.9, 0.1, "uniform"), (10, 0.0, 1.0, "kmeanz")],
)
def test_bad_settings_are_named_before_anything_runs(count, start, stop, algo):
    assert fe.validate(count, start, stop, algo) is not None


def test_good_settings_pass_validation():
    assert fe.validate(20, 0.0, 1.0, "kmeans") is None
