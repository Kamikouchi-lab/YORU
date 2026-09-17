# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Choose which frames to label, instead of scrubbing a video for them.

A training set starts as a few hundred stills pulled out of hours of video, and
*which* stills they are decides what the model can learn.  Stepping through the
video by hand gives frames that were easy to reach rather than frames that are
worth labelling: they cluster around whatever the user happened to be watching,
and they over-represent the long stretches where nothing moves.

This module is YORU's answer to that, and it follows the two algorithms
DeepLabCut's ``extract_frames`` offers, because they answer two different
questions (the code here is an independent implementation, not a port):

``uniform``
    Sample frames uniformly at random from a slice of the video.  The right
    default when the behaviour of interest is frequent -- the sample matches
    the distribution of the video itself, so the training set looks like what
    the model will be shown at run time.

``kmeans``
    Shrink every candidate frame to a thumbnail, cluster the thumbnails, and
    take one random frame from each cluster.  A rare posture occupies a small
    cluster but still a whole cluster, so it survives the sample instead of
    being drowned out by the thousands of near-identical frames of an animal
    sitting still.  This is what makes 200 frames cover the behaviour rather
    than the background.

Two design points worth keeping:

* **Selection is separate from saving, and neither touches the GUI.**  The
  functions here take an already-open capture and plain numbers, so the
  algorithms can be tested against a fake capture with no video file, no codec
  and no window -- see ``tests/test_frame_extraction.py``.
* **The capture is passed in, never shared.**  Extraction runs on a worker
  thread while the preview still seeks on the GUI thread; two threads calling
  ``set``/``read`` on one ``cv2.VideoCapture`` interleave into garbled frames
  or a hard crash, so the caller opens its own.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

# The two algorithm names the GUI offers, in the order it offers them.
ALGORITHMS = ("uniform", "kmeans")

DEFAULT_COUNT = 20
DEFAULT_STEP = 1
DEFAULT_RESIZE_WIDTH = 30

# Clustering reads every candidate frame into one matrix.  At the default
# thumbnail size a candidate costs ~2 kB, so this cap keeps the matrix in the
# tens of megabytes and, more to the point, keeps the decode from taking longer
# than anyone is willing to wait.  Going over it coarsens the step instead of
# failing: a three-hour video is sampled every n-th frame rather than refused.
MAX_CLUSTER_CANDIDATES = 5000

ProgressFn = Callable[[int, int, str], None]
StopFn = Callable[[], bool]


@dataclass(frozen=True)
class ExtractionResult:
    """What one extraction run produced, in the words the status line needs."""

    ok: bool
    message: str
    indices: tuple = ()
    paths: tuple = ()
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Which frames
# ---------------------------------------------------------------------------


def _as_rng(rng):
    """Accept a Generator, a seed, or nothing, and always return a Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def frame_window(nframes: int, start: float = 0.0, stop: float = 1.0):
    """Return the half-open ``[first, last)`` frame range named by two fractions.

    ``start``/``stop`` are fractions of the video, the way DeepLabCut spells
    them, so "the second half" is ``0.5``/``1.0`` and keeps its meaning when
    the same settings are reused on a video of a different length.
    """
    n = max(0, int(nframes))
    start = float(start)
    stop = float(stop)
    if not (0.0 <= start < stop <= 1.0):
        raise ValueError(
            "the range must satisfy 0 <= start < stop <= 1 "
            f"(got start={start}, stop={stop})"
        )
    first = int(math.floor(n * start))
    last = min(n, int(math.ceil(n * stop)))
    return first, max(first, last)


def candidate_indices(
    nframes: int, start: float = 0.0, stop: float = 1.0, step: int = DEFAULT_STEP
):
    """Every frame clustering may look at: the window, thinned by ``step``."""
    first, last = frame_window(nframes, start, stop)
    step = max(1, int(step))
    return list(range(first, last, step))


def uniform_indices(
    nframes: int,
    count: int = DEFAULT_COUNT,
    start: float = 0.0,
    stop: float = 1.0,
    rng=None,
) -> list:
    """Pick ``count`` frames uniformly at random, without repeats.

    Returned in ascending order -- unlike DeepLabCut, which returns them in
    draw order.  Saving walks the video forwards, and a sorted list turns that
    walk into forward seeks instead of a random-access scramble.
    """
    first, last = frame_window(nframes, start, stop)
    available = last - first
    if available <= 0:
        raise ValueError("the selected range holds no frames")
    count = max(1, int(count))
    if count >= available:
        return list(range(first, last))
    picked = _as_rng(rng).choice(np.arange(first, last), size=count, replace=False)
    return sorted(int(i) for i in picked)


def kmeans_indices_from_features(
    features,
    indices: Sequence[int],
    count: int = DEFAULT_COUNT,
    rng=None,
    max_iter: int = 50,
) -> list:
    """Cluster ``features`` into ``count`` groups and take one frame per group.

    ``features[i]`` is the flattened thumbnail of ``indices[i]``.  Empty
    clusters are normal -- k-means++ can place a centroid where no frame lands
    -- and DeepLabCut simply returns fewer frames when that happens.  Here the
    shortfall is topped up with unpicked candidates instead: the user asked for
    a number of frames to label, and quietly handing back fewer is a surprise
    that surfaces much later, in the labelling tool.
    """
    indices = [int(i) for i in indices]
    n = len(indices)
    if n == 0:
        raise ValueError("the selected range holds no readable frames")
    count = max(1, int(count))
    if count >= n:
        return sorted(indices)

    from scipy.cluster.vq import kmeans2

    rng = _as_rng(rng)
    data = np.asarray(features, dtype=np.float32)
    if data.ndim != 2 or data.shape[0] != n:
        raise ValueError("features must hold one row per candidate frame")

    # An empty cluster is not an error here, it is the expected outcome of
    # asking for more frames than the video has distinct appearances -- a
    # fixed camera on a still arena is the normal case, not the exotic one.
    # scipy says so twice: a UserWarning about the empty cluster, and a
    # RuntimeWarning from k-means++ dividing by a zero total distance once
    # every distinct point is already a centre.  Both are handled by the
    # top-up below, so in the user's console they would be pure noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        _, labels = kmeans2(
            data, count, iter=max_iter, minit="++", missing="warn", seed=rng
        )

    picked = []
    for cluster in range(count):
        members = np.flatnonzero(labels == cluster)
        if members.size:
            picked.append(indices[int(members[rng.integers(members.size)])])

    if len(picked) < count:
        taken = set(picked)
        spare = [i for i in indices if i not in taken]
        if spare:
            extra = rng.choice(
                np.asarray(spare),
                size=min(count - len(picked), len(spare)),
                replace=False,
            )
            picked.extend(int(i) for i in np.atleast_1d(extra))
    return sorted(set(picked))


# ---------------------------------------------------------------------------
# Reading the video
# ---------------------------------------------------------------------------


def frame_count(cap) -> int:
    """Frames in ``cap``, as an int and never negative."""
    try:
        return max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    except Exception:
        return 0


def thumbnail(frame, resize_width: int = DEFAULT_RESIZE_WIDTH, color: bool = False):
    """Flatten one frame into the feature row that clustering compares.

    A 30 px-wide thumbnail is deliberately too small to show an animal's legs:
    what should decide whether two frames are "the same" is the pose and the
    layout of the arena, not the pixel noise that dominates a full-resolution
    difference.
    """
    h, w = frame.shape[:2]
    out_w = max(2, int(resize_width))
    out_h = max(2, int(round(h * out_w / max(1, w))))
    small = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if not color and small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return (small.astype(np.float32) / 255.0).ravel()


def read_features(
    cap,
    indices: Sequence[int],
    resize_width: int = DEFAULT_RESIZE_WIDTH,
    color: bool = False,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
):
    """Read the candidate frames and return ``(features, indices_read)``.

    The walk is sequential -- one seek to the first frame, then ``grab`` past
    the frames the step skips -- because ``set(CAP_PROP_POS_FRAMES)`` on a
    compressed video rewinds to the previous keyframe and decodes forward from
    there.  Seeking per frame turns a one-minute scan into a ten-minute one.
    """
    wanted = [int(i) for i in indices]
    if not wanted:
        return np.empty((0, 0), dtype=np.float32), []

    rows = []
    read_indices = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
    pos = wanted[0]
    for done, idx in enumerate(wanted, start=1):
        if should_stop is not None and should_stop():
            break
        while pos < idx:
            if not cap.grab():
                pos = idx
                break
            pos += 1
        ok, frame = cap.read()
        pos += 1
        if ok and frame is not None:
            rows.append(thumbnail(frame, resize_width, color))
            read_indices.append(idx)
        if progress is not None:
            progress(done, len(wanted), "scanning")

    if not rows:
        return np.empty((0, 0), dtype=np.float32), []
    return np.vstack(rows), read_indices


def select_indices(
    cap,
    count: int = DEFAULT_COUNT,
    algo: str = "uniform",
    start: float = 0.0,
    stop: float = 1.0,
    step: int = DEFAULT_STEP,
    resize_width: int = DEFAULT_RESIZE_WIDTH,
    color: bool = False,
    rng=None,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
) -> list:
    """The frames to extract from ``cap``, by whichever algorithm was asked for."""
    algo = str(algo).lower()
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algo!r}; expected one of {ALGORITHMS}")
    nframes = frame_count(cap)
    if nframes <= 0:
        raise ValueError("the video reports no frames")

    if algo == "uniform":
        return uniform_indices(nframes, count, start, stop, rng=rng)

    first, last = frame_window(nframes, start, stop)
    step = max(1, int(step))
    span = last - first
    if span > MAX_CLUSTER_CANDIDATES * step:
        step = int(math.ceil(span / MAX_CLUSTER_CANDIDATES))
    candidates = list(range(first, last, step))
    features, read = read_features(
        cap, candidates, resize_width, color, progress=progress, should_stop=should_stop
    )
    if should_stop is not None and should_stop():
        return []
    return kmeans_indices_from_features(features, read, count, rng=rng)


# ---------------------------------------------------------------------------
# Writing the frames out
# ---------------------------------------------------------------------------


def imwrite(path: str, frame) -> bool:
    """``cv2.imwrite`` that also works when the path is not ASCII.

    OpenCV hands the file name to the C runtime in the system code page, so a
    save directory with Japanese characters in it makes ``cv2.imwrite`` return
    False and write nothing -- silently, since nobody checks its return value.
    Encoding in memory and letting numpy open the file avoids the code page.
    """
    ext = os.path.splitext(path)[1] or ".png"
    try:
        ok, buf = cv2.imencode(ext, frame)
        if not ok:
            return False
        buf.tofile(path)
        return True
    except Exception:
        return False


def save_frames(
    cap,
    indices: Sequence[int],
    out_dir: str,
    name: str,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
) -> list:
    """Write each selected frame as ``<name>_<frame number>.png``.

    The name matches what "Grab Current Frame" writes, so frames picked by hand
    and frames picked automatically land in one folder as one series, and the
    number in the file name still says where in the video the frame came from.
    """
    wanted = sorted(int(i) for i in indices)
    if not wanted:
        return []
    os.makedirs(out_dir, exist_ok=True)

    paths = []
    for done, idx in enumerate(wanted, start=1):
        if should_stop is not None and should_stop():
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            path = os.path.join(out_dir, f"{name}_{idx}.png")
            if imwrite(path, frame):
                paths.append(path)
        if progress is not None:
            progress(done, len(wanted), "saving")
    return paths


def validate(count, start, stop, algo) -> Optional[str]:
    """Return the sentence to show the user, or None when the settings are fine.

    Called before the worker thread starts, so a typo is answered at once
    instead of after a minute of scanning.
    """
    try:
        count = int(count)
    except (TypeError, ValueError):
        return "Number of frames must be a whole number."
    if count < 1:
        return "Number of frames must be at least 1."
    try:
        start = float(start)
        stop = float(stop)
    except (TypeError, ValueError):
        return "Range start and stop must be numbers."
    if not (0.0 <= start < stop <= 1.0):
        return "Range must satisfy 0 <= start < stop <= 1."
    if str(algo).lower() not in ALGORITHMS:
        return f"Unknown algorithm: {algo}."
    return None


def extract_frames(
    video_path: str,
    out_dir: str,
    name: str,
    count: int = DEFAULT_COUNT,
    algo: str = "uniform",
    start: float = 0.0,
    stop: float = 1.0,
    step: int = DEFAULT_STEP,
    resize_width: int = DEFAULT_RESIZE_WIDTH,
    color: bool = False,
    seed=None,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[StopFn] = None,
    open_capture: Optional[Callable] = None,
) -> ExtractionResult:
    """Select frames from ``video_path`` and write them into ``out_dir``.

    Opens, and closes, its own capture: see the module docstring on why it may
    not borrow the one the preview is using.  ``open_capture`` is there so the
    tests can hand this a fake video.
    """
    problem = validate(count, start, stop, algo)
    if problem:
        return ExtractionResult(False, problem)

    opener = open_capture or cv2.VideoCapture
    indices = []
    paths = []
    cap = opener(video_path)
    try:
        if not cap.isOpened():
            return ExtractionResult(False, f"Could not open the video: {video_path}")
        indices = select_indices(
            cap,
            count=count,
            algo=algo,
            start=start,
            stop=stop,
            step=step,
            resize_width=resize_width,
            color=color,
            rng=seed,
            progress=progress,
            should_stop=should_stop,
        )
        if not (should_stop is not None and should_stop()):
            paths = save_frames(
                cap, indices, out_dir, name, progress=progress, should_stop=should_stop
            )
    except Exception as exc:  # a worker thread has nowhere to raise
        return ExtractionResult(False, f"Extraction failed: {exc}")
    finally:
        try:
            cap.release()
        except Exception:
            pass

    if should_stop is not None and should_stop():
        return ExtractionResult(
            False,
            f"Stopped after saving {len(paths)} frame(s).",
            tuple(indices),
            tuple(paths),
            cancelled=True,
        )
    if not paths:
        return ExtractionResult(False, "No frames could be read from that range.")
    # Without the directory in it: the GUI shows this next to the box holding
    # that directory, and a long path there wraps onto three lines and pushes
    # the rest of the panel out of the window.
    return ExtractionResult(
        True,
        f"Saved {len(paths)} frame(s).",
        tuple(indices),
        tuple(paths),
    )
