# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.
#
# Ported from YOAKE (src/htrtdetr/gui/click_segment.py), MIT License,
# Copyright (c) 2026 Hayato M Yamanouchi.  The "local" (OpenCV) backend and the
# mask -> OBB reduction are carried over essentially unchanged; the SAM sidecar
# and the model-candidate backends were not ported, and the geometry helpers
# now come from yoru.libs.obb so that YORU has one definition of a rotated box.

"""Click to Box — one click on an animal gives back a box that fits it.

The annotation tool's most expensive gesture is drawing a rectangle around a
small, tilted animal: a careful drag plus a correction, per animal, per frame.
This module replaces it with a single click.

**The quality of a click-to-box tool is decided by how a mask is reduced to a
rectangle, not by how good the mask is.**  ``cv2.minAreaRect`` on a raw fly
mask swallows the legs and wings, which both inflates the box and twists its
axis away from the body.  :func:`mask_to_obb` is therefore the one place that
reduction happens, and everything else here exists to hand it a mask.

Invariants, asserted by ``tests/test_click_segment.py``:

* No Qt and no dearpygui import, at module scope or inside a function — the
  engine is a pure function of (frame, click) and is tested without a display.
* ``cv2`` / ``numpy`` are imported lazily, inside the functions that use them,
  so this module imports on a machine with neither.
* Every public function is deterministic and never raises on bad input.  A
  failure comes back as a :class:`ClickResult` with ``ok == False`` carrying a
  named diagnosis and the exact sentence the status bar should show — this is
  called straight from a mouse handler, where an exception is not an option and
  a bare ``None`` would throw away the reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from yoru.libs.obb import normalize_angle, obb_to_aabb

__all__ = [
    "ClickRequest",
    "ClickResult",
    "NOTHING_FOUND",
    "MERGED_BACKGROUND",
    "TWO_ANIMALS",
    "TOUCHES_EDGE",
    "OUT_OF_FRAME",
    "NO_FRAME",
    "DIAGNOSES",
    "MIN_BODY_PX",
    "DEFAULT_RADIUS",
    "MAX_RADIUS",
    "CORE_FRACTION",
    "BODY_FRACTION",
    "MAX_AREA_FRACTION",
    "TWIN_CORE_RATIO",
    "mask_to_obb",
    "segment_local",
    "segment",
]


# --------------------------------------------------------------------------- #
# 1. Constants
# --------------------------------------------------------------------------- #

#: Below this, a box is noise, not an animal.
MIN_BODY_PX = 5

#: Half-side of the first search window, in image pixels.  The window doubles
#: (up to :data:`MAX_RADIUS`) while the blob still runs off its edge, so this is
#: a starting guess, not a limit.
DEFAULT_RADIUS = 48

#: Refuse to search wider than this.  Past it, Otsu is thresholding a scene
#: rather than an animal, and a wrong box is worse than no box.
MAX_RADIUS = 256

#: A pixel belongs to the "body core" when its distance to the nearest
#: background pixel is at least this fraction of the blob's maximum.  Legs and
#: wings are thin, so they fall out; the thorax and abdomen stay.  The core
#: decides the *axis*, where robustness matters more than exact size.
CORE_FRACTION = 0.40

#: Radius of the opening that decides the *extent*, as a fraction of the body's
#: half-thickness.  A morphological opening deletes everything thinner than
#: twice its radius — legs, wings, antennae — while **preserving the size of
#: what survives**, which is exactly the property the extent needs, and the
#: reason the extent is measured neither on the raw blob (legs would inflate
#: it) nor on the core (deliberately shrunken).  Scaling the radius by the
#: blob's own thickness is what keeps this free of any pixel-count tuning.
BODY_FRACTION = 0.15

#: A blob covering more than this fraction of the search window is not one
#: animal — the threshold ran into the background.
MAX_AREA_FRACTION = 0.40

#: Two body cores whose areas are this similar mean two touching animals, not
#: one animal with a dent in it.
TWIN_CORE_RATIO = 0.60


# --------------------------------------------------------------------------- #
# 2. Diagnoses — name the cause, and name what to do instead
# --------------------------------------------------------------------------- #

NOTHING_FOUND = "nothing_found"
MERGED_BACKGROUND = "merged_background"
TWO_ANIMALS = "two_animals"
TOUCHES_EDGE = "touches_edge"
OUT_OF_FRAME = "out_of_frame"
NO_FRAME = "no_frame"

#: The exact sentence to show for each failure.  Every one names the cause
#: *and* the next action, and every one offers the same escape hatch: draw the
#: box by hand, which always works.
DIAGNOSES = {
    NOTHING_FOUND: (
        "Nothing found at that point. Click on the animal itself, or draw the "
        "box by hand with W."
    ),
    MERGED_BACKGROUND: (
        "The animal merged with the background here. Zoom in and click again, "
        "or draw the box by hand with W."
    ),
    TWO_ANIMALS: (
        "Two animals look joined at that point. Click nearer this one's "
        "centre, or draw the box by hand with W."
    ),
    TOUCHES_EDGE: (
        "The shape kept running past the search area. Click nearer its centre, "
        "or draw the box by hand with W."
    ),
    OUT_OF_FRAME: "That point is outside the image.",
    NO_FRAME: "No image is loaded yet.",
}


# --------------------------------------------------------------------------- #
# 3. Request / result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClickRequest:
    """One click, in image pixels (zoom and pan already undone)."""

    x: float
    y: float
    #: Half-side of the first search window.  ``None`` uses
    #: :data:`DEFAULT_RADIUS`.  The caller passes back the last successful body
    #: length so a second click on a similar animal starts at the right scale.
    radius_hint: Optional[float] = None
    #: ``(x0, y0, x1, y1)`` pinning the search window instead of guessing it.
    roi: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class ClickResult:
    """What one click produced — or why it produced nothing.

    A failure is a normal result, not an exception: it carries the diagnosis
    and the sentence to show.  Callers branch on :attr:`ok`.
    """

    obb: Optional[Tuple[float, float, float, float, float]]
    bbox: Optional[Tuple[int, int, int, int]]
    backend: str
    detail: str
    #: ``""`` on success, otherwise a key of :data:`DIAGNOSES`.
    diagnosis: str = ""

    @property
    def ok(self) -> bool:
        return self.obb is not None


def _failed(diagnosis: str, backend: str = "local") -> ClickResult:
    return ClickResult(
        obb=None,
        bbox=None,
        backend=backend,
        detail=DIAGNOSES.get(diagnosis, "Could not fit a box there."),
        diagnosis=diagnosis,
    )


# --------------------------------------------------------------------------- #
# 4. mask -> OBB — the whole quality question, in one function
# --------------------------------------------------------------------------- #


def mask_to_obb(mask, at):
    """Reduce a binary mask to a body-axis OBB.

    Returns ``(obb, "")`` on success and ``(None, diagnosis)`` otherwise, where
    ``diagnosis`` is a key of :data:`DIAGNOSES`.  ``at`` is the click, in the
    mask's own coordinates.

    Four steps, in this order and for these reasons:

    1. **Keep only the blob under the click.**  Anything else in the window is
       another animal or background texture.
    2. **Strip the appendages.**  A distance transform, cut at
       :data:`CORE_FRACTION` of its own maximum, deletes everything thinner
       than 40% of the thickest part and keeps the body.  Cutting at a fraction
       of the blob's own maximum is what makes this scale-free: it needs no
       idea of how many pixels an animal is.
    3. **Axis from the core by PCA, extent from the opened body.**  PCA rather
       than ``minAreaRect`` because the axis stays put when the core is dented.
       The extent comes from a morphological opening, which deletes the legs
       outright and returns the body at its original size — so the measurement
       needs no correction term, which is what the raw blob and the core both
       lack.
    4. **Canonicalise the angle**, never the side assignment: ``w`` is the body
       axis whether or not it is the longer side.

    Head and tail are *not* distinguished.  Orientation lives mod pi here, so a
    head-tail flip is not representable and guessing one would be inventing
    information.
    """
    import cv2
    import numpy as np

    m = np.ascontiguousarray(mask)
    if m.ndim != 2:
        return None, NOTHING_FOUND
    m = (m > 0).astype(np.uint8)
    h, w = m.shape[:2]
    if h < 2 or w < 2 or int(m.sum()) == 0:
        return None, NOTHING_FOUND

    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None, NOTHING_FOUND

    # -- 1. the blob under the click --------------------------------------
    px, py = int(round(at[0])), int(round(at[1]))
    lab = 0
    if 0 <= px < w and 0 <= py < h:
        lab = int(labels[py, px])
    if lab == 0:
        # The click landed a pixel or two off the animal (thresholding is not
        # exact at the edge).  Accept the nearest blob within a quarter of the
        # window, and only that.
        reach = 0.25 * max(h, w)
        best = None
        for i in range(1, n):
            d = math.hypot(float(cents[i][0]) - px, float(cents[i][1]) - py)
            if d <= reach and (best is None or d < best[0]):
                best = (d, i)
        if best is None:
            return None, NOTHING_FOUND
        lab = best[1]

    area = int(stats[lab, cv2.CC_STAT_AREA])
    if area < MIN_BODY_PX * MIN_BODY_PX:
        return None, NOTHING_FOUND

    bx = int(stats[lab, cv2.CC_STAT_LEFT])
    by = int(stats[lab, cv2.CC_STAT_TOP])
    bw = int(stats[lab, cv2.CC_STAT_WIDTH])
    bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
    # Runs off the window: the caller can widen the window and try again, which
    # is a different situation from "this really is one huge blob".
    if bx == 0 or by == 0 or bx + bw >= w or by + bh >= h:
        return None, TOUCHES_EDGE
    if area > MAX_AREA_FRACTION * h * w:
        return None, MERGED_BACKGROUND

    comp = (labels == lab).astype(np.uint8)

    # -- 2. strip the appendages ------------------------------------------
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
    dmax = float(dist.max())
    core = (dist >= CORE_FRACTION * dmax).astype(np.uint8) if dmax > 0 else comp
    if int(core.sum()) < 4:
        core = comp

    # Two cores of comparable size are two animals that touch.  One core with a
    # small satellite is one animal, so the ratio — not the count — decides.
    nc, clabels, cstats, ccents = cv2.connectedComponentsWithStats(core, 8)
    if nc > 2:
        areas = sorted(
            (int(cstats[i, cv2.CC_STAT_AREA]) for i in range(1, nc)), reverse=True
        )
        if areas[0] > 0 and areas[1] >= TWIN_CORE_RATIO * areas[0]:
            return None, TWO_ANIMALS
        # Otherwise keep the core nearest the click and drop the specks.
        pick, best_d = 1, None
        for i in range(1, nc):
            d = math.hypot(float(ccents[i][0]) - px, float(ccents[i][1]) - py)
            if best_d is None or d < best_d:
                pick, best_d = i, d
        core = (clabels == pick).astype(np.uint8)

    # -- 3. axis from the core, extent from the opened body ---------------
    ys, xs = np.nonzero(core)
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    mean = pts.mean(axis=0)
    centred = pts - mean
    cov = (centred.T @ centred) / max(len(centred) - 1, 1)
    try:
        _evals, evecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None, NOTHING_FOUND
    u = np.asarray(evecs[:, -1], dtype=np.float64)  # largest eigenvalue last
    if not np.isfinite(u).all() or float(np.hypot(u[0], u[1])) < 1e-12:
        u = np.array([1.0, 0.0])
    u = u / float(np.hypot(u[0], u[1]))
    v = np.array([-u[1], u[0]])

    radius = max(1, int(round(BODY_FRACTION * dmax))) if dmax > 0 else 0
    body = comp
    if radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        opened = cv2.morphologyEx(comp, cv2.MORPH_OPEN, kernel)
        if int(opened.sum()) >= 4:
            body = opened
    # An opening can split a blob at a waist; keep the piece under the click.
    nb, blabels, _bstats, bcents = cv2.connectedComponentsWithStats(body, 8)
    if nb > 2:
        pick, best_d = 1, None
        for i in range(1, nb):
            d = math.hypot(float(bcents[i][0]) - mean[0], float(bcents[i][1]) - mean[1])
            if best_d is None or d < best_d:
                pick, best_d = i, d
        body = (blabels == pick).astype(np.uint8)

    cys, cxs = np.nonzero(body)
    cpts = np.stack([cxs.astype(np.float64), cys.astype(np.float64)], axis=1) - mean
    pu = cpts @ u
    pv = cpts @ v
    u0, u1 = float(pu.min()), float(pu.max())
    v0, v1 = float(pv.min()), float(pv.max())
    # +1 because a single pixel spans one pixel, not zero.
    length = (u1 - u0) + 1.0
    breadth = (v1 - v0) + 1.0
    mid_u, mid_v = (u0 + u1) / 2.0, (v0 + v1) / 2.0
    cx = float(mean[0] + mid_u * u[0] + mid_v * v[0])
    cy = float(mean[1] + mid_u * u[1] + mid_v * v[1])

    if length < MIN_BODY_PX or breadth < 1.0:
        return None, NOTHING_FOUND

    # -- 4. canonical angle, body axis stays in w -------------------------
    theta = normalize_angle(math.atan2(float(u[1]), float(u[0])))
    return (cx, cy, float(length), float(breadth), theta), ""


# --------------------------------------------------------------------------- #
# 5. Backend: local (OpenCV only)
# --------------------------------------------------------------------------- #


def _window(x, y, radius, width, height):
    """Search window around a point, clamped to the image."""
    r = max(8.0, float(radius))
    x0 = int(max(0, math.floor(x - r)))
    y0 = int(max(0, math.floor(y - r)))
    x1 = int(min(width, math.ceil(x + r)))
    y1 = int(min(height, math.ceil(y + r)))
    return x0, y0, x1, y1


def _otsu_mask(roi_bgr, at_local):
    """Binary mask of the ROI, with the polarity chosen from the click itself.

    Deciding dark-on-light versus light-on-dark from the pixels under the
    cursor is what makes one gesture work on both a dark fly on a lit arena and
    a pale larva on a dark plate, with nothing for the user to configure.
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    px, py = int(round(at_local[0])), int(round(at_local[1]))
    h, w = gray.shape[:2]
    px = max(0, min(px, w - 1))
    py = max(0, min(py, h - 1))
    y0, y1 = max(0, py - 3), min(h, py + 4)
    x0, x1 = max(0, px - 3), min(w, px + 4)
    here = float(np.median(gray[y0:y1, x0:x1]))
    overall = float(np.median(gray))
    dark_object = here <= overall

    flag = cv2.THRESH_BINARY_INV if dark_object else cv2.THRESH_BINARY
    _thr, bw = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    # A 3x3 open removes single-pixel noise.  It also thins legs slightly,
    # which is free help for the distance transform later.
    return cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )


def _grabcut_mask(roi_bgr, at_local):
    """Slower second opinion, used only after Otsu has already failed.

    Returns ``None`` when GrabCut cannot run (window too small, degenerate
    rect), which is a normal outcome and not an error.
    """
    import cv2
    import numpy as np

    h, w = roi_bgr.shape[:2]
    if h < 16 or w < 16:
        return None
    rx = max(1, w // 4)
    ry = max(1, h // 4)
    rect = (rx, ry, max(2, w - 2 * rx), max(2, h - 2 * ry))
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi_bgr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    out = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0)
    return out.astype(np.uint8)


def segment_local(frame_bgr, req: ClickRequest) -> ClickResult:
    """Fit an OBB to the animal under *req*, using OpenCV only.

    Needs no model, no download and no torch, and runs in the calling process
    in a few milliseconds — which is why it is both the default backend and the
    reason the Click-to-Box button is never disabled.
    """
    backend = "local"
    if frame_bgr is None or getattr(frame_bgr, "ndim", 0) != 3:
        return _failed(NO_FRAME, backend)
    height, width = frame_bgr.shape[:2]
    if not (0 <= req.x < width and 0 <= req.y < height):
        return _failed(OUT_OF_FRAME, backend)

    if req.roi is not None:
        x0, y0, x1, y1 = (int(v) for v in req.roi)
        x0, x1 = max(0, min(x0, x1)), min(width, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(height, max(y0, y1))
        windows = [(x0, y0, x1, y1)]
    else:
        base = float(req.radius_hint or DEFAULT_RADIUS)
        windows = []
        for step in range(3):
            r = min(base * (2 ** step), float(MAX_RADIUS))
            windows.append(_window(req.x, req.y, r, width, height))
            if r >= MAX_RADIUS:
                break

    obb = None
    why = NOTHING_FOUND
    win = windows[0]
    for win in windows:
        x0, y0, x1, y1 = win
        if x1 - x0 < 8 or y1 - y0 < 8:
            why = OUT_OF_FRAME
            continue
        roi = frame_bgr[y0:y1, x0:x1]
        at_local = (req.x - x0, req.y - y0)
        obb, why = mask_to_obb(_otsu_mask(roi, at_local), at_local)
        if obb is not None:
            break
        # Only a blob running off the window is worth a wider window; every
        # other failure would reproduce identically at any size.
        if why != TOUCHES_EDGE:
            break

    if obb is None:
        x0, y0, x1, y1 = win
        if x1 - x0 >= 16 and y1 - y0 >= 16:
            at_local = (req.x - x0, req.y - y0)
            cut = _grabcut_mask(frame_bgr[y0:y1, x0:x1], at_local)
            if cut is not None:
                obb, grab_why = mask_to_obb(cut, at_local)
                if obb is None:
                    why = grab_why if why == TOUCHES_EDGE else why

    if obb is None:
        return _failed(why, backend)

    x0, y0 = win[0], win[1]
    cx, cy, w, h, theta = obb
    placed = (cx + x0, cy + y0, w, h, theta)
    ax1, ay1, ax2, ay2 = obb_to_aabb(placed, width, height)
    bbox = (
        int(math.floor(ax1)),
        int(math.floor(ay1)),
        int(math.ceil(ax2)),
        int(math.ceil(ay2)),
    )
    if bbox[2] - bbox[0] < MIN_BODY_PX or bbox[3] - bbox[1] < MIN_BODY_PX:
        return _failed(NOTHING_FOUND, backend)

    return ClickResult(
        obb=placed,
        bbox=bbox,
        backend=backend,
        detail="Fitted a %.0f x %.0f px box from your click." % (w, h),
    )


#: Backends that answer synchronously in this process.  ``local`` is the only
#: one YORU ships, and it needs nothing installed — which is the property that
#: lets the Click-to-Box button never be disabled.
_IMPLEMENTATIONS = {"local": segment_local}


def segment(frame_bgr, req: ClickRequest, backend: str = "local") -> ClickResult:
    """Dispatch to a backend, falling back to ``local`` if it is not available."""
    return _IMPLEMENTATIONS.get(backend, segment_local)(frame_bgr, req)
