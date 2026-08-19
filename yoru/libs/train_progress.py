# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

r"""Collapse the per-batch progress output of a training subprocess.

Ultralytics draws its progress bar by rewriting one terminal line: every update
is written as ``\r\x1b[K<line>`` with no newline in between.  The trainer runs
in a subprocess whose stdout is a pipe opened in universal-newline mode
(``Popen(..., text=True)``), and that mode translates every ``\r`` into ``\n``.
So each redraw reaches the GUI as a separate line, and echoing them verbatim
fills the console with one row per batch -- 161 rows per epoch instead of 1.

:class:`ProgressPrinter` re-collapses them.  A line that is a *step* progress
update (batch ``n`` of ``N``, ``n < N``) is transient: on a terminal it is
redrawn in place, and when stdout is redirected to a file or another pipe it is
dropped.  Every other line -- including the final ``N/N`` update that carries
the epoch's losses -- is printed permanently.  Either way the scrollback keeps
a single line per epoch.
"""

import re
import shutil
import sys

__all__ = ["ProgressPrinter"]

# ESC [ ... <letter>  (colours, erase-to-end-of-line, cursor moves, ...)
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")

# Ultralytics / RT-DETR bar, e.g.
#   "  2/300  4.59G  0.4013  0.5818  0.214  13  640: 71% ━━━━━━━━╸─── 115/161 3.5it/s 31.0s<13.1s"
# The bar glyphs are U+2501 (heavy), U+2578 (half heavy) and U+2500 (light).
_BAR_RE = re.compile(r"\d+%\s+[\u2501\u2578\u2500]+\s+(\d+)/(\d+)(?![/\d])")

# train_torchvision.py, e.g. "Epoch [1/50] Step [10/161] Loss: 0.4231"
_STEP_RE = re.compile(r"\bStep\s*\[\s*(\d+)\s*/\s*(\d+)\s*\]")

# Any bar, including the ones drawn without a total (downloads of unknown
# size, streamed sources). Those carry no n/N, and TQDM fills them solid
# only when it closes -- which is what tells a final draw from a redraw.
_BAR_RUN_RE = re.compile(r"[\u2501\u2578\u2500]{4,}")
_FILLED = "\u2501"


class ProgressPrinter:
    """Echo training output, collapsing per-batch progress redraws.

    Args:
        stream (IO[str], optional): where to print. Defaults to ``sys.stdout``.
        in_place (bool, optional): redraw transient lines with a carriage
            return. Auto-detected from ``stream.isatty()`` when not given;
            transient lines are dropped entirely when it is False.
    """

    def __init__(self, stream=None, in_place=None):
        self.stream = sys.stdout if stream is None else stream
        if in_place is None:
            try:
                in_place = bool(self.stream.isatty())
            except Exception:
                in_place = False
        self.in_place = in_place
        # Width of the transient line currently sitting on screen, 0 if none.
        self._transient_len = 0
        # Blank lines held back: the pipe emits one just before each bar starts
        # (the bar's leading "\r"), and printing it would cost a row per epoch.
        self._blank_lines = 0

    @staticmethod
    def clean(raw_line):
        """Strip ANSI sequences and trailing whitespace from a raw pipe line."""
        return _ANSI_RE.sub("", raw_line).rstrip()

    @staticmethod
    def step_progress(line):
        """Return ``(done, total)`` if *line* is a step-progress line, else None."""
        m = _BAR_RE.search(line) or _STEP_RE.search(line)
        if m is None:
            return None
        return int(m.group(1)), int(m.group(2))

    @classmethod
    def is_redraw(cls, line):
        """Is *line* a progress draw that a later one supersedes?"""
        progress = cls.step_progress(line)
        if progress is not None:
            return progress[0] < progress[1]
        run = _BAR_RUN_RE.search(line)
        if run is not None:
            return set(run.group()) != {_FILLED}
        return False

    def write(self, line):
        """Print one cleaned line, collapsing it if it is a progress redraw."""
        if not line:
            self._blank_lines += 1
            return

        if self.is_redraw(line):
            # About to be overwritten by the next redraw: never spend a blank
            # line on it, and never let it become part of the scrollback.
            self._blank_lines = 0
            self._write_transient(line)
        else:
            self._flush_blank_lines()
            self._write_final(line)

    def close(self):
        """Terminate any in-place line still on screen."""
        self._blank_lines = 0
        if self._transient_len:
            self._emit("\n")
            self._transient_len = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- internals ---------------------------------------------------------

    def _write_transient(self, line):
        if not self.in_place:
            return
        width = self._terminal_width()
        if width and len(line) >= width:
            # A wrapped line cannot be redrawn in place: the carriage return
            # would only rewind to the start of the last visual row.
            line = line[: width - 1]
        pad = " " * max(0, self._transient_len - len(line))
        self._emit("\r" + line + pad)
        self._transient_len = len(line)

    def _write_final(self, line):
        if self._transient_len:
            pad = " " * max(0, self._transient_len - len(line))
            self._emit("\r" + line + pad + "\n")
            self._transient_len = 0
        else:
            self._emit(line + "\n")

    def _flush_blank_lines(self):
        if not self._blank_lines:
            return
        count, self._blank_lines = self._blank_lines, 0
        if self._transient_len:
            self._emit("\n")
            self._transient_len = 0
        self._emit("\n" * count)

    def _emit(self, text):
        # Console writes must never take down the monitoring thread.
        try:
            self.stream.write(text)
            self.stream.flush()
        except Exception:
            pass

    @staticmethod
    def _terminal_width():
        try:
            return shutil.get_terminal_size().columns
        except Exception:
            return 0
