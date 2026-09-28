# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

r"""Collapse the per-batch progress output of a training subprocess.

Ultralytics draws its progress bar by rewriting one terminal line: every update
is written as ``\r\x1b[K<line>`` with no newline in between.  The trainer runs
in a subprocess whose stdout is a pipe opened in universal-newline mode
(``Popen(..., text=True)``), and that mode translates every ``\r`` into ``\n``.
So each redraw reaches the GUI as a separate line, and echoing them verbatim
fills the console with one row per batch -- 161 rows per epoch instead of 1.

YOLOv5 draws the same way with tqdm itself (``\r<line>``), so it is collapsed
too.  tqdm also draws a finished bar once more when it closes, usually with a
slightly different rate, which would still cost two rows per bar.

:class:`ProgressPrinter` re-collapses them.  A line that is a *step* progress
update (batch ``n`` of ``N``, ``n < N``) is transient: on a terminal it is
redrawn in place, and when stdout is redirected to a file or another pipe it is
dropped.  The final ``N/N`` update that carries the epoch's losses is printed
permanently -- once, however many times it was drawn -- and so is every other
line.  Either way the scrollback keeps a single line per epoch.
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
_BAR_RE = re.compile(
    r"\d+%\s+[\u2501\u2578\u2500]+\s+(?P<done>\d+)/(?P<total>\d+)(?![/\d])"
)

# tqdm's own bar, which YOLOv5 draws with bar_format "{l_bar}{bar:10}{r_bar}":
#   "     60/299   3.54G   0.0677   0.0185  0.009944   51   640:  50%|#####     | 1/2 [00:00<00:00,  9.69it/s]"
# The bar is "#" and digits on a pipe that is not UTF-8, block glyphs otherwise.
_TQDM_RE = re.compile(r"\d+%\|[^|]*\|\s*(?P<done>\d+)/(?P<total>\d+)(?![/\d.])")

# The same bar counting bytes ("| 14.1M/14.1M [...]"), as when yolov5 fetches
# weights: only the percentage tells how far along it is.
_TQDM_PERCENT_RE = re.compile(r"(?P<percent>\d+)%\|[^|]*\|")

# train_torchvision.py, e.g. "Epoch [1/50] Step [10/161] Loss: 0.4231"
_STEP_RE = re.compile(r"\bStep\s*\[\s*(?P<done>\d+)\s*/\s*(?P<total>\d+)\s*\]")

_STEP_RES = (_BAR_RE, _TQDM_RE, _STEP_RE)

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
        # A finished bar draw not printed yet, as (label, line): tqdm may draw
        # it again on closing, and only one of the two may reach the scrollback.
        self._finished = None

    @staticmethod
    def clean(raw_line):
        """Strip ANSI sequences and trailing whitespace from a raw pipe line."""
        return _ANSI_RE.sub("", raw_line).rstrip()

    @staticmethod
    def step_progress(line):
        """Return ``(done, total)`` if *line* is a step-progress line, else None."""
        for regex in _STEP_RES:
            m = regex.search(line)
            if m is not None:
                return int(m.group("done")), int(m.group("total"))
        return None

    @staticmethod
    def bar_state(line):
        """Return ``(label, finished)`` if *line* draws a progress bar, else None.

        *label* is the text in front of the bar.  It is what tells two draws of
        the same bar from the first draw of the next one: a finished epoch bar
        drawn twice carries the same losses, the next epoch's another number.
        """
        for regex in _STEP_RES:
            m = regex.search(line)
            if m is not None:
                finished = int(m.group("done")) >= int(m.group("total"))
                return line[: m.start()].rstrip(), finished
        m = _TQDM_PERCENT_RE.search(line)
        if m is not None:
            return line[: m.start()].rstrip(), int(m.group("percent")) >= 100
        m = _BAR_RUN_RE.search(line)
        if m is not None:
            return line[: m.start()].rstrip(), set(m.group()) == {_FILLED}
        return None

    @classmethod
    def is_redraw(cls, line):
        """Is *line* a progress draw that a later one supersedes?"""
        state = cls.bar_state(line)
        return state is not None and not state[1]

    def write(self, line):
        """Print one cleaned line, collapsing it if it is a progress redraw."""
        if not line:
            self._blank_lines += 1
            return

        state = self.bar_state(line)
        if (
            state is not None and state[1]
            and self._finished is not None and self._finished[0] == state[0]
            and not self._blank_lines
        ):
            # The same finished bar drawn again (tqdm's close()): it takes
            # the place of the earlier draw instead of a row of its own.
            self._hold_finished(state[0], line)
            return

        self._commit_finished()
        if state is None:
            self._flush_blank_lines()
            self._write_final(line)
        elif state[1]:
            self._flush_blank_lines()
            self._hold_finished(state[0], line)
        else:
            # About to be overwritten by the next redraw: never spend a blank
            # line on it, and never let it become part of the scrollback.
            self._blank_lines = 0
            self._write_transient(line)

    def close(self):
        """Terminate any in-place line still on screen."""
        self._commit_finished()
        self._blank_lines = 0
        if self._transient_len:
            self._emit("\n")
            self._transient_len = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- internals ---------------------------------------------------------

    def _hold_finished(self, label, line):
        # Shown at once on a terminal, printed for good by the next line.
        self._finished = (label, line)
        self._write_transient(line)

    def _commit_finished(self):
        if self._finished is None:
            return
        line = self._finished[1]
        self._finished = None
        self._write_final(line)

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
