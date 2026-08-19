"""The training console must keep one line per epoch, not one per batch.

Ultralytics redraws its progress bar with a carriage return; the training
subprocess pipe is read in universal-newline mode, so every redraw arrives as
its own line.  Echoing them verbatim produced ~160 rows per epoch.
"""

import io

from yoru.libs.train_progress import ProgressPrinter

HEADER = "      Epoch    GPU_mem  giou_loss   cls_loss    l1_loss  Instances       Size"
BAR = "\u2501" * 8 + "\u2578" + "\u2500" * 3


def _epoch_writes(epoch: int, batches: int = 5) -> str:
    """Bytes an ultralytics TQDM bar writes for one epoch."""
    parts = []
    for i in range(1, batches + 1):
        parts.append(
            "\r\x1b[K      {e}/300      4.59G     0.4013     0.5818      0.214"
            "         13        640: {p}% {bar} {i}/{n} 3.5it/s 31.0s".format(
                e=epoch, p=int(i / batches * 100), bar=BAR, i=i, n=batches
            )
        )
    parts.append("\n")  # TQDM.close() with leave=True
    return "".join(parts)


def _through_pipe(raw: str):
    """Split *raw* the way ``Popen(..., text=True)`` does: ``\r`` ends a line."""
    return raw.replace("\r\n", "\n").replace("\r", "\n").splitlines(True)


def _render(text: str):
    """Rows a terminal would show, applying carriage-return overwrites."""
    rows = []
    for chunk in text.split("\n"):
        row = ""
        for seg in chunk.split("\r"):
            row = seg + row[len(seg):]
        rows.append(row.rstrip())
    return rows


def _run(raw: str, in_place: bool) -> str:
    out = io.StringIO()
    printer = ProgressPrinter(stream=out, in_place=in_place)
    for raw_line in _through_pipe(raw):
        printer.write(printer.clean(raw_line))
    printer.close()
    return out.getvalue()


def test_pipe_really_splits_every_redraw():
    """The premise: without collapsing, one epoch costs one row per batch."""
    lines = _through_pipe(HEADER + "\n" + _epoch_writes(1, batches=5))
    assert len(lines) == 7  # header + a blank + 5 redraws


def test_step_progress_detection():
    bar = "      2/300      4.59G: 71% {b} 115/161 3.5it/s".format(b=BAR)
    assert ProgressPrinter.step_progress(bar) == (115, 161)
    assert ProgressPrinter.step_progress("Epoch [1/50] Step [10/161] Loss: 0.42") == (10, 161)
    assert ProgressPrinter.step_progress(HEADER) is None
    assert ProgressPrinter.step_progress("Epoch [1/50] Avg Loss: 0.4231") is None
    assert ProgressPrinter.step_progress("Results saved to runs/train/exp2") is None


def test_one_row_per_epoch_on_a_terminal():
    raw = HEADER + "\n" + _epoch_writes(1) + _epoch_writes(2)
    rows = _render(_run(raw, in_place=True))

    assert rows[0] == HEADER
    assert rows[1].startswith("      1/300") and "5/5" in rows[1]
    assert rows[2].startswith("      2/300") and "5/5" in rows[2]
    assert [r for r in rows[3:] if r] == []


def test_intermediate_rows_are_dropped_when_redirected():
    raw = HEADER + "\n" + _epoch_writes(1) + _epoch_writes(2)
    out = _run(raw, in_place=False)

    assert "\r" not in out  # a log file must not collect control characters
    rows = [r for r in out.split("\n") if r]
    assert len(rows) == 3
    assert rows[0] == HEADER
    assert "1/5" not in out and "4/5" not in out


def test_ansi_sequences_are_stripped():
    assert ProgressPrinter.clean("\x1b[K\x1b[34mhello\x1b[0m  \n") == "hello"


def test_plain_output_is_passed_through_unchanged():
    raw = "Ultralytics 8.4.21\n\nresults saved\n"
    assert _run(raw, in_place=True) == raw
    assert _run(raw, in_place=False) == raw


def test_torchvision_steps_collapse_into_the_epoch_summary():
    raw = "".join(
        "Epoch [1/50] Step [{i}0/161] Loss: 0.4\n".format(i=i) for i in range(1, 4)
    ) + "Epoch [1/50] Avg Loss: 0.4231\n"
    rows = [r for r in _render(_run(raw, in_place=True)) if r]
    assert rows == ["Epoch [1/50] Avg Loss: 0.4231"]


def test_unknown_total_bar_is_a_redraw_until_it_closes():
    """Downloads of unknown size draw a bar with no n/N; only close() fills it."""
    running = "Downloading yolo11n.pt: " + "\u2500" * 12 + " 1.2M 3.4MB/s 2.0s"
    closed = "Downloading yolo11n.pt: " + "\u2501" * 12 + " 5.4M 3.4MB/s 2.0s"
    assert ProgressPrinter.is_redraw(running) is True
    assert ProgressPrinter.is_redraw(closed) is False


def test_byte_download_bar_completion_is_kept():
    """A finished byte bar shows the size, not "n/N" -- it must still print."""
    mid = "yolo11n.pt: 40% " + "\u2501" * 4 + "\u2578" + "\u2500" * 7 + " 2.1/5.4MB 3.4MB/s 1.0s"
    end = "yolo11n.pt: 100% " + "\u2501" * 12 + " 5.4MB 3.4MB/s 2.0s"
    assert ProgressPrinter.is_redraw(mid) is True
    assert ProgressPrinter.is_redraw(end) is False


def test_epoch_header_and_summaries_are_never_redraws():
    for line in (HEADER, "3 epochs completed in 0.001 hours.",
                 "Results saved to runs/detect/train", "Epoch [1/50] Avg Loss: 0.42"):
        assert ProgressPrinter.is_redraw(line) is False
