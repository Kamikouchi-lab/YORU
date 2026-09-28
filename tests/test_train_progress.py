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


# -- YOLOv5 (tqdm) -----------------------------------------------------------

V5_HEADER = ("%11s" * 7) % (
    "Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"
)
V5_VAL_HEADER = ("%22s" + "%11s" * 6) % (
    "Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95"
)
V5_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # yolov5 utils.general.TQDM_BAR_FORMAT


def _yolov5_epoch_writes(epoch: int, batches: int = 3, ascii=None) -> str:
    """What yolov5's train.py and val.py write for one epoch, drawn by tqdm.

    ``mininterval=0`` makes tqdm draw on every update, so the finished bar is
    drawn twice -- by the last update and again by close() -- as it is
    whenever the last batch comes more than 0.1 s after the previous draw.
    """
    from tqdm import tqdm

    out = io.StringIO()
    out.write("\n" + V5_HEADER + "\n")
    pbar = tqdm(range(batches), total=batches, bar_format=V5_BAR_FORMAT,
                file=out, mininterval=0, ascii=ascii)
    for i in pbar:
        pbar.set_description(("%11s" * 2 + "%11.4g" * 5) % (
            f"{epoch}/299", "3.54G", 0.0677 - i / 100, 0.0185, 0.009944, 51 - i, 640,
        ))
    pbar.close()
    for _ in tqdm(range(1), desc=V5_VAL_HEADER, bar_format=V5_BAR_FORMAT,
                  file=out, mininterval=0, ascii=ascii):
        pass
    out.write(("%22s" + "%11i" * 2 + "%11.3g" * 4) % ("all", 3, 6, 0.47, 0.667, 0.537, 0.113) + "\n")
    return out.getvalue()


def test_yolov5_bar_detection():
    ascii_bar = "     60/299      3.54G     0.0677     0.0185   0.009944         51        640:  50%|#####     | 1/2 [00:00<00:00,  9.69it/s]"
    unicode_bar = "     60/299      3.54G     0.0677     0.0185   0.009944         51        640:  50%|█████     | 1/2 [00:00<00:00,  9.69it/s]"
    done_bar = "     60/299      3.54G     0.0688    0.02932    0.01008          7        640: 100%|##########| 2/2 [00:00<00:00, 11.32it/s]"
    assert ProgressPrinter.step_progress(ascii_bar) == (1, 2)
    assert ProgressPrinter.step_progress(unicode_bar) == (1, 2)
    assert ProgressPrinter.step_progress("  0%|          | 0/2 [00:00<?, ?it/s]") == (0, 2)
    assert ProgressPrinter.is_redraw(ascii_bar) is True
    assert ProgressPrinter.is_redraw(done_bar) is False
    for line in (V5_HEADER, V5_VAL_HEADER,
                 "                   all          3          6       0.47      0.667      0.537      0.113"):
        assert ProgressPrinter.bar_state(line) is None


def test_yolov5_byte_bar_uses_its_percentage():
    """yolov5 fetching weights counts bytes: "14.1M/14.1M", no integer n/N."""
    mid = "yolov5s.pt:  40%|####      | 5.62M/14.1M [00:00<00:00, 30.1MB/s]"
    end = "yolov5s.pt: 100%|##########| 14.1M/14.1M [00:00<00:00, 30.1MB/s]"
    assert ProgressPrinter.is_redraw(mid) is True
    assert ProgressPrinter.is_redraw(end) is False


def test_yolov5_one_row_per_bar_on_a_terminal():
    for ascii in (True, False):  # "#" on a cp932 pipe, block glyphs on UTF-8
        raw = _yolov5_epoch_writes(0, ascii=ascii) + _yolov5_epoch_writes(1, ascii=ascii)
        rows = _render(_run(raw, in_place=True))
        assert rows == [
            "",
            V5_HEADER,
            rows[2],
            rows[3],
            rows[4],
            "",
            V5_HEADER,
            rows[7],
            rows[8],
            rows[9],
            "",
        ], rows
        for epoch, (train, val, metrics) in enumerate((rows[2:5], rows[7:10])):
            assert train.split()[0] == f"{epoch}/299" and "| 3/3 [" in train
            assert val.startswith(V5_VAL_HEADER) and "| 1/1 [" in val
            assert metrics.lstrip().startswith("all")


def test_yolov5_finished_bar_is_printed_once_when_redirected():
    raw = _yolov5_epoch_writes(0) + _yolov5_epoch_writes(1)
    # The premise: tqdm draws each finished bar twice.
    assert sum("| 3/3 [" in line for line in _through_pipe(raw)) == 4
    out = _run(raw, in_place=False)

    assert "\r" not in out
    rows = [r for r in out.split("\n") if r]
    assert len(rows) == 8  # per epoch: header, train bar, val bar, metrics
    assert sum("| 3/3 [" in r for r in rows) == 2
    assert sum("| 1/1 [" in r for r in rows) == 2
    assert "| 0/3 [" not in out and "| 2/3 [" not in out


def test_the_next_epoch_is_not_mistaken_for_a_second_draw():
    """Only a draw of the *same* bar replaces a finished one."""
    first = "      0/299      3.54G     0.0677: 100%|##########| 2/2 [00:00<00:00, 9.1it/s]"
    second = "      1/299      3.54G     0.0621: 100%|##########| 2/2 [00:00<00:00, 9.3it/s]"
    out = io.StringIO()
    printer = ProgressPrinter(stream=out, in_place=False)
    for line in (first, second):
        printer.write(line)
    printer.close()
    assert out.getvalue() == first + "\n" + second + "\n"


def test_epoch_header_and_summaries_are_never_redraws():
    for line in (HEADER, "3 epochs completed in 0.001 hours.",
                 "Results saved to runs/detect/train", "Epoch [1/50] Avg Loss: 0.42"):
        assert ProgressPrinter.is_redraw(line) is False
