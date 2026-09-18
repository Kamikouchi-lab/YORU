"""Guards for the window shell: fonts, fitting, and the saved layout.

None of this needs a window on screen.  What it needs is that the rules the
shell depends on keep holding in the source -- that the layout ini is addressed
by a stable key, that the viewport is not capped again, and above all that no
label outgrows the fixed-width control it sits in, which is the failure mode a
font change reintroduces silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from yoru import gui_layout


GUI_MODULES = (
    "analysis_GUI.py",
    "config_creator_GUI.py",
    "create_labels_GUI.py",
    "evaluation_GUI.py",
    "grab_GUI.py",
    "realtime_yoru_GUI.py",
    "refine_GUI.py",
    "train_GUI.py",
)


def _gui_sources(repo_root: Path):
    for name in GUI_MODULES:
        path = repo_root / "yoru" / name
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def _calls(tree, *names):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) in names:
            yield node


def _kwarg(node, name):
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


# ---------------------------------------------------------------------------
# Fitting the screen
# ---------------------------------------------------------------------------


class TestFitToScreen:
    def test_a_window_bigger_than_the_screen_is_brought_inside_it(self, monkeypatch):
        monkeypatch.setattr(gui_layout, "work_area", lambda: (1366, 728))
        width, height, x, y = gui_layout.fit_to_screen(1920, 1200)
        assert width <= 1366 and height <= 728
        assert x >= 0 and y >= 0
        assert x + width <= 1366
        assert y + height <= 728

    def test_a_window_that_already_fits_keeps_its_size(self, monkeypatch):
        monkeypatch.setattr(gui_layout, "work_area", lambda: (2560, 1032))
        width, height, _, _ = gui_layout.fit_to_screen(1320, 900)
        assert (width, height) == (1320, 900)

    def test_it_is_centred(self, monkeypatch):
        monkeypatch.setattr(gui_layout, "work_area", lambda: (2000, 1000))
        width, height, x, y = gui_layout.fit_to_screen(1000, 600)
        assert x == (2000 - width) // 2
        assert y == (1000 - height) // 2

    def test_a_screen_smaller_than_the_minimum_still_gives_a_usable_window(
        self, monkeypatch
    ):
        # A 1024x600 netbook is below MIN_VIEWPORT; the window has to fit the
        # screen even so, or its bottom edge is somewhere the mouse cannot go.
        monkeypatch.setattr(gui_layout, "work_area", lambda: (1024, 600))
        width, height, x, y = gui_layout.fit_to_screen(1320, 900)
        assert width <= 1024 and height <= 600
        assert width > 0 and height > 0

    def test_the_work_area_is_a_plausible_screen(self):
        width, height = gui_layout.work_area()
        assert width >= 640 and height >= 400


# ---------------------------------------------------------------------------
# Stable ini keys
# ---------------------------------------------------------------------------


class TestWindowIdentity:
    def test_the_ini_key_survives_a_renamed_title(self):
        before = gui_layout.window_id("Analyzing Movies", "analysis_movies")
        after = gui_layout.window_id("Movie Analysis", "analysis_movies")
        assert before.split("###")[1] == after.split("###")[1]

    def test_two_windows_do_not_share_a_key(self):
        assert gui_layout.window_id("A", "one") != gui_layout.window_id("A", "two")

    def test_the_label_keeps_the_internal_one_out_of_the_key(self):
        kwargs = gui_layout.window_kwargs("Analyzing Movies", "analysis_movies")
        # Without this DearPyGui appends '###<uuid>' of its own and the key
        # moves whenever the number of items created before the window does.
        assert kwargs["use_internal_label"] is False
        assert kwargs["tag"] == "analysis_movies"

    def test_every_gui_window_is_built_with_a_stable_key(self, repo_root: Path):
        offenders = []
        for path, tree in _gui_sources(repo_root):
            for call in _calls(tree, "window", "add_window"):
                label = _kwarg(call, "label")
                internal = _kwarg(call, "use_internal_label")
                if _kwarg(call, "id") is not None:
                    offenders.append(f"{path.name}:{call.lineno}: id=... window")
                    continue
                if label is None and not call.args:
                    # **window_kwargs(...) expansion
                    if any(kw.arg is None for kw in call.keywords):
                        continue
                    offenders.append(f"{path.name}:{call.lineno}: window without a label")
                    continue
                if isinstance(label, ast.Constant) and "###" not in str(label.value):
                    if not (isinstance(internal, ast.Constant) and internal.value is False):
                        offenders.append(
                            f"{path.name}:{call.lineno}: label={label.value!r} has no ### key"
                        )
        assert not offenders, "windows whose saved layout will not survive a code change:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# What the GUI modules must not go back to
# ---------------------------------------------------------------------------


class TestViewportSetup:
    def test_no_gui_caps_the_window_size(self, repo_root: Path):
        offenders = []
        for path, tree in _gui_sources(repo_root):
            for call in _calls(tree, "create_viewport"):
                for capped in ("max_width", "max_height"):
                    if _kwarg(call, capped) is not None:
                        offenders.append(f"{path.name}:{call.lineno}: {capped}")
        assert not offenders, (
            "max_width/max_height stop the window being enlarged or maximised:\n"
            + "\n".join(offenders)
        )

    def test_no_gui_writes_its_layout_to_a_relative_path(self, repo_root: Path):
        offenders = []
        for path, tree in _gui_sources(repo_root):
            for call in _calls(tree, "configure_app"):
                init_file = _kwarg(call, "init_file")
                if isinstance(init_file, ast.Constant) and not Path(str(init_file.value)).is_absolute():
                    offenders.append(f"{path.name}:{call.lineno}: {init_file.value}")
        assert not offenders, (
            "a relative init_file is resolved against the working directory, so "
            "the layout is silently not saved when YORU is started elsewhere:\n"
            + "\n".join(offenders)
        )


# ---------------------------------------------------------------------------
# Labels against the controls that hold them
# ---------------------------------------------------------------------------

#: Advance of one character, in pixels, for a monospace face at
#: :data:`gui_layout.BASE_FONT_SIZE`.  The faces YORU loads are all half-em.
CHAR_PX = gui_layout.BASE_FONT_SIZE / 2

#: ImGui puts FramePadding.x on each side of a button label; YORU's themes all
#: set it to 6.
FRAME_PADDING_X = 6


def _fixed_width_labels(repo_root: Path):
    for path, tree in _gui_sources(repo_root):
        for call in _calls(tree, "add_button", "add_checkbox"):
            label = _kwarg(call, "label")
            width = _kwarg(call, "width")
            if not (isinstance(label, ast.Constant) and isinstance(label.value, str)):
                continue
            if not (isinstance(width, ast.Constant) and isinstance(width.value, int)):
                continue
            if width.value <= 0:  # 0 auto-sizes, negative fills: neither clips
                continue
            yield path.name, call.lineno, label.value, width.value


class TestLabelsFit:
    def test_every_fixed_width_button_fits_its_label(self, repo_root: Path):
        """The failure this catches was shipping.

        ``evaluation_GUI`` gave "Run YORU Frame Capture" a 150px button -- 16px
        short of the label -- so it read "Run YORU Frame Captur".  A font with
        a wider advance turns a handful of near-misses like that into a screen
        full of them, which is why it is checked rather than eyeballed.
        """
        offenders = []
        for name, lineno, label, width in _fixed_width_labels(repo_root):
            needed = len(label) * CHAR_PX + 2 * FRAME_PADDING_X
            if needed > width:
                offenders.append(
                    f"{name}:{lineno}: {label!r} needs {needed:.0f}px, has {width}px"
                )
        assert not offenders, "clipped labels:\n" + "\n".join(offenders)

    def test_labels_still_fit_at_the_largest_text_size_offered(self, repo_root: Path):
        """The Text size menu must not offer a setting that breaks the layout.

        ``set_global_font_scale`` scales the text and nothing else, so a scale
        the buttons have no room for clips them.  Either widen the button or
        drop the scale; do not ship an option that visibly breaks.
        """
        largest = max(scale for _, scale in gui_layout.TEXT_SCALES)
        offenders = []
        for name, lineno, label, width in _fixed_width_labels(repo_root):
            needed = (len(label) * CHAR_PX) * largest + 2 * FRAME_PADDING_X
            if needed > width:
                offenders.append(
                    f"{name}:{lineno}: {label!r} needs {needed:.0f}px at x{largest}, has {width}px"
                )
        assert not offenders, (
            f"labels clipped at the largest offered text scale (x{largest}):\n"
            + "\n".join(offenders)
        )


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------


class TestFonts:
    def test_a_ui_font_is_available_on_this_machine(self):
        path = gui_layout.find_ui_font()
        assert path is not None, (
            "no candidate font found; the GUI would fall back to the ASCII-only "
            "built-in face and non-ASCII text would not render"
        )
        assert Path(path).is_file()

    def test_the_japanese_range_is_asked_for(self):
        # Everything else follows from this: without the hint the atlas has no
        # kana or kanji and a Japanese path renders as blanks.
        import dearpygui.dearpygui as dpg

        assert hasattr(dpg, "mvFontRangeHint_Japanese")

    def test_the_extra_ranges_cover_the_signs_a_lab_ui_uses(self):
        covered = set()
        for first, last in gui_layout.EXTRA_FONT_RANGES:
            assert first < last
            covered.update(range(first, last + 1))
        for char in "°µ×±—≥→":
            assert ord(char) in covered, f"{char!r} is not in any font range"

    def test_the_default_size_matches_the_built_in_font_advance(self):
        """14px is what keeps every ``width=`` in the forms meaning what it did.

        ProggyClean advances 7px at 13px; a half-em monospace face advances
        7px at 14px.  Change this and the fixed-width controls need re-measuring
        (``TestLabelsFit`` will say so).
        """
        assert gui_layout.BASE_FONT_SIZE / 2 == 7.0


# ---------------------------------------------------------------------------
# Saved preferences
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_layout_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gui_layout, "_layout_dir_cache", tmp_path)
    return tmp_path


class TestWindowPrefs:
    def test_nothing_saved_reads_back_as_nothing(self, temp_layout_dir):
        assert gui_layout.load_window_prefs("analysis") == {}

    def test_a_size_survives_a_round_trip(self, temp_layout_dir):
        gui_layout.save_window_prefs("analysis", width=1100, height=700)
        assert gui_layout.load_window_prefs("analysis") == {
            "width": 1100,
            "height": 700,
        }

    def test_two_screens_keep_their_own_size(self, temp_layout_dir):
        gui_layout.save_window_prefs("analysis", width=1100, height=700)
        gui_layout.save_window_prefs("train", width=1240, height=920)
        assert gui_layout.load_window_prefs("analysis")["width"] == 1100
        assert gui_layout.load_window_prefs("train")["width"] == 1240

    def test_a_later_save_merges_rather_than_replaces(self, temp_layout_dir):
        gui_layout.save_window_prefs("analysis", width=1100, height=700)
        gui_layout.save_window_prefs("analysis", text_scale=1.15)
        prefs = gui_layout.load_window_prefs("analysis")
        assert prefs == {"width": 1100, "height": 700, "text_scale": 1.15}

    def test_a_corrupt_value_is_skipped_not_fatal(self, temp_layout_dir):
        (temp_layout_dir / gui_layout.VIEWPORT_INI_NAME).write_text(
            "[analysis]\nwidth = not-a-number\nheight = 700\n", encoding="utf-8"
        )
        # A hand-edited or truncated file must not stop the GUI opening.
        assert gui_layout.load_window_prefs("analysis") == {"height": 700}

    def test_an_unreadable_file_is_skipped_not_fatal(self, temp_layout_dir):
        (temp_layout_dir / gui_layout.VIEWPORT_INI_NAME).write_text(
            "this is not an ini file at all", encoding="utf-8"
        )
        assert gui_layout.load_window_prefs("analysis") == {}

    def test_the_layout_path_is_absolute_and_its_directory_exists(self, temp_layout_dir):
        path = Path(gui_layout.layout_ini_path("analysis"))
        assert path.is_absolute()
        assert path.parent.is_dir()

    def test_each_screen_gets_its_own_layout_file(self, temp_layout_dir):
        assert gui_layout.layout_ini_path("analysis") != gui_layout.layout_ini_path("train")


# ---------------------------------------------------------------------------
# Dynamic textures
# ---------------------------------------------------------------------------


class TestDynamicTextures:
    """A dynamic texture is four floats per pixel, and getting that wrong kills
    the process rather than drawing something odd.

    ``realtime_yoru_GUI`` seeded both of its textures with
    ``np.ones((width, height, 3), np.uint8)``.  DearPyGui sized its upload from
    the texture (``width * height * 4``) and read a quarter past the end of
    that buffer on the first rendered frame: an access violation, before the
    window was ever shown, on the screen the whole application exists for.
    """

    def test_the_shared_converter_gives_four_channels_per_pixel(self):
        import numpy as np

        from yoru.gui_base import frame_to_data_rgba

        data = frame_to_data_rgba(np.zeros((7, 5, 3), np.uint8))
        assert data.size == 7 * 5 * 4

    def test_no_texture_is_seeded_with_a_three_channel_array(self, repo_root: Path):
        offenders = []
        for path, tree in _gui_sources(repo_root):
            for call in _calls(tree, "add_dynamic_texture"):
                seed = _kwarg(call, "default_value")
                if not isinstance(seed, ast.Call):
                    continue
                if getattr(seed.func, "attr", None) not in ("ones", "zeros", "empty"):
                    continue
                shape = seed.args[0] if seed.args else None
                if not isinstance(shape, ast.Tuple) or not shape.elts:
                    continue
                last = shape.elts[-1]
                if isinstance(last, ast.Constant) and last.value != 4:
                    offenders.append(
                        f"{path.name}:{call.lineno}: seeded with {last.value} channels, needs 4"
                    )
        assert not offenders, (
            "a dynamic texture seeded with too few channels crashes on the "
            "first rendered frame:\n" + "\n".join(offenders)
        )
