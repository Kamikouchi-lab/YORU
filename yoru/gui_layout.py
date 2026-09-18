# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Window shell shared by every DearPyGui screen: fonts, sizing, saved layout.

Every YORU GUI used to open the same way::

    dpg.configure_app(init_file="./logs/custom_layout_x.ini", docking=True, ...)
    dpg.create_viewport(title=..., width=1000, height=800,
                        max_width=1000, max_height=800)

which broke in four separate ways, all of them visible to the user:

* **Non-ASCII text was unreadable.**  DearPyGui's built-in font is ProggyClean,
  an ASCII-only bitmap face.  A Japanese directory name, a class label with a
  macron, a "µm" in a status line -- all of it rendered as blanks or boxes.
  Nothing in YORU ever loaded another font, so there was no way around it.
  :func:`setup_fonts` loads a system UI font with the Japanese, Latin, Greek,
  punctuation and symbol ranges attached.

* **The window did not fit the screen it was on.**  1000x800 is taller than the
  work area of a 1366x768 laptop, and ``max_width``/``max_height`` pinned it
  there: the window could not be enlarged on a big monitor or maximised
  anywhere.  :func:`fit_to_screen` sizes the viewport from the actual desktop
  work area and drops the caps.

* **The saved layout was addressed by a number that moved.**  DearPyGui writes
  each window into the ini under its ImGui id, and YORU's windows were built
  with ``id=dpg.generate_uuid()``, so the id was "whatever counter value we
  happened to reach".  Add a texture or a theme ahead of the window and every
  key shifts.  ``config/custom_layout_analysis.ini`` in this checkout still
  shows the damage -- ``###24``/``###25`` from one build of the code and
  ``###67``/``###68`` from another, four sections for two windows, none of them
  matching what the code now creates.  :func:`window_id` builds a stable id
  instead.

* **Nothing positioned the windows.**  Once ``init_file`` is configured, ImGui
  owns window geometry, and the ``pos``/``width``/``height`` passed to
  ``dpg.window()`` are ignored even on a first run with no ini present.  Every
  window therefore opened auto-sized at (60, 60): on the analysis screen the
  second window sat exactly on top of the first, hiding it completely, with a
  third of the viewport left empty.  :meth:`GuiSession.finish` applies a
  fraction-of-viewport default layout on the first run, after the frame where
  ImGui will accept it.

The viewport's own size is not part of ImGui's ini at all, so it is kept here
in ``yoru_windows.ini`` next to the layout files, and restored -- clamped to
the current screen -- the next time that GUI opens.
"""

import atexit
import configparser
import logging
import os
import sys
from pathlib import Path

import dearpygui.dearpygui as dpg

from yoru.libs.paths import project_root

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Where layout state lives
# ---------------------------------------------------------------------------

#: Name of the configparser file holding viewport geometry for every GUI.
VIEWPORT_INI_NAME = "yoru_windows.ini"

_layout_dir_cache = None


def _user_state_dir():
    """Per-user fallback for when the checkout is not writable (pip install)."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(base) / "YORU"
    return Path(os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))) / "yoru"


def layout_dir():
    """Directory holding the layout files, created if need be.

    Prefers ``logs/`` in the project (where YORU has always kept them, and
    where ``.gitignore`` already excludes them) and falls back to a per-user
    directory when that is read-only -- an installed copy under
    ``site-packages`` is not somewhere we can write.
    """
    global _layout_dir_cache
    if _layout_dir_cache is not None:
        return _layout_dir_cache

    for candidate in (project_root() / "logs", _user_state_dir()):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".yoru-write-test"
            probe.touch()
            probe.unlink()
        except OSError:
            continue
        _layout_dir_cache = candidate
        return candidate

    # Last resort: the working directory.  Saving may fail, but nothing else
    # should, and a GUI that cannot remember its layout still has to open.
    logger.warning("No writable layout directory; window layout will not be saved.")
    _layout_dir_cache = Path.cwd()
    return _layout_dir_cache


def layout_ini_path(name):
    """Absolute path of the ImGui layout file for the GUI called *name*.

    Absolute on purpose.  The old ``"./logs/custom_layout_x.ini"`` was resolved
    against the working directory at save time, so launching YORU from anywhere
    but a checkout that already had a ``logs/`` directory meant ImGui silently
    wrote nothing and the layout was lost on every exit.
    """
    return str(layout_dir() / f"custom_layout_{name}.ini")


def _viewport_ini_path():
    return layout_dir() / VIEWPORT_INI_NAME


def _read_viewport_ini():
    parser = configparser.ConfigParser()
    try:
        parser.read(_viewport_ini_path(), encoding="utf-8")
    except (OSError, configparser.Error):
        logger.debug("Could not read %s", _viewport_ini_path(), exc_info=True)
    return parser


def load_window_prefs(name):
    """Saved viewport size and text scale for *name* (empty dict when unset)."""
    parser = _read_viewport_ini()
    if not parser.has_section(name):
        return {}
    prefs = {}
    for key, cast in (("width", int), ("height", int), ("text_scale", float)):
        try:
            prefs[key] = cast(parser.get(name, key))
        except (configparser.Error, ValueError):
            continue
    return prefs


def save_window_prefs(name, **values):
    """Merge *values* into the ``[name]`` section of the viewport ini."""
    parser = _read_viewport_ini()
    if not parser.has_section(name):
        parser.add_section(name)
    for key, value in values.items():
        parser.set(name, key, str(value))
    try:
        with open(_viewport_ini_path(), "w", encoding="utf-8") as handle:
            parser.write(handle)
    except OSError:
        logger.warning("Could not save window size to %s", _viewport_ini_path(), exc_info=True)


# ---------------------------------------------------------------------------
# Stable window ids
# ---------------------------------------------------------------------------


def window_id(title, key):
    """Label for ``dpg.window`` that keeps its ini key stable.

    ImGui identifies a window by everything after ``###`` in its label, and
    that identifier is the section name it writes into the ini.  Passing this
    together with ``use_internal_label=False`` fixes the key to *key*, so the
    saved layout survives both a renamed title and any change to how many items
    are created before the window::

        with dpg.window(**window_kwargs("Analyzing Movies", "analysis_movies")):
    """
    return f"{title}###yoru_{key}"


def window_kwargs(title, key, **extra):
    """``dpg.window`` arguments giving *title* a stable ini key and tag."""
    kwargs = {
        "label": window_id(title, key),
        "use_internal_label": False,
        "tag": key,
    }
    kwargs.update(extra)
    return kwargs


# ---------------------------------------------------------------------------
# Screen geometry
# ---------------------------------------------------------------------------

#: Used when the desktop cannot be measured (headless CI, an unusual platform).
FALLBACK_SCREEN = (1600, 900)

#: Smallest viewport we will ask for; clamped again to whatever the screen has.
MIN_VIEWPORT = (820, 560)


def work_area():
    """Usable desktop size in pixels, with the taskbar excluded.

    DearPyGui 1.11 never marks the process DPI-aware, so Windows reports -- and
    positions windows in -- the same virtualised pixels regardless of the
    display scaling.  That makes this measurement directly comparable with the
    size we hand to ``create_viewport``, which is the whole point: a window
    sized from it fits the screen on a 150%-scaled laptop just as it does on a
    100% desktop.
    """
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            rect = wintypes.RECT()
            spi_getworkarea = 0x0030
            if ctypes.windll.user32.SystemParametersInfoW(
                spi_getworkarea, 0, ctypes.byref(rect), 0
            ):
                width = int(rect.right - rect.left)
                height = int(rect.bottom - rect.top)
                if width > 0 and height > 0:
                    return width, height
        except Exception:
            logger.debug("Could not read the desktop work area", exc_info=True)

    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), int(root.winfo_screenheight() * 0.94))
        root.destroy()
        if size[0] > 0 and size[1] > 0:
            return size
    except Exception:
        logger.debug("Could not read the screen size from Tk", exc_info=True)

    return FALLBACK_SCREEN




def fit_to_screen(width, height, margin=0.94):
    """Clamp a desired viewport size to the screen and centre it.

    Returns ``(width, height, x, y)``.  *margin* leaves a little of the desktop
    visible around the window rather than filling it edge to edge.
    """
    screen_w, screen_h = work_area()
    max_w = max(MIN_VIEWPORT[0], int(screen_w * margin))
    max_h = max(MIN_VIEWPORT[1], int(screen_h * margin))

    fitted_w = max(min(int(width), max_w), min(MIN_VIEWPORT[0], screen_w))
    fitted_h = max(min(int(height), max_h), min(MIN_VIEWPORT[1], screen_h))

    x = max(0, (screen_w - fitted_w) // 2)
    y = max(0, (screen_h - fitted_h) // 2)
    return fitted_w, fitted_h, x, y


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

#: Default font size, and not an arbitrary one.
#:
#: DearPyGui's built-in ProggyClean advances 7px per character at its 13px
#: size, and every fixed ``width=`` in YORU's forms was chosen against that:
#: ``width=150`` is "room for 21 characters".  The monospace faces below
#: advance exactly half their size, so 14px advances 7px too, and swapping the
#: font in changes no measurement anywhere in the application.  Raising it
#: instead is what clips ``Run YORU Frame Capture`` down to ``Run YORU Frame
#: Capt``; that is what the Text size menu is for, per user rather than by
#: default.
BASE_FONT_SIZE = 14

#: Ranges added on top of the Japanese hint.  The hint covers ASCII, kana,
#: the common kanji and the fullwidth forms but stops there, so the characters
#: that turn up in scientific UI text -- degrees, micro, plus-minus, the
#: en/em dashes, arrows in a status line -- still need naming.
EXTRA_FONT_RANGES = (
    (0x00A0, 0x024F),  # Latin-1 supplement + Latin Extended-A: ° µ × ÷ é ü
    (0x0370, 0x03FF),  # Greek: α β μ σ Δ
    (0x2000, 0x206F),  # general punctuation: – — ' " …
    (0x2100, 0x22FF),  # letterlike, arrows, maths: ™ → ≤ ≥ ≠ ± ∞
    (0x25A0, 0x26FF),  # geometric shapes and symbols: ■ ● ▶ ★ ⚠
)


def _font_candidates():
    """Fonts worth trying, best first.

    Every one of these is **monospaced**, and that is the requirement, not a
    preference.  YORU's forms line their fields up by padding the label with
    spaces -- ``dpg.add_text("Project Name      ")`` -- which is exact under
    the built-in ProggyClean and ragged under anything proportional.  Swapping
    in Yu Gothic for its kanji coverage left every form in the application
    stepped by up to 30px.  BIZ UDGothic is Microsoft's universal-design
    monospace face (Windows 10 1809 and later); MS Gothic is the fallback that
    has shipped with every Japanese Windows there has ever been.
    """
    if sys.platform == "win32":
        fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        names = (
            "BIZ-UDGothicR.ttc",  # universal-design monospace, full JP coverage
            "msgothic.ttc",       # monospace, present on every Windows
            "consola.ttf",        # monospace but Latin only
            "cour.ttf",
        )
        return [fonts / name for name in names]

    if sys.platform == "darwin":
        return [
            Path(p)
            for p in (
                "/System/Library/Fonts/Osaka.ttf",  # Osaka-Mono, JP + monospace
                "/System/Library/Fonts/Menlo.ttc",
                "/System/Library/Fonts/Monaco.ttf",
            )
        ]

    return [
        Path(p)
        for p in (
            "/usr/share/fonts/opentype/noto/NotoSansMonoCJKjp-Regular.otf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        )
    ]


def find_ui_font():
    """First available Unicode-capable UI font, or ``None``."""
    for path in _font_candidates():
        try:
            if path.is_file():
                return str(path)
        except OSError:
            continue
    return None


#: Height of DearPyGui's built-in ProggyClean face, used when no system font
#: could be loaded and the layout still has to reserve room for a menu bar.
BUILTIN_FONT_SIZE = 13


def setup_fonts(size=None):
    """Load and bind a Unicode UI font.  Returns ``(font_item, size_px)``.

    Must run before ``setup_dearpygui()`` -- that is where the font atlas is
    built.  A missing font is not fatal: DearPyGui keeps its built-in face and
    the GUI still opens, only without the non-ASCII glyphs.
    """
    path = find_ui_font()
    if path is None:
        logger.warning(
            "No Unicode UI font found; non-ASCII text (Japanese paths, degree "
            "and micro signs) will not render."
        )
        return None, BUILTIN_FONT_SIZE

    if size is None:
        size = BASE_FONT_SIZE
    size_px = int(round(size))

    try:
        with dpg.font_registry():
            with dpg.font(path, size_px) as font:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Japanese)
                for first, last in EXTRA_FONT_RANGES:
                    dpg.add_font_range(first, last)
        dpg.bind_font(font)
    except Exception:
        logger.exception("Could not load the UI font %s", path)
        return None, BUILTIN_FONT_SIZE

    logger.info("UI font: %s at %dpx", path, size_px)
    return font, size_px


# ---------------------------------------------------------------------------
# The session object the GUIs drive
# ---------------------------------------------------------------------------

#: Text-size choices offered in the Window menu.
#:
#: ``set_global_font_scale`` scales the text and nothing else, so the largest
#: entry here is bounded by the tightest fixed-width control in the
#: application -- raise it and a button label starts getting cut off.
#: ``tests/test_gui_layout.py::TestLabelsFit`` holds the two together.
TEXT_SCALES = (("Small", 0.9), ("Normal", 1.0), ("Large", 1.15))

#: Frames :meth:`GuiSession.finish` will spend waiting for the window manager
#: to report the client size it really gave us.  Windows takes about five; the
#: cap only matters on a platform that never fires a resize at startup.
LAYOUT_SETTLE_FRAMES = 20


class GuiSession:
    """Opens, sizes, and remembers the window for one YORU screen.

    Typical use::

        session = GuiSession("analysis", "YORU - Video Analysis",
                             width=1320, height=900)
        session.begin()
        ...build windows with session.window_kwargs(...)...
        session.finish(default_layout={"analysis_movies": (0, 0, 0.5, 1.0)})

    *docking* is for screens that really have more than one window to arrange.
    A single-window screen passes ``fill_window`` to :meth:`finish` instead and
    gets a window that takes the whole viewport, which needs no saved layout
    and cannot be dragged into a corner and lost.
    """

    def __init__(self, name, title, width, height, docking=False,
                 min_width=None, min_height=None):
        self.name = name
        self.title = title
        self.requested_size = (width, height)
        self.docking = docking
        # A screen that needs more room than the general minimum -- Frame
        # Capture puts two panes side by side and clips them below 960 -- says
        # so here rather than letting the window shrink past what it can draw.
        self.min_size = (
            min_width if min_width is not None else MIN_VIEWPORT[0],
            min_height if min_height is not None else MIN_VIEWPORT[1],
        )

        self.had_saved_layout = False
        self.text_scale = 1.0
        self._default_layout = {}
        self._last_size = None
        self._font_px = BUILTIN_FONT_SIZE
        self._has_menu_bar = False
        self._pending_default_layout = False
        self._applied_rects = {}
        self._on_resize_hook = None

    # -- construction ---------------------------------------------------

    def begin(self):
        """Create the context and the viewport.  Call before building items."""
        dpg.create_context()

        prefs = load_window_prefs(self.name)
        self.text_scale = prefs.get("text_scale", 1.0)
        width = prefs.get("width", self.requested_size[0])
        height = prefs.get("height", self.requested_size[1])

        if self.docking:
            ini = layout_ini_path(self.name)
            self.had_saved_layout = os.path.exists(ini)
            # auto_save_init_file makes the save explicit rather than relying
            # on ImGui's shutdown write, which never happened when the process
            # was killed from the launcher.
            dpg.configure_app(
                init_file=ini,
                auto_save_init_file=True,
                docking=True,
                docking_space=True,
            )

        fitted_w, fitted_h, x, y = fit_to_screen(width, height)
        self._last_size = (fitted_w, fitted_h)
        dpg.create_viewport(
            title=self.title,
            width=fitted_w,
            height=fitted_h,
            x_pos=x,
            y_pos=y,
            # No max_*: the old 1000x800 cap meant the window could not be
            # enlarged on a large monitor, nor maximised at all.
            # Never larger than what we just fitted: a minimum bigger than the
            # screen is a window the user cannot shrink to see the desktop.
            min_width=min(self.min_size[0], fitted_w),
            min_height=min(self.min_size[1], fitted_h),
        )
        return self

    def window_kwargs(self, title, key, **extra):
        """``dpg.window`` arguments with a stable ini key (see :func:`window_id`)."""
        return window_kwargs(title, key, **extra)

    # -- the Window menu ------------------------------------------------

    def add_layout_menu(self, extra_builder=None):
        """Add the viewport menu bar carrying the Window menu.

        Everything this module does that a user might want to undo is reachable
        from here: the saved layout, the saved size, and the text scale.  The
        alternative -- deleting an ini by hand -- is not something a GUI should
        ask for.
        """
        self._has_menu_bar = True
        with dpg.viewport_menu_bar(tag=f"{self.name}_menu_bar"):
            with dpg.menu(label="Window"):
                dpg.add_menu_item(
                    label="Fit window to this screen",
                    callback=lambda: self.fit_window(),
                )
                dpg.add_menu_item(
                    label="Maximize window",
                    callback=lambda: dpg.maximize_viewport(),
                )
                dpg.add_separator()
                with dpg.menu(label="Text size"):
                    for label, scale in TEXT_SCALES:
                        dpg.add_menu_item(
                            label=label,
                            tag=f"{self.name}_text_{label.lower()}",
                            check=True,
                            default_value=abs(scale - self.text_scale) < 1e-6,
                            callback=lambda s, a, u: self.set_text_scale(u),
                            user_data=scale,
                        )
                dpg.add_separator()
                dpg.add_menu_item(
                    label="Save layout now",
                    callback=lambda: self.save(),
                )
                dpg.add_menu_item(
                    label="Reset layout to default",
                    callback=lambda: self.reset_layout(),
                )
            if extra_builder is not None:
                extra_builder()

    # -- finishing ------------------------------------------------------

    def finish(self, default_layout=None, fill_window=None, font_size=None,
               on_resize=None):
        """Build the font atlas, show the window, and place it.

        *default_layout* maps a window tag to ``(x, y, width, height)`` as
        fractions of the viewport's content area.  It is applied when no layout
        has been saved yet, and re-applied on resize for as long as the user
        has not rearranged anything.  *fill_window* is the shorthand for a
        screen with a single window: it takes the whole content area and loses
        the chrome that would let it be dragged out of it.

        ``fill_window`` deliberately does not go through
        ``dpg.set_primary_window``.  A primary window is pinned to (0, 0) of
        the viewport, which on a screen that also has the Window menu puts its
        first row of widgets underneath the menu bar.

        *on_resize* is for a screen that lays out its own contents on resize;
        the session owns the single resize callback DearPyGui allows and calls
        this after its own work.
        """
        self._default_layout = dict(default_layout or {})
        self._on_resize_hook = on_resize

        if fill_window is not None:
            self._default_layout[fill_window] = (0.0, 0.0, 1.0, 1.0)
            dpg.configure_item(
                fill_window,
                no_title_bar=True,
                no_resize=True,
                no_move=True,
                no_collapse=True,
                no_bring_to_front_on_focus=True,
            )

        _, self._font_px = setup_fonts(font_size)

        self._pending_default_layout = not self.had_saved_layout
        dpg.set_viewport_resize_callback(self._on_resize)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_global_font_scale(self.text_scale)

        # ImGui owns window geometry from the moment init_file is configured,
        # and it ignores the pos/size passed to dpg.window() -- including on a
        # first run with no ini to restore.  It does accept set_item_pos once a
        # frame has been drawn, which is why the default layout is applied
        # here rather than at construction time.
        #
        # Which frame matters: for the first few, the client size still reads
        # back as the size we asked for, and only once the window manager has
        # taken its decorations out of it (frame ~5 on Windows) does it report
        # the real one.  Laying out against the earlier number overhangs the
        # right and bottom edges by the width of the border.  The resize
        # callback fires exactly when the real size arrives, so that is what
        # triggers the layout; the loop is only here to give it the frames it
        # needs, and the fallback covers a platform that never resizes us.
        for _ in range(LAYOUT_SETTLE_FRAMES):
            dpg.render_dearpygui_frame()
            if not self._pending_default_layout:
                break
        if self._pending_default_layout:
            self._pending_default_layout = False
            self.apply_default_layout()

        atexit.register(self._save_viewport_size)
        return self

    # -- layout actions -------------------------------------------------

    def menu_bar_height(self):
        """Vertical room the viewport menu bar takes, in pixels.

        Derived rather than measured: a viewport menu bar carries no rect of
        its own to query (``get_item_state`` has no ``rect_size`` for it).
        ImGui sizes it as the font height plus the frame padding, so this is
        exact to within a pixel or two, and a pixel or two of slack above the
        topmost window is not something anyone can see.
        """
        if not self._has_menu_bar:
            return 0
        return int(round(self._font_px * self.text_scale)) + 10

    def content_region(self):
        """``(x, y, width, height)`` of the viewport area below the menu bar."""
        width = dpg.get_viewport_client_width()
        height = dpg.get_viewport_client_height()
        top = self.menu_bar_height()
        return 0, top, max(1, width), max(1, height - top)

    def apply_default_layout(self):
        """Place every window in ``default_layout`` proportionally."""
        if not self._default_layout:
            return
        x0, y0, width, height = self.content_region()
        applied = {}
        for tag, (x, y, w, h) in self._default_layout.items():
            if not dpg.does_item_exist(tag):
                logger.debug("Default layout names a missing window: %s", tag)
                continue
            pos = (x0 + int(x * width), y0 + int(y * height))
            size = (max(1, int(w * width)), max(1, int(h * height)))
            dpg.set_item_pos(tag, list(pos))
            dpg.configure_item(tag, width=size[0], height=size[1])
            applied[tag] = (pos, size)
        self._applied_rects = applied

    def _layout_is_untouched(self):
        """True while every window still sits where the default layout put it.

        Decides whether a viewport resize should re-tile the windows.  Leaving
        them alone means enlarging the window just adds empty space -- exactly
        the dead area the default layout exists to avoid -- but re-tiling an
        arrangement the user built by hand would be worse.  So: reflow until
        the first time they move something, then never again.

        The tolerance is for ImGui's own rounding and for the pixel or two a
        window shifts when it is clamped inside the viewport; a deliberate
        drag moves things much further than that.
        """
        if not self._applied_rects:
            return False
        tolerance = 6
        for tag, (pos, size) in self._applied_rects.items():
            if not dpg.does_item_exist(tag):
                return False
            current_pos = dpg.get_item_pos(tag)
            current_size = dpg.get_item_rect_size(tag)
            for expected, actual in zip(pos + size, list(current_pos) + list(current_size)):
                if abs(expected - actual) > tolerance:
                    return False
        return True

    def _defer(self, func, frames=2):
        """Run *func* a couple of frames from now.

        Menu callbacks run inside ``render_dearpygui_frame``, so they cannot
        render another frame themselves to let a viewport resize take effect.
        They schedule the follow-up instead.
        """
        try:
            dpg.set_frame_callback(dpg.get_frame_count() + frames, lambda s, a: func())
        except Exception:
            logger.debug("Could not defer %s; running it now", func, exc_info=True)
            func()

    def _resize_viewport_to_screen(self):
        width, height, x, y = fit_to_screen(*self.requested_size)
        dpg.set_viewport_width(width)
        dpg.set_viewport_height(height)
        dpg.set_viewport_pos([x, y])
        self._last_size = (width, height)

    def fit_window(self):
        """Resize and recentre the viewport for the screen it is on now.

        The one control that matters when a project moves between machines --
        a window sized on a 2560px-wide desktop is off the edge of a laptop,
        and no amount of dragging brings back the part that is past the right
        border.
        """
        self._resize_viewport_to_screen()
        # The client size only catches up with the request a frame later, and
        # the default layout is a fraction of it.
        self._defer(self.apply_default_layout)

    def set_text_scale(self, scale):
        """Apply and remember a text size from the Window menu."""
        self.text_scale = float(scale)
        dpg.set_global_font_scale(self.text_scale)
        for label, value in TEXT_SCALES:
            tag = f"{self.name}_text_{label.lower()}"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, abs(value - self.text_scale) < 1e-6)
        # The menu bar grows with the text, so the top of the content area
        # moves; windows the user has not placed themselves follow it.
        if self._layout_is_untouched():
            self._defer(self.apply_default_layout)
        save_window_prefs(self.name, text_scale=self.text_scale)

    def save(self):
        """Write the layout and the window size out now."""
        if self.docking:
            try:
                dpg.save_init_file(layout_ini_path(self.name))
            except Exception:
                logger.warning("Could not save the window layout", exc_info=True)
        self._save_viewport_size()

    def reset_layout(self):
        """Forget the saved layout and go back to the built-in arrangement.

        The escape hatch for a layout that has become unusable -- a window
        dragged mostly off-screen, or a size left over from a much larger
        monitor.  Without it the only fix is finding and deleting an ini file.
        """
        try:
            os.remove(layout_ini_path(self.name))
        except OSError:
            pass
        self.had_saved_layout = False
        self._resize_viewport_to_screen()

        def _restore():
            self.apply_default_layout()
            self.save()

        self._defer(_restore)

    # -- viewport size persistence --------------------------------------

    def _on_resize(self):
        # Only cached here; writing the ini on every pixel of a resize drag
        # would be hundreds of writes per second.  It goes to disk at exit and
        # whenever the user asks for it from the menu.
        self._last_size = (
            dpg.get_viewport_width(),
            dpg.get_viewport_height(),
        )
        if self._pending_default_layout:
            self._pending_default_layout = False
            self.apply_default_layout()
        elif self._layout_is_untouched():
            self.apply_default_layout()

        if self._on_resize_hook is not None:
            self._on_resize_hook()

    def _save_viewport_size(self):
        if not self._last_size:
            return
        width, height = self._last_size
        if width <= 0 or height <= 0:
            return
        save_window_prefs(self.name, width=int(width), height=int(height))
