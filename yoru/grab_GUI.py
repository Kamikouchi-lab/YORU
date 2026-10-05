# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Frame Capture — the window that turns a video into frames to label.

Three things about this screen are load-bearing and easy to undo by accident:

* **It is three dockable windows, not one fixed pane layout.**  Preview, Save
  Frame and Automatic Extraction live in a docking space, so they can be
  resized against each other, re-tiled, or dragged on top of one another into
  a tab bar, and the arrangement comes back next time -- the same deal every
  other YORU screen with more than one thing to look at offers.

  What made a saved layout unusable here before was not the docking.  It was a
  viewport pinned to its start size by ``max_width``/``max_height``: a restored
  layout shorter than its content hid the lower half of the controls behind a
  scrollbar with no way to grow the window, and dragging an edge fought
  DearPyGui, which kept forcing the size back inside its own limits.
  ``GuiSession`` drops that clamp, keeps the layout in a file that
  ``Window > Reset layout to default`` deletes, and each window now carries
  only the controls that belong to it, so a window too short for them scrolls
  its own content instead of hiding somebody else's.
* **The preview is sized from its own window, not from the viewport.**  A
  docked window is resized by dragging a splitter and a floating one by its
  own edge; neither fires the viewport resize callback, so
  :meth:`grab_gui._fit_layout` runs from the render loop instead, and touches
  DearPyGui only when the size it computes has actually moved.
* **Extraction runs on a worker thread and talks back through a dict.**  The
  thread never calls DearPyGui; it writes its progress into ``_extract_state``
  and the render loop copies that into the widgets.  Only the render loop
  touches the GUI, and only the worker touches its own ``cv2.VideoCapture``.
"""

import os
import threading

import cv2
import dearpygui.dearpygui as dpg

from yoru.gui_base import apply_default_theme, frame_to_data_rgba, process_frame as _process_frame
from yoru.gui_layout import GuiSession
from yoru.gui_lifecycle import run_gui
from yoru.libs import frame_extraction
from yoru.libs.file_operation_grab import file_dialog_tk
from yoru.libs.gui_error import GuiErrorMixin


class grab_gui(GuiErrorMixin):
    # The preview texture is allocated once and the image item is scaled to
    # whatever the window currently affords; a texture cannot be resized in
    # place, and reallocating it mid-drag is exactly the kind of work that
    # makes a resize stutter.
    PREVIEW_TEXTURE = 600
    MIN_PREVIEW = 220
    # What the Preview window spends on everything that is not the image: its
    # title bar and padding, the source row and its status line above it, and
    # the slider, the transport row and Quit below it.  Measured against the
    # default theme at text scale 1.0 and scaled with the text, so a larger
    # text size shrinks the image rather than pushing the transport row out of
    # sight.
    PREVIEW_CHROME_H = 200
    # Window padding on both sides, plus room for a vertical scrollbar.
    PREVIEW_CHROME_W = 36
    # The same, for the windows whose only width-sensitive item is wrapped
    # text.
    TEXT_CHROME_W = 34

    # Text fields whose caret the arrow-key shortcuts must leave alone: typing
    # a frame name or a range used to step the video instead of moving the
    # caret.
    TEXT_INPUTS = ("save_name", "extract_count", "extract_start", "extract_stop")

    def __init__(self, m_dict={}):
        print("Grab-gui")
        self.m_dict = m_dict
        self.file_path = "./web/image/YORU_logo.png"

        if self.file_path:
            print("File: " + self.file_path)
        else:
            print("Open-file dialog")

        self.vid = cv2.imread(self.file_path)
        self.height, self.width, _ = self.vid.shape
        self.framecount = 1
        self.current_frame_num = 1
        self.frame = self.vid
        self.process_frame()
        self.grab_count = 0
        self.speed = 1
        self.grab_dir = ""
        self.has_video = False

        # Shared with the extraction worker.  Plain dict assignments are atomic
        # enough under the GIL for a progress report, and the alternative -- a
        # lock held across a cv2 read -- would stall the render loop.
        self._extract_thread = None
        self._extract_stop = False
        self._extract_state = self._idle_extract_state()

        # The last geometry _fit_layout handed to DearPyGui.  Kept so that the
        # render loop, which recomputes it every frame, can tell when nothing
        # has moved and do nothing.
        self._preview_side = 0
        self._wraps = {}

    @staticmethod
    def _idle_extract_state():
        return {
            "running": False,
            "finished": False,
            "applied": True,
            "fraction": 0.0,
            "overlay": "",
            "text": "",
            "saved": 0,
        }

    def process_frame(self):
        self.frame_re = _process_frame(self.frame, self.PREVIEW_TEXTURE)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def gui_configure(self):
        # Docking, like the other screens with more than one window: the three
        # windows below are arranged by the user -- side by side, stacked, or
        # tabbed on top of each other -- and the arrangement is restored next
        # time.  Window > Reset layout to default puts it back.
        self.session = GuiSession(
            "grab", "YORU - Frame Capture", width=1240, height=860,
            docking=True,
            # Small enough for a laptop, wide enough that the default
            # side-by-side arrangement still gets its contents in rather than
            # clipping them.
            min_width=960, min_height=720,
        )
        self.session.begin()

        # Theme
        apply_default_theme()
        self.session.add_layout_menu()

        # GUI-settings
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=self.PREVIEW_TEXTURE,
                height=self.PREVIEW_TEXTURE,
                default_value=self.frame_to_data(self.frame_re),
                tag="imwin_tag0",
            )

        # Three windows rather than one: each docks on its own, so the preview
        # can be given the whole left half, or the two control panes dropped on
        # top of each other into a single tab bar out of its way.
        #
        # None of them closes.  A window closed here is a pane of this screen
        # gone -- the layout menu restores an arrangement, not a window that
        # was shut -- and hiding one is what docking it into a tab behind
        # another already does, reversibly.
        with dpg.window(
            **self.session.window_kwargs("Preview", "grab_preview", no_close=True)
        ):
            self._build_source_row()
            self._build_preview_pane()
            self._build_footer()

        with dpg.window(
            **self.session.window_kwargs("Save Frame", "grab_save", no_close=True)
        ):
            self._build_save_pane()

        with dpg.window(
            **self.session.window_kwargs(
                "Automatic Extraction", "grab_extract", no_close=True
            )
        ):
            self._build_extract_pane()

        # Shortcuts through DearPyGui's own handlers rather than a pynput
        # listener: a pynput listener is a global OS hook, so Left/Right/Alt
        # stepped and grabbed frames even while another application had focus.
        # Key *release* (not press) gives one action per keystroke instead of
        # repeating every rendered frame.
        with dpg.handler_registry():
            dpg.add_key_release_handler(
                dpg.mvKey_Right, callback=lambda: self.advance_frame_bt()
            )
            dpg.add_key_release_handler(
                dpg.mvKey_Left, callback=lambda: self.reverse_frame_bt()
            )
            dpg.add_key_release_handler(
                dpg.mvKey_Alt, callback=lambda: self.grab_btn_cb()
            )

        # setup
        self.session.finish(default_layout={
            "grab_preview": (0.0, 0.0, 0.62, 1.0),
            "grab_save": (0.62, 0.0, 0.38, 0.44),
            "grab_extract": (0.62, 0.44, 0.38, 0.56),
        })
        self._fit_layout()

    def _build_source_row(self):
        with dpg.group(horizontal=True):
            dpg.add_text(default_value="Video Path")
            dpg.add_input_text(
                tag="video_path", readonly=True, hint="Path/to/movie", width=-124
            )
            dpg.add_button(
                label="Select Video",
                width=116,
                callback=lambda: self.file_open(),
                enabled=True,
            )
        # Every window says for itself what went wrong in it.  Once they can be
        # docked into separate tabs, a message left in another window's status
        # line is a message nobody sees.
        dpg.add_text(
            tag="source_status", default_value="", wrap=520, color=(150, 170, 200)
        )

    def _build_preview_pane(self):
        dpg.add_image("imwin_tag0", tag="preview_image", width=520, height=520)
        dpg.add_slider_int(
            default_value=0,
            min_value=0,
            max_value=max(0, self.framecount - 2),
            tag="frame_bar",
            width=520,
            callback=lambda: self.slide_bar_cb(),
            enabled=False,
        )
        with dpg.group(horizontal=True):
            dpg.add_checkbox(
                label="Streaming",
                default_value=False,
                tag="streamingChkBox",
                callback=lambda: self.stream_cb(),
                enabled=False,
            )
            dpg.add_spacer(width=8)
            dpg.add_button(
                tag="minus_frame",
                label="< Prev",
                callback=lambda: self.reverse_frame_bt(),
            )
            dpg.add_button(
                tag="plus_frame",
                label="Next >",
                callback=lambda: self.advance_frame_bt(),
            )
            dpg.add_spacer(width=8)
            dpg.add_text(default_value="Speed")
            dpg.add_combo(
                items=[1, 2, 5, 10, 20, 50, 100, 200, 500],
                tag="speed_list",
                default_value=1,
                width=70,
                callback=lambda: self.list_of_speed(),
            )
            dpg.add_spacer(width=8)
            dpg.add_text(tag="frame_pos", default_value="Frame 0 / 0")

    def _build_save_pane(self):
        # No section header of its own: the window's title bar -- or its tab,
        # once it is docked on top of another window -- already carries the
        # name.
        dpg.add_text(default_value="Save Directory")
        with dpg.group(horizontal=True):
            dpg.add_input_text(
                tag="grab_path",
                readonly=True,
                hint="Path/to/save/frame",
                width=-92,
            )
            dpg.add_button(
                label="Select",
                width=84,
                callback=lambda: self.select_grab_dir(),
                enabled=True,
            )
        dpg.add_text(default_value="Frame Name")
        dpg.add_input_text(
            tag="save_name", default_value="", width=-92, hint="Save frame name"
        )
        dpg.add_spacer(height=2)
        dpg.add_button(
            label="Grab Current Frame  (Alt)",
            tag="grab_btn",
            width=-1,
            height=30,
            callback=lambda: self.grab_btn_cb(),
        )
        with dpg.group(horizontal=True):
            dpg.add_text(
                tag="count_frames",
                default_value=f"{self.grab_count} frames grabbed",
            )
            dpg.add_spacer(width=8)
            dpg.add_button(label="Reset Count", callback=lambda: self.count_reset_bt())
        # Where a frame grabbed by hand reports itself.  The extraction window
        # keeps a status line of its own: two windows that can end up in
        # different tabs cannot share one.
        dpg.add_text(tag="grab_status", default_value="", wrap=380)

    def _build_extract_pane(self):
        dpg.add_text(
            tag="extract_hint",
            default_value="Pick frames across the video the way DeepLabCut does.",
            wrap=380,
            color=(150, 170, 200),
        )
        dpg.add_spacer(height=2)
        with dpg.group(horizontal=True):
            dpg.add_text(default_value="Frames to pick")
            dpg.add_spacer(width=8)
            dpg.add_input_int(
                tag="extract_count",
                default_value=frame_extraction.DEFAULT_COUNT,
                min_value=1,
                min_clamped=True,
                width=110,
                step=1,
            )
        with dpg.group(horizontal=True):
            dpg.add_text(default_value="Algorithm     ")
            dpg.add_spacer(width=8)
            dpg.add_combo(
                items=list(frame_extraction.ALGORITHMS),
                tag="extract_algo",
                default_value=frame_extraction.ALGORITHMS[0],
                width=150,
            )
        with dpg.tooltip("extract_algo"):
            dpg.add_text(
                "uniform: frames drawn at random from the range. Fast, and "
                "it mirrors how often each thing actually happens.\n\n"
                "kmeans: frames clustered by appearance, one frame taken "
                "per cluster. Slower -- it reads the range once -- but rare "
                "postures survive the sample.",
                wrap=330,
            )
        with dpg.group(horizontal=True):
            dpg.add_text(default_value="Video range   ")
            dpg.add_spacer(width=8)
            dpg.add_input_float(
                tag="extract_start",
                default_value=0.0,
                min_value=0.0,
                max_value=1.0,
                min_clamped=True,
                max_clamped=True,
                format="%.2f",
                step=0.05,
                width=110,
            )
            dpg.add_text(default_value="to")
            dpg.add_input_float(
                tag="extract_stop",
                default_value=1.0,
                min_value=0.0,
                max_value=1.0,
                min_clamped=True,
                max_clamped=True,
                format="%.2f",
                step=0.05,
                width=110,
            )
        with dpg.tooltip("extract_start"):
            dpg.add_text(
                "Fractions of the video, so 0.25 to 0.75 means the middle "
                "half. Use them to skip the handling at the start of a "
                "recording.",
                wrap=330,
            )
        dpg.add_spacer(height=4)
        with dpg.group(horizontal=True):
            dpg.add_button(
                label="Extract Frames",
                tag="extract_btn",
                width=170,
                height=30,
                callback=lambda: self.extract_btn_cb(),
            )
            dpg.add_spacer(width=8)
            dpg.add_button(
                label="Stop",
                tag="extract_stop_btn",
                width=90,
                height=30,
                enabled=False,
                callback=lambda: self.extract_stop_cb(),
            )
        dpg.add_progress_bar(
            tag="extract_progress", default_value=0.0, width=-1, overlay=""
        )
        dpg.add_text(tag="extract_status", default_value="", wrap=380)

    def _build_footer(self):
        dpg.add_separator()
        dpg.add_button(
            label="Quit",
            tag="quit_btn",
            width=110,
            height=30,
            callback=lambda: self.quit_cb(),
            enabled=True,
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _fit_layout(self):
        """Fit the preview and the wrapped texts to the windows holding them.

        Runs once at start-up and then once per rendered frame, because what
        it depends on changes without the viewport changing: a docked window
        is resized by dragging the splitter between it and its neighbour, and
        a floating one by its own edge, and neither of those is a viewport
        resize.  Every frame is often enough to follow a drag and cheap enough
        to afford -- provided nothing here decodes a frame or reallocates a
        texture, and provided DearPyGui is left alone while the numbers are
        unchanged, which is what ``_preview_side`` and ``_wraps`` are for.
        """
        scale = getattr(self.session, "text_scale", 1.0) or 1.0

        width, height = self._window_size("grab_preview")
        if width:
            side = min(
                width - self.PREVIEW_CHROME_W,
                height - int(self.PREVIEW_CHROME_H * scale),
            )
            side = max(self.MIN_PREVIEW, int(side))
            if side != self._preview_side:
                self._preview_side = side
                dpg.configure_item("preview_image", width=side, height=side)
                dpg.configure_item("frame_bar", width=side)
            self._fit_wrap("source_status", width - self.PREVIEW_CHROME_W)

        width, _ = self._window_size("grab_save")
        if width:
            self._fit_wrap("grab_status", width - self.TEXT_CHROME_W)

        width, _ = self._window_size("grab_extract")
        if width:
            self._fit_wrap("extract_hint", width - self.TEXT_CHROME_W)
            self._fit_wrap("extract_status", width - self.TEXT_CHROME_W)

    @staticmethod
    def _window_size(tag):
        """``(width, height)`` of a window, or ``(0, 0)`` before it is drawn.

        A window that has not been rendered yet has no rect to report, and
        sizing the preview against a zero would collapse it to the minimum for
        a frame and snap it back on the next one.
        """
        if not dpg.does_item_exist(tag):
            return 0, 0
        size = dpg.get_item_rect_size(tag)
        if not size or len(size) < 2:
            return 0, 0
        width, height = int(size[0]), int(size[1])
        if width < 2 or height < 2:
            return 0, 0
        return width, height

    def _fit_wrap(self, tag, width):
        """Wrap a hint or status text at the width of the window it sits in.

        A fixed wrap was fine while these lived in a pane of a known width.
        In a window the user resizes, one too wide puts a horizontal scrollbar
        under a one-line message, and one too narrow wastes half the window.
        """
        width = max(180, int(width))
        if self._wraps.get(tag) == width or not dpg.does_item_exist(tag):
            return
        self._wraps[tag] = width
        dpg.configure_item(tag, wrap=width)

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def run(self):
        run_gui(self, dpg, self.gui_configure, self.plot_callback, self._shutdown)

    def _shutdown(self):
        """Cancel extraction before the common lifecycle releases the window."""
        self._extract_stop = True
        thread = self._extract_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


    def plot_callback(self) -> None:
        self._fit_layout()
        self._pump_extraction()
        if not self.has_video:
            return
        if dpg.get_value("streamingChkBox"):
            if (
                int(self.speed) + dpg.get_value("frame_bar")
                > self.framecount - int(self.speed) - 1
            ):
                dpg.set_value("frame_bar", 0)
            else:
                dpg.set_value("frame_bar", int(self.speed) + dpg.get_value("frame_bar"))
            self.slide_bar_cb()

    # ------------------------------------------------------------------
    # Video source and navigation
    # ------------------------------------------------------------------

    def file_open(self):
        self.fd_tk = file_dialog_tk(self.m_dict)
        file_path = self.fd_tk.video_file_open()

        if not file_path:
            print("Failed open files")
            dpg.set_value("video_path", self.file_path if self.has_video else "")
            return
        print("File: " + file_path)

        vid = cv2.VideoCapture(file_path)
        if not vid.isOpened():
            vid.release()
            self._set_source_status(f"Could not open the video: {file_path}")
            self._report_error(
                "Failed to open video file",
                IOError(f"Could not open movie file: {file_path}"),
            )
            return

        if isinstance(self.vid, cv2.VideoCapture):
            self.vid.release()

        self.file_path = file_path
        self.vid = vid
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.framecount = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0
        self.has_video = True
        self.status, frame = self.vid.read()
        if self.status and frame is not None:
            self.frame = frame
            self.process_frame()
        print("Movie size: ", self.width, self.height)
        self._set_source_status("")
        dpg.configure_item("frame_bar", max_value=max(0, self.framecount - 2))
        dpg.set_value("frame_bar", 0)
        dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))
        dpg.enable_item("streamingChkBox")
        dpg.enable_item("frame_bar")
        self._update_frame_pos()

    def select_grab_dir(self):
        self.fd_tk = file_dialog_tk(self.m_dict)
        chosen = self.fd_tk.grab_dir_open()
        if chosen:
            self.grab_dir = chosen
        else:
            # A cancelled dialog blanks the field on its way out; putting the
            # directory back keeps the box and where frames actually go from
            # disagreeing.
            dpg.set_value("grab_path", self.grab_dir)

    def _show_current_frame(self):
        """Seek the preview capture to ``current_frame_num`` and draw it."""
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        self.status, frame = self.vid.read()
        if self.status and frame is not None:
            self.frame = frame
            self.process_frame()
            dpg.set_value("imwin_tag0", self.frame_to_data(self.frame_re))
        self._update_frame_pos()

    def _update_frame_pos(self):
        dpg.set_value(
            "frame_pos", f"Frame {self.current_frame_num} / {max(0, self.framecount - 1)}"
        )

    def slide_bar_cb(self):
        if not self.has_video:
            return
        self.current_frame_num = dpg.get_value("frame_bar")
        self._show_current_frame()

    def list_of_speed(self):
        tf = dpg.get_value("speed_list")
        self.speed = tf

    def _is_typing(self):
        """True while the caret is in a text field, so shortcuts stay quiet."""
        return any(
            dpg.does_item_exist(tag) and dpg.is_item_active(tag)
            for tag in self.TEXT_INPUTS
        )

    def advance_frame_bt(self):
        if not self.has_video or self._is_typing():
            return
        if self.current_frame_num < self.framecount - 2:
            self.current_frame_num = self.current_frame_num + int(self.speed)
            if self.current_frame_num >= self.framecount - 2:
                self.current_frame_num = self.framecount - 2
            self._show_current_frame()
            dpg.set_value("frame_bar", self.current_frame_num)
        else:
            print("final frame")

    def reverse_frame_bt(self):
        if not self.has_video or self._is_typing():
            return
        if self.current_frame_num > 0:
            self.current_frame_num = self.current_frame_num - int(self.speed)
            if self.current_frame_num <= 0:
                self.current_frame_num = 0
            self._show_current_frame()
            dpg.set_value("frame_bar", self.current_frame_num)
        else:
            print("initial frame")

    # ------------------------------------------------------------------
    # Saving frames
    # ------------------------------------------------------------------

    def _save_name(self):
        """The base name for saved frames, falling back to the video's own."""
        name = (dpg.get_value("save_name") or "").strip()
        if name:
            return name
        if self.has_video:
            return os.path.splitext(os.path.basename(self.file_path))[0]
        return ""

    def grab_btn_cb(self):
        # No _is_typing guard here, unlike the arrow keys: Alt types nothing
        # into a text field, so grabbing right after entering the frame name
        # has to keep working.
        if not self.has_video:
            self._set_grab_status("Select a video first.")
            return
        self.grab_name = self._save_name()
        if not self.grab_name or not self.grab_dir:
            print("Not selected file dir or name")
            self._set_grab_status("Select a save directory and a frame name first.")
            return
        self.grab_path = os.path.join(
            self.grab_dir,
            f"{self.grab_name}_{self.current_frame_num}.png",
        )
        print(self.grab_path)
        # Reuse the existing VideoCapture instead of creating a new one
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_num)
        ret, frame = self.vid.read()
        if ret and frame is not None and frame_extraction.imwrite(self.grab_path, frame):
            self.grab_count = self.grab_count + 1
            self._update_grab_count()
            self._set_grab_status(f"Saved {os.path.basename(self.grab_path)}")
        else:
            self._set_grab_status("Could not save that frame.")
            self._report_error(
                "Failed to grab frame",
                IOError(
                    f"Could not read or write frame {self.current_frame_num} "
                    f"from {self.file_path}"
                ),
            )

    def count_reset_bt(self):
        self.grab_count = 0
        self._update_grab_count()

    def _update_grab_count(self):
        dpg.set_value("count_frames", f"{self.grab_count} frames grabbed")

    def _set_status(self, text):
        """Say something in the Automatic Extraction window."""
        dpg.set_value("extract_status", text)

    def _set_grab_status(self, text):
        """Say something in the Save Frame window."""
        dpg.set_value("grab_status", text)

    def _set_source_status(self, text):
        """Say something in the Preview window, under the video path."""
        dpg.set_value("source_status", text)

    # ------------------------------------------------------------------
    # Automatic extraction
    # ------------------------------------------------------------------

    def extract_btn_cb(self):
        if self._extract_thread is not None and self._extract_thread.is_alive():
            return
        if not self.has_video:
            self._set_status("Select a video first.")
            return
        if not self.grab_dir:
            self._set_status("Select a save directory first.")
            return
        name = self._save_name()
        if not name:
            self._set_status("Enter a frame name first.")
            return

        count = dpg.get_value("extract_count")
        algo = dpg.get_value("extract_algo")
        start = dpg.get_value("extract_start")
        stop = dpg.get_value("extract_stop")
        problem = frame_extraction.validate(count, start, stop, algo)
        if problem:
            self._set_status(problem)
            return

        self._extract_stop = False
        self._extract_state = {
            "running": True,
            "finished": False,
            "applied": False,
            "fraction": 0.0,
            "overlay": "starting",
            "text": f"Extracting {int(count)} frames ({algo})...",
            "saved": 0,
        }
        dpg.configure_item("extract_btn", enabled=False)
        dpg.configure_item("extract_stop_btn", enabled=True)
        self._extract_thread = threading.Thread(
            target=self._extract_worker,
            args=(self.file_path, self.grab_dir, name, int(count), str(algo),
                  float(start), float(stop)),
            daemon=True,
        )
        self._extract_thread.start()

    def extract_stop_cb(self):
        if self._extract_thread is not None and self._extract_thread.is_alive():
            self._extract_stop = True
            self._extract_state["overlay"] = "stopping"

    def _extract_worker(self, video_path, out_dir, name, count, algo, start, stop):
        """Run one extraction. Touches ``_extract_state``, never DearPyGui."""
        state = self._extract_state

        def progress(done, total, phase):
            state["fraction"] = done / total if total else 0.0
            state["overlay"] = f"{phase} {done}/{total}"

        try:
            result = frame_extraction.extract_frames(
                video_path,
                out_dir,
                name,
                count=count,
                algo=algo,
                start=start,
                stop=stop,
                progress=progress,
                should_stop=lambda: self._extract_stop,
            )
            state["text"] = result.message
            state["saved"] = len(result.paths)
            state["fraction"] = 1.0 if result.ok else state["fraction"]
            state["overlay"] = "done" if result.ok else ""
        except Exception as exc:  # pragma: no cover - the library already guards
            state["text"] = f"Extraction failed: {exc}"
            state["overlay"] = ""
        finally:
            state["running"] = False
            state["finished"] = True

    def _pump_extraction(self):
        """Copy the worker's progress into the widgets, from the render loop."""
        state = self._extract_state
        if state["running"]:
            dpg.set_value("extract_progress", state["fraction"])
            dpg.configure_item("extract_progress", overlay=state["overlay"])
            dpg.set_value("extract_status", state["text"])
        elif state["finished"] and not state["applied"]:
            state["applied"] = True
            dpg.set_value("extract_progress", state["fraction"])
            dpg.configure_item("extract_progress", overlay=state["overlay"])
            dpg.set_value("extract_status", state["text"])
            dpg.configure_item("extract_btn", enabled=True)
            dpg.configure_item("extract_stop_btn", enabled=False)
            self.grab_count += state["saved"]
            self._update_grab_count()

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def stream_cb(self):
        if dpg.get_value("streamingChkBox"):
            dpg.disable_item("frame_bar")
            dpg.disable_item("minus_frame")
            dpg.disable_item("plus_frame")
        else:
            dpg.enable_item("frame_bar")
            dpg.enable_item("minus_frame")
            dpg.enable_item("plus_frame")

    def frame_to_data(self, frame):
        return frame_to_data_rgba(frame)

    def quit_cb(self):
        self.m_dict["quit"] = True

    def __del__(self):
        pass


def main():
    d = {}
    d["quit"] = False
    grabWin = grab_gui(d)
    grabWin.run()


if __name__ == "__main__":
    main()
