# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Frame Capture — the window that turns a video into frames to label.

Two things about this window are load-bearing and easy to undo by accident:

* **It owns the whole viewport and lays itself out from the viewport size.**
  It used to be a floating, dockable window whose position and size were
  restored from ``logs/custom_layout_grab.ini``, inside a viewport that was
  pinned to its start size by ``max_width``/``max_height``.  That combination
  hid the lower half of the controls behind a scrollbar -- the saved layout was
  shorter than the content and the viewport could not be grown to make room --
  and dragging the window edge fought DearPyGui, which kept forcing the size
  back inside its own limits, so the window appeared to freeze.  The window is
  now the primary window, there is no init file and no size clamp, and
  :meth:`grab_gui._relayout` recomputes the preview and the two panes from the
  current viewport on every resize.
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
from yoru.libs import frame_extraction
from yoru.libs.file_operation_grab import file_dialog_tk


class grab_gui:
    # The preview texture is allocated once and the image item is scaled to
    # whatever the window currently affords; a texture cannot be resized in
    # place, and reallocating it mid-drag is exactly the kind of work that
    # makes a resize stutter.
    PREVIEW_TEXTURE = 600
    MIN_PREVIEW = 220
    SIDE_PANE_WIDTH = 430
    # The preview pane never goes narrower than its transport row, or the
    # speed box and the frame counter are quietly cut off the right of it.
    LEFT_MIN_WIDTH = 470
    # What the preview pane spends on the slider, the transport row and the
    # frame counter, plus the child window's own padding.
    LEFT_CHROME = 116
    HEADER_H = 100
    FOOTER_H = 62

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
        dpg.create_context()
        # No init file and no docking: this GUI is one window, and restoring a
        # saved layout for it only ever reintroduced the stale-size scrollbar.
        dpg.create_viewport(
            title="YORU - Frame Capture",
            width=1240,
            height=800,
            # Small enough for a laptop, wide enough that both panes still get
            # their contents in rather than clipping them.
            min_width=960,
            min_height=720,
        )

        # Theme
        apply_default_theme()

        # GUI-settings
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=self.PREVIEW_TEXTURE,
                height=self.PREVIEW_TEXTURE,
                default_value=self.frame_to_data(self.frame_re),
                tag="imwin_tag0",
            )

        with dpg.window(
            label="Frame Capture",
            tag="main_window",
            no_title_bar=True,
            no_resize=True,
            no_move=True,
            no_collapse=True,
        ):
            self._build_source_row()
            with dpg.group(horizontal=True):
                self._build_preview_pane()
                self._build_control_pane()
            self._build_footer()

        dpg.set_primary_window("main_window", True)
        dpg.set_viewport_resize_callback(lambda: self._relayout())

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
        dpg.setup_dearpygui()
        dpg.show_viewport()
        self._relayout()

    def _build_source_row(self):
        dpg.add_text(default_value="Video Source")
        dpg.add_separator()
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

    def _build_preview_pane(self):
        with dpg.child_window(tag="left_pane", border=False, no_scrollbar=True):
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

    def _build_control_pane(self):
        with dpg.child_window(tag="right_pane", width=-1, border=False):
            # ---- manual grab -------------------------------------------
            dpg.add_text(default_value="Save Frame")
            dpg.add_separator()
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
                dpg.add_button(
                    label="Reset Count", callback=lambda: self.count_reset_bt()
                )

            # ---- automatic extraction ----------------------------------
            dpg.add_spacer(height=10)
            dpg.add_text(default_value="Automatic Extraction")
            dpg.add_separator()
            dpg.add_text(
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

    def _relayout(self):
        """Fit the preview and the two panes to the viewport.

        Called once at start-up and on every viewport resize, so it must stay
        cheap: DearPyGui runs the resize callback for each step of a drag, and
        anything that decodes a frame or reallocates a texture here would be
        paid dozens of times per second while the mouse moves.
        """
        vw = max(1, dpg.get_viewport_client_width())
        vh = max(1, dpg.get_viewport_client_height())

        body_h = max(self.MIN_PREVIEW + self.LEFT_CHROME, vh - self.HEADER_H - self.FOOTER_H)
        side = min(body_h - self.LEFT_CHROME, vw - self.SIDE_PANE_WIDTH - 56)
        side = max(self.MIN_PREVIEW, int(side))

        dpg.configure_item(
            "left_pane", width=max(side + 26, self.LEFT_MIN_WIDTH), height=body_h
        )
        dpg.configure_item("right_pane", width=-1, height=body_h)
        dpg.configure_item("preview_image", width=side, height=side)
        dpg.configure_item("frame_bar", width=side)

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def run(self):
        self.gui_configure()
        try:
            while dpg.is_dearpygui_running():
                self.plot_callback()
                dpg.render_dearpygui_frame()
                if self.m_dict["quit"]:
                    break
        finally:
            self._shutdown()

    def _shutdown(self):
        """Tear the window down from the render loop, never from a callback.

        Destroying the context inside a button callback frees everything the
        half-finished frame is still drawing from, which is its own way of
        hanging on exit.
        """
        self._extract_stop = True
        thread = self._extract_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if isinstance(self.vid, cv2.VideoCapture):
            self.vid.release()
        dpg.destroy_context()

    def plot_callback(self) -> None:
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
            print("Failed to open video: " + file_path)
            self._set_status(f"Could not open the video: {file_path}")
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
            self._set_status("Select a video first.")
            return
        self.grab_name = self._save_name()
        if not self.grab_name or not self.grab_dir:
            print("Not selected file dir or name")
            self._set_status("Select a save directory and a frame name first.")
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
            self._set_status(f"Saved {os.path.basename(self.grab_path)}")
        else:
            print("Failed to read frame for grab")
            self._set_status("Could not save that frame.")

    def count_reset_bt(self):
        self.grab_count = 0
        self._update_grab_count()

    def _update_grab_count(self):
        dpg.set_value("count_frames", f"{self.grab_count} frames grabbed")

    def _set_status(self, text):
        dpg.set_value("extract_status", text)

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
        print("quit_pushed")
        # The render loop notices the flag and calls _shutdown; a callback must
        # not destroy the context it is currently being drawn inside of.
        self.m_dict["quit"] = True

    def __del__(self):
        if hasattr(self, "m_dict"):
            self.m_dict["quit"] = True
        print("=== GUI window quit ===")


def main():
    d = {}
    d["quit"] = False
    grabWin = grab_gui(d)
    grabWin.run()


if __name__ == "__main__":
    main()
