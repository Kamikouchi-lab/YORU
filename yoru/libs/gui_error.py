"""Shared exception-handling helpers for the GUIs.

By having each ``*_GUI`` class inherit :class:`GuiErrorMixin`, the following
can be handled uniformly:

- On failure, write a traceback to standard error
  (``app.py``'s ``_run_gui_subprocess`` forwards it to the home screen), and
- Display the error details in a DearPyGui modal window.

This generalizes the pattern first implemented in ``analysis_GUI.py``.
"""

import sys
import traceback

import dearpygui.dearpygui as dpg


class GuiErrorMixin:
    """Mixin providing common error-reporting features for DearPyGui-based GUIs."""

    # Use a fixed tag so that multiple error popups are not opened at once
    _error_popup_tag = "error_popup"

    def _report_error(self, context, exc):
        """Log the exception to standard error and show it in an on-screen popup.

        User notification of caught errors is handled by the modal popup this
        method displays. The traceback written to standard error is for logging;
        note that ``app.py``'s ``_run_gui_subprocess`` only forwards it to the
        home screen when the GUI process exits with a non-zero code (i.e. it
        crashed without catching the exception).
        """
        detail = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {context}: {detail}", file=sys.stderr, flush=True)
        traceback.print_exc()
        self._show_error_popup(context, detail)

    def _show_error_popup(self, context, detail):
        """Display the error details in a modal window."""
        try:
            tag = self._error_popup_tag
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
            with dpg.window(
                label="Error",
                modal=True,
                tag=tag,
                no_resize=True,
                width=480,
                pos=(210, 300),
            ):
                dpg.add_text(context, color=(255, 160, 120))
                dpg.add_separator()
                dpg.add_text(detail, wrap=460, color=(255, 120, 120))
                dpg.add_spacer(height=8)
                dpg.add_button(
                    label="Close",
                    width=80,
                    callback=lambda: dpg.delete_item(tag),
                )
        except Exception:
            # Even if showing the popup fails, the message remains in standard error
            pass

    def _safe_enable(self, tag):
        """Enable the item only if it exists (ignore failures)."""
        try:
            if dpg.does_item_exist(tag):
                dpg.enable_item(tag)
        except Exception:
            pass
