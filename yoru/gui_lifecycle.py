"""Run and close DearPyGui windows on the thread that renders them."""

import threading

from yoru.libs.user_paths import log_exception


def start_gui_task(gui, dpg, worker, title, status_tag=None):
    """Keep lengthy prediction/evaluation out of DearPyGui's callback queue."""
    current = getattr(gui, "_gui_task", None)
    if gui.m_dict.get("quit", False) or current is not None:
        return False
    task = {"done": False, "error": None, "title": title, "status_tag": status_tag}
    gui._gui_task = task
    if status_tag:
        dpg.set_value(status_tag, "Running...")

    def run():
        try:
            worker()
        except Exception as exc:
            if not gui.m_dict.get("quit", False):
                log_exception(title, exc)
                task["error"] = exc
        finally:
            task["done"] = True

    task["thread"] = threading.Thread(target=run, daemon=True)
    task["thread"].start()
    return True


def _poll_task(gui, dpg):
    task = getattr(gui, "_gui_task", None)
    if task is None or not task["done"]:
        return
    gui._gui_task = None
    if task["status_tag"]:
        dpg.set_value(task["status_tag"], "Error" if task["error"] else "Complete!!")
    if task["error"]:
        gui._report_error(task["title"], task["error"])


def run_gui(gui, dpg, setup, frame=None, cleanup=None, *, reopen_home=True):
    """Callbacks set ``m_dict['quit']``; only this owner destroys the context.

    The same cleanup runs for Quit, the native close button, and exceptions.
    Resources and layouts are handled before the context is destroyed, and
    Home opens the launcher only after the old window has been torn down.
    """
    try:
        setup()
        while not gui.m_dict.get("quit", False) and dpg.is_dearpygui_running():
            _poll_task(gui, dpg)
            if frame is not None:
                frame()
            if gui.m_dict.get("quit", False):
                break
            dpg.render_dearpygui_frame()
    finally:
        gui.m_dict["quit"] = True
        actions = []
        task = getattr(gui, "_gui_task", None)
        if task is not None:
            actions.append(lambda: task["thread"].join(timeout=3.0))
        if cleanup is not None:
            actions.append(cleanup)
        video = getattr(gui, "vid", None)
        if callable(getattr(video, "release", None)):
            actions.append(video.release)
        session = getattr(gui, "session", None)
        if session is not None:
            actions.append(session.save)
        try:
            for action in actions:
                try:
                    action()
                except Exception as exc:
                    log_exception("GUI shutdown cleanup failed", exc)
        finally:
            dpg.destroy_context()
    if reopen_home and gui.m_dict.get("back_to_home", False):
        from yoru import app
        app.main()
