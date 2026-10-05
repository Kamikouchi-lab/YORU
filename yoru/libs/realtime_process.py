"""Supervise realtime workers and bound shutdown even if a driver hangs."""

import time

from yoru.libs.realtime_state import clear_detection


def run_workers(processes, state, shutdown_timeout=30.0):
    started = []
    forced = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        while not state.get("quit", False):
            if any(p.exitcode is not None for p in started):
                break
            time.sleep(0.05)
    finally:
        state["quit"] = True
        state["stream"] = False
        state["Trigger"] = False
        clear_detection(state)
        deadline = time.monotonic() + shutdown_timeout
        for process in started:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in started:
            if process.is_alive():
                forced.append(process.name)
                process.terminate()
        for process in started:
            process.join(timeout=5.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)
    failed = [p.name for p in started if p.exitcode != 0]
    if forced or failed:
        raise RuntimeError(
            f"Realtime workers failed: {', '.join(failed)}; "
            f"forced shutdown: {', '.join(forced) or 'none'}. "
            "Check ~/.yoru/logs/yoru.log; interrupted recordings may be incomplete."
        )
