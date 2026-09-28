"""Atomic detection snapshots shared by capture, inference and triggers."""

import time


def clear_detection(state):
    state["detection_snapshot"] = None
    state["yolo_results"] = []
    state["yolo_class_names"] = []


def fresh_results(state, now=None):
    """Fail closed unless a result belongs to a current, recent captured frame."""
    if (state.get("quit", False) or not state.get("yolo_detection", False)
            or not state.get("yolo_process_state", False)
            or not state.get("capture_running", False)):
        return []
    snapshot = state.get("detection_snapshot")
    if snapshot is None:
        return []
    _, captured_at, _, generation, rows = snapshot
    if generation != state.get("detection_generation", 0):
        return []
    age = (time.perf_counter() if now is None else now) - captured_at
    if not 0 <= age <= float(state.get("result_max_age", 1.0)):
        return []
    return rows
