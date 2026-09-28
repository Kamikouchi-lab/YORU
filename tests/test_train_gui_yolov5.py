"""The training GUI with YOLOv5: the selection, OBB projects, and the epoch count.

Uses the recording-dpg fixture of test_train_stop_gui.py, so no window opens.
"""

from tests.test_train_stop_gui import _FakeProc, gui  # noqa: F401  (fixture)


def _select_yolo(gui, version, size="s", obb=False):
    gui.m_dict.update(
        model_family="YOLO", yolo_version=version, yolo_size=size, obb=obb,
    )


def test_yolov5_builds_a_yolov5_weight(gui):
    _select_yolo(gui, "YOLOv5", "m")
    assert gui._build_weight() == "yolov5m.pt"


def test_choosing_yolov5_in_an_obb_project_is_refused(gui):
    """YOLOv5 has no rotated-box head; say so now, not at training time."""
    _select_yolo(gui, "YOLO11", obb=True)
    gui.dpg.values["yolo_version_combo"] = "YOLOv5"
    gui.select_version()
    assert gui.m_dict["yolo_version"] == "YOLO11"
    assert gui.dpg.values["yolo_version_combo"] == "YOLO11"
    assert gui.m_dict["weight"] == "yolo11s-obb.pt"


def test_ticking_obb_on_the_yolov5_default_moves_to_yolo11(gui):
    """New projects start on YOLOv5; the OBB box must not leave them there."""
    from yoru.libs.init_train import init_train

    init_train(m_dict=gui.m_dict)
    assert gui.m_dict["weight"] == "yolov5s.pt"
    gui.dpg.values["obb_chk"] = True
    gui.select_obb()
    assert gui.m_dict["yolo_version"] == "YOLO11"
    assert gui.m_dict["weight"] == "yolo11s-obb.pt"


def test_a_missing_version_never_builds_a_yolov5_obb_weight(gui):
    gui.m_dict.update(model_family="YOLO", yolo_size="s", obb=True)
    gui.m_dict.pop("yolo_version", None)
    assert gui._build_weight() == "yolo11s-obb.pt"


def test_an_obb_project_moves_off_yolov5(gui):
    _select_yolo(gui, "YOLOv5", obb=True)
    gui._sync_obb_ui()
    assert gui.m_dict["yolo_version"] == "YOLO11"
    assert gui.m_dict["weight"] == "yolo11s-obb.pt"


def test_a_v1_yolov5_project_is_restored_as_yolov5(gui):
    """v2.0 beta switched these to YOLO11; they must come back as they were."""
    gui.m_dict.update(yolo_size_list=["n", "s", "m", "l", "x"], obb=False)
    gui._restore_model_ui("yolov5l.pt")
    assert gui.m_dict["yolo_version"] == "YOLOv5"
    assert gui.m_dict["yolo_size"] == "l"
    assert gui.m_dict["weight"] == "yolov5l.pt"


def test_yolov5_epochs_are_shown_counted_from_one(gui):
    """yolov5 prints the first of 300 epochs as 0/299."""
    gui.m_dict["train_stop_mode"] = ""
    gui._monitor_training(
        _FakeProc(["        0/299      3.65G     0.1063  640: 100%"]), 300, epoch_base=0,
    )
    assert gui.m_dict["train_total_epoch"] == 300
    assert gui.m_dict["train_epoch"] == 300  # ran to its end: the bar is filled


def test_a_stopped_yolov5_run_reports_the_epoch_it_ended_after(gui):
    gui.m_dict["train_stop_mode"] = "graceful"
    gui._monitor_training(
        _FakeProc(["       11/299      3.65G     0.1063  640: 100%"]), 300, epoch_base=0,
    )
    assert gui.m_dict["train_epoch"] == 12
    assert gui.m_dict["train_total_epoch"] == 300
