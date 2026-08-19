"""The training GUI must be able to predict an out-of-memory run beforehand.

A batch size that does not fit only announces itself minutes into training,
as a CUDA OOM traceback, so the estimate that warns about it has to hold for
every model the GUI can select -- not just the ones anyone happened to try.
"""

import pytest

from yoru.libs.init_train import MODEL_FAMILY_CONFIG
from yoru.libs.vram_estimate import (
    CUDA_CONTEXT_GB,
    DatasetStats,
    GPUInfo,
    check_training_vram,
    estimate_training_vram,
    largest_batch_that_fits,
    profile_for,
    read_dataset_stats,
)


def _selectable_weights():
    """Every weight file name train_GUI._build_weight can produce."""
    names = []
    for version, prefix in (("YOLOv8", "yolov8"), ("YOLO11", "yolo11")):
        assert version in MODEL_FAMILY_CONFIG["YOLO"]["versions"]
        names += [f"{prefix}{s}.pt" for s in MODEL_FAMILY_CONFIG["YOLO"]["sizes"]]
    names += [f"rtdetr-{s}.pt" for s in MODEL_FAMILY_CONFIG["RT-DETR"]["sizes"]]
    names += [
        "fasterrcnn_resnet50_best.pt",
        "maskrcnn_resnet50_best.pt",
        "ssd_vgg16_best.pt",
    ]
    return names


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("weight", _selectable_weights())
def test_every_selectable_model_has_a_profile(weight):
    """Adding a model to the GUI without a memory profile must not go unnoticed."""
    assert profile_for(weight) is not None, f"no VRAM profile for {weight}"
    est = estimate_training_vram(weight, 640, 8)
    assert est is not None and est.total_gb > 0


def test_unknown_weight_yields_no_estimate():
    """A checkpoint outside the table must say so rather than guess."""
    assert estimate_training_vram("some_custom_model.pt", 640, 16) is None
    verdict = check_training_vram("some_custom_model.pt", 640, 16)
    assert verdict.level == "unknown"
    assert not verdict.is_warning


def test_size_ordering_within_a_family():
    """A bigger model must never estimate cheaper than a smaller one."""
    sizes = ["n", "s", "m", "l", "x"]
    totals = [estimate_training_vram(f"yolo11{s}.pt", 640, 16).total_gb for s in sizes]
    assert totals == sorted(totals), totals


# ---------------------------------------------------------------------------
# Scaling behaviour
# ---------------------------------------------------------------------------


def test_activations_are_linear_in_batch():
    one = estimate_training_vram("yolo11s.pt", 640, 1)
    ten = estimate_training_vram("yolo11s.pt", 640, 10)
    assert ten.activation_gb == pytest.approx(one.activation_gb * 10)
    # The model term is a fixed cost and must not be multiplied by the batch.
    assert ten.model_gb == pytest.approx(one.model_gb)


def test_activations_are_quadratic_in_image_size():
    small = estimate_training_vram("yolo11s.pt", 320, 8)
    large = estimate_training_vram("yolo11s.pt", 640, 8)
    assert large.activation_gb == pytest.approx(small.activation_gb * 4)


def test_torchvision_ignores_the_image_size_setting():
    """train_torchvision.py never resizes; the transform does, to its own size."""
    a = estimate_training_vram("fasterrcnn_resnet50_best.pt", 320, 4)
    b = estimate_training_vram("fasterrcnn_resnet50_best.pt", 1280, 4)
    assert a.total_gb == pytest.approx(b.total_gb)
    assert any("resize internally" in n for n in a.notes)


def test_crowded_frames_cost_assigner_memory():
    """Many labels per image blow up YOLO's (batch, labels, anchors) tensors."""
    sparse = estimate_training_vram(
        "yolo11s.pt", 640, 16, DatasetStats(max_instances=3)
    )
    crowded = estimate_training_vram(
        "yolo11s.pt", 640, 16, DatasetStats(max_instances=300)
    )
    assert crowded.assigner_gb > sparse.assigner_gb
    assert crowded.total_gb > sparse.total_gb + 1.0


def test_rtdetr_has_no_assigner_term():
    """RT-DETR matches a fixed query set, so label count does not size a grid."""
    est = estimate_training_vram("rtdetr-l.pt", 640, 8, DatasetStats(max_instances=300))
    assert est.assigner_gb == 0.0


# ---------------------------------------------------------------------------
# Calibration anchor
# ---------------------------------------------------------------------------


def test_rtdetr_l_matches_a_measured_run():
    """rtdetr-l, 640 px, batch 5 was measured at ~5.5 GB by ultralytics.

    ``GPU_mem`` is torch.cuda.memory_reserved, which excludes the CUDA
    context, so the context term is taken back off before comparing.  The
    estimate is documented as +/-30%; this pins that claim to a real number
    so a careless edit to the profile table cannot silently break it.
    """
    est = estimate_training_vram("rtdetr-l.pt", 640, 5)
    reserved = est.total_gb - CUDA_CONTEXT_GB
    assert reserved == pytest.approx(5.5, rel=0.3), reserved


# ---------------------------------------------------------------------------
# Batch suggestion
# ---------------------------------------------------------------------------


def test_suggested_batch_actually_fits():
    budget = 8.0
    fit = largest_batch_that_fits("yolo11m.pt", 640, budget)
    assert fit >= 1
    assert estimate_training_vram("yolo11m.pt", 640, fit).total_gb <= budget
    assert estimate_training_vram("yolo11m.pt", 640, fit + 1).total_gb > budget


def test_no_batch_fits_a_tiny_budget():
    assert largest_batch_that_fits("rtdetr-x.pt", 640, 1.0) == 0


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def _gpu(total_gb, used_gb=0.0):
    return GPUInfo("Test GPU", total_gb, used_gb, "test")


def test_comfortable_run_is_ok():
    verdict = check_training_vram("yolo11n.pt", 640, 8, gpu=_gpu(24.0))
    assert verdict.level == "ok"
    assert not verdict.is_warning


def test_oversized_run_warns_and_suggests_a_smaller_batch():
    verdict = check_training_vram("rtdetr-x.pt", 640, 64, gpu=_gpu(12.0))
    assert verdict.level == "over"
    assert verdict.is_warning
    assert verdict.suggested_batch and verdict.suggested_batch < 64
    assert str(verdict.suggested_batch) in verdict.detail


def test_memory_taken_by_other_processes_counts_against_the_budget():
    """A second training run on the same card is the usual way this bites."""
    free_card = check_training_vram("yolo11m.pt", 640, 16, gpu=_gpu(16.0))
    busy_card = check_training_vram("yolo11m.pt", 640, 16, gpu=_gpu(16.0, used_gb=10.0))
    assert free_card.level == "ok"
    assert busy_card.level == "over"
    assert "already in use" in busy_card.detail


def test_no_gpu_is_reported_as_unknown_not_as_a_failure():
    verdict = check_training_vram("yolo11s.pt", 640, 16, gpu=None)
    # gpu=None asks the module to probe; on a CPU-only machine that yields
    # "unknown", on a CUDA machine a real verdict.  Neither may raise.
    assert verdict.level in ("ok", "tight", "over", "unknown")


# ---------------------------------------------------------------------------
# Dataset inspection
# ---------------------------------------------------------------------------


def _make_project(tmp_path, n_train=5, n_val=2, instances=(1, 4, 2)):
    import yaml as _yaml

    for split, count in (("train", n_train), ("val", n_val)):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
        for i in range(count):
            (tmp_path / split / "images" / f"{i}.png").write_bytes(b"")
    for i, n in enumerate(instances):
        lines = "\n".join("0 0.5 0.5 0.1 0.1" for _ in range(n))
        (tmp_path / "train" / "labels" / f"{i}.txt").write_text(lines)

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        _yaml.dump(
            {
                "path": str(tmp_path),
                "train": str(tmp_path / "train") + "/",
                "val": str(tmp_path / "val") + "/",
                "nc": 2,
                "names": ["a", "b"],
            }
        )
    )
    return cfg


def test_dataset_stats_are_read_from_a_yoru_project(tmp_path):
    cfg = _make_project(tmp_path)
    stats = read_dataset_stats(cfg)
    assert stats.n_train == 5
    assert stats.n_val == 2
    assert stats.n_classes == 2
    assert stats.max_instances == 4
    assert stats.sampled_labels == 3


def test_dataset_stats_accept_an_images_directory_directly(tmp_path):
    """A hand-written YAML may point at train/images rather than train/."""
    import yaml as _yaml

    _make_project(tmp_path)
    cfg = tmp_path / "direct.yaml"
    cfg.write_text(
        _yaml.dump(
            {
                "path": str(tmp_path),
                "train": str(tmp_path / "train" / "images"),
                "val": str(tmp_path / "val" / "images"),
                "nc": 2,
            }
        )
    )
    stats = read_dataset_stats(cfg)
    assert stats.n_train == 5
    assert stats.max_instances == 4


def test_classes_txt_is_not_counted_as_a_label(tmp_path):
    cfg = _make_project(tmp_path)
    (tmp_path / "train" / "labels" / "classes.txt").write_text("a\nb\nc\nd\ne\nf\ng\n")
    assert read_dataset_stats(cfg).max_instances == 4


@pytest.mark.parametrize("bad", ["", None, "no/such/file.yaml"])
def test_missing_dataset_yaml_is_not_an_error(bad):
    """The readout must survive being asked before a project is loaded."""
    stats = read_dataset_stats(bad)
    assert stats == DatasetStats()


def test_malformed_dataset_yaml_is_not_an_error(tmp_path):
    cfg = tmp_path / "broken.yaml"
    cfg.write_text("[[[not: valid: yaml")
    assert read_dataset_stats(cfg) == DatasetStats()
