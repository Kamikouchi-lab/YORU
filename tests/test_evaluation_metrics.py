import math
from types import SimpleNamespace

import numpy as np
import pytest

from yoru.libs import evaluation_calculation as module
from yoru.libs.obb import obb_corners


@pytest.fixture
def evaluator(monkeypatch):
    obj = module.Evaluator({"pr_curve_dir": "unused"})
    monkeypatch.setattr(obj, "plot_precision_recall_curve", lambda *args: None)
    return obj


GT = [0, .5, .5, .2, .2]
TP = [0, .9, .5, .5, .2, .2]
FP = [0, .1, .1, .1, .1, .1]


@pytest.mark.parametrize("predictions", [[TP, FP], [FP, TP]])
def test_confidence_ranking_and_precision_envelope(evaluator, predictions):
    metrics, _, _, _ = evaluator.evaluate([[GT]], [predictions], {0: "fly"}, "test")
    assert metrics["mAP@.50"] == pytest.approx(1)
    assert metrics["mAP@[.50:.05:.95]"] == pytest.approx(1)


@pytest.mark.parametrize("reverse", [False, True])
def test_confidence_is_sorted_across_images(evaluator, reverse):
    truth, predictions = [[], [GT]], [[FP], [TP]]
    if reverse:
        truth.reverse()
        predictions.reverse()
    metrics, _, _, _ = evaluator.evaluate(truth, predictions, {0: "fly"}, "test")
    assert metrics["mAP@.50"] == pytest.approx(1)


def test_high_confidence_false_positive_reduces_ap(evaluator):
    metrics, _, _, _ = evaluator.evaluate([[GT]], [[TP, [0, .99, .1, .1, .1, .1]]], {0: "fly"}, "test")
    assert metrics["mAP@.50"] == pytest.approx(.5)


def test_no_predictions_and_unannotated_classes(evaluator):
    metrics, _, _, _ = evaluator.evaluate([[GT]], [[]], {0: "fly", 1: "absent"}, "test")
    assert metrics["mAP@.50"] == 0
    assert metrics["AP@.50_per_class"][1] is None
    empty, _, _, _ = evaluator.evaluate([], [], {}, "test")
    assert empty["mAP@.50"] is None


def test_matching_uses_an_available_ground_truth(evaluator):
    # Both predictions favour GT 0, but the second still overlaps GT 1 enough.
    truth = [[0, .5, .5, .4, .4], [0, .55, .5, .4, .4]]
    predictions = [[0, .9, .5, .5, .4, .4], [0, .8, .51, .5, .4, .4]]
    tp, fp, _ = evaluator.calculate_tp_fp(truth, predictions, .5)
    assert list(tp) == [1, 1]
    assert list(fp) == [0, 0]
    tp, fp, _ = evaluator.calculate_tp_fp([GT], [TP, TP], .5)
    assert list(tp) == [1, 0]
    assert list(fp) == [0, 1]


def corners(angle):
    return [v for point in obb_corners((.5, .5, .4, .1, angle)) for v in point]


def test_rotated_iou_does_not_compare_upright_envelopes(evaluator):
    a, b = corners(math.pi / 4), corners(-math.pi / 4)
    assert evaluator.convert_to_corners(a) == pytest.approx(evaluator.convert_to_corners(b))
    assert evaluator.label_iou(a, b) == pytest.approx(1 / 7, abs=1e-6)
    assert evaluator.label_iou(a, a) == pytest.approx(1, abs=1e-6)
    metrics, _, _, _ = evaluator.evaluate([[[0, *a]]], [[[0, .9, *b]]], {0: "fly"}, "test")
    assert metrics["mAP@.50"] == 0


def test_mixed_formats_and_degenerate_boxes(evaluator):
    square = [.4, .4, .6, .4, .6, .6, .4, .6]
    assert evaluator.label_iou(GT[1:], square) == pytest.approx(1, abs=1e-6)
    assert evaluator.label_iou([.5, .5, 0, .2], square) == 0
    assert evaluator.label_iou([.1, .1, .1, .1], square) == 0


def test_export_preserves_rotation_on_non_square_images(tmp_path, monkeypatch, evaluator):
    image = tmp_path / "sample.jpeg"
    module.cv2.imwrite(str(image), np.zeros((100, 200, 3), np.uint8))
    detection = dict(class_id=0, conf=.9, cx=100, cy=50, w=80, h=20, angle=.6)
    monkeypatch.setattr(module, "get_detector", lambda *a: SimpleNamespace(names={0: "fly"}, detect=lambda f: [detection]))
    module.EvaluationImageAnalyzer({"model_path": "fake", "data_dir": str(tmp_path)}).analyze_image()
    row = list(map(float, (tmp_path / "sample_yolo.txt").read_text().split()))
    expected = [v for x, y in obb_corners((100, 50, 80, 20, .6)) for v in (x/200, y/100)]
    assert len(row) == 10
    assert row[2:] == pytest.approx(expected)
    assert evaluator.label_iou(row[2:], expected) == pytest.approx(1, abs=1e-6)


def test_overall_precision_requires_localisation_and_counts_misses(tmp_path, evaluator):
    (tmp_path / "one.jpeg").touch()
    (tmp_path / "one.txt").write_text("0 .5 .5 .2 .2\n")
    (tmp_path / "one_yolo.txt").write_text("0 .9 .1 .1 .1 .1\n")
    assert evaluator.calculate_precision_recall(tmp_path) == (0, 0, 1, 1, 0)
    (tmp_path / "one_yolo.txt").write_text("")
    assert evaluator.calculate_precision_recall(tmp_path) == (0, 0, 1, 0, 0)


def test_ap_preserves_precision_recall_pairs(evaluator):
    assert evaluator.calculate_ap([.5, 1], [1, .5]) == pytest.approx(.75)
