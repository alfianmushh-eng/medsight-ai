import numpy as np

from medsight import metrics


def test_perfect_classification() -> None:
    y = np.array([0, 1, 1, 0, 1])
    m = metrics.binary_classification(y, y)
    assert m.accuracy == 1.0
    assert m.f1 > 0.999


def test_dice_identical() -> None:
    a = np.array([[1, 0], [1, 1]], dtype=bool)
    assert abs(metrics.dice_score(a, a) - 1.0) < 1e-6


def test_iou_disjoint() -> None:
    a = np.array([1, 0, 0], dtype=bool)
    b = np.array([0, 1, 0], dtype=bool)
    assert metrics.iou_score(a, b) < 1e-6
