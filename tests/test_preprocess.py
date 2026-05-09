import numpy as np

from medsight import preprocess


def test_to_grayscale_passthrough() -> None:
    a = np.ones((4, 4), dtype=np.uint8) * 200
    out = preprocess.to_grayscale(a)
    assert out.shape == (4, 4)


def test_to_grayscale_rgb() -> None:
    a = np.zeros((2, 2, 3), dtype=np.uint8)
    a[..., 0] = 255  # pure red
    out = preprocess.to_grayscale(a)
    assert out.shape == (2, 2)
    # Red maps to ~76 (0.299 * 255)
    assert 70 <= int(out.mean()) <= 80


def test_min_max_normalize() -> None:
    a = np.array([[0.0, 50.0], [100.0, 200.0]])
    out = preprocess.min_max_normalize(a)
    assert out.min() == 0.0
    assert abs(out.max() - 1.0) < 1e-6
