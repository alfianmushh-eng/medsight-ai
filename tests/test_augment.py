import numpy as np

from medsight import augment


def test_horizontal_flip_2d() -> None:
    a = np.array([[1, 2, 3], [4, 5, 6]])
    out = augment.horizontal_flip(a)
    assert out[0, 0] == 3 and out[0, -1] == 1


def test_compose_runs() -> None:
    pipe = augment.Compose([augment.horizontal_flip, augment.vertical_flip])
    a = np.arange(9).reshape(3, 3)
    out = pipe(a)
    assert out.shape == a.shape
