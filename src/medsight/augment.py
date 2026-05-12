"""Lightweight image augmentations (pure numpy)."""
from __future__ import annotations

import random
from typing import Callable

import numpy as np


def horizontal_flip(arr: np.ndarray) -> np.ndarray:
    return arr[..., ::-1, :] if arr.ndim == 3 else arr[:, ::-1]


def vertical_flip(arr: np.ndarray) -> np.ndarray:
    return arr[..., ::-1, :, :] if arr.ndim == 4 else arr[::-1]


def random_brightness(arr: np.ndarray, *, delta: float = 0.1, rng: random.Random | None = None) -> np.ndarray:
    r = rng or random
    shift = r.uniform(-delta, delta)
    out = arr.astype(np.float32) + shift * 255.0
    return np.clip(out, 0, 255).astype(arr.dtype)


def random_rotate_90(arr: np.ndarray, *, rng: random.Random | None = None) -> np.ndarray:
    r = rng or random
    k = r.choice([0, 1, 2, 3])
    return np.rot90(arr, k=k, axes=(-2, -1) if arr.ndim >= 2 else (0, 1))


class Compose:
    """Apply a sequence of augmentations in order."""

    def __init__(self, transforms: list[Callable[[np.ndarray], np.ndarray]]) -> None:
        self.transforms = transforms

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            arr = t(arr)
        return arr
