Advanced augmentations: rotation, crop, elastic.
from __future__ import annotations
import random
import numpy as np
from scipy import ndimage

def random_rotation(img: np.ndarray, *, max_angle: float = 15.0, rng: random.Random | None = None) -> np.ndarray:
    angle = (rng or random).uniform(-max_angle, max_angle)
    return ndimage.rotate(img, angle, reshape=False, order=1)

def random_crop(img: np.ndarray, *, size: int, rng: random.Random | None = None) -> np.ndarray:
    r = rng or random
    h, w = img.shape[:2]
    if h < size or w < size:
        return img
    y = r.randint(0, h - size)
    x = r.randint(0, w - size)
    return img[y:y + size, x:x + size]

def random_cutout(img: np.ndarray, *, mask_size: int = 16, fill: float = 0.0, rng: random.Random | None = None) -> np.ndarray:
    r = rng or random
    out = img.copy()
    h, w = img.shape[:2]
    y = r.randint(0, max(1, h - mask_size))
    x = r.randint(0, max(1, w - mask_size))
    if img.ndim == 3:
        out[y:y + mask_size, x:x + mask_size, :] = fill
    else:
        out[y:y + mask_size, x:x + mask_size] = fill
    return out
