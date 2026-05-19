Segmentation post-processing utilities.
from __future__ import annotations
import numpy as np
from scipy import ndimage

def remove_small_objects(mask: np.ndarray, *, min_size: int = 100) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.where(sizes >= min_size)[0] + 1
    return np.isin(labeled, keep)

def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(mask)

def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n == 0: return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    return labeled == np.argmax(sizes) + 1
