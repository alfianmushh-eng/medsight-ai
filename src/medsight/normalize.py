Intensity normalization for medical images.
from __future__ import annotations
import numpy as np

def zscore(img: np.ndarray) -> np.ndarray:
    return (img - img.mean()) / (img.std() + 1e-8)

def rescale(img: np.ndarray, *, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    a = img.astype(np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-8) * (hi - lo) + lo

def percentile_clip(img: np.ndarray, *, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    a = img.astype(np.float32)
    vmin, vmax = np.percentile(a, [lo, hi])
    return np.clip((a - vmin) / (vmax - vmin + 1e-8), 0, 1)
