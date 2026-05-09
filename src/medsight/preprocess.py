"""Image preprocessing utilities for medical imaging."""
from __future__ import annotations

import numpy as np
from PIL import Image


def to_grayscale(arr: np.ndarray) -> np.ndarray:
    """Convert an HWC or HW array to single-channel grayscale (HW)."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        if arr.shape[-1] == 1:
            return arr[..., 0]
        # Rec. 601 luma coefficients
        rgb = arr[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        return gray.astype(arr.dtype)
    raise ValueError(f"unexpected array shape: {arr.shape}")


def min_max_normalize(arr: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Scale array to the [0, 1] range using per-image min/max."""
    a = arr.astype(np.float32)
    lo = float(a.min())
    hi = float(a.max())
    return (a - lo) / (hi - lo + eps)


def clahe_lite(arr: np.ndarray, *, bins: int = 256) -> np.ndarray:
    """Simple histogram equalization (dependency-free CLAHE substitute)."""
    a = arr.astype(np.float32)
    flat = a.flatten()
    hist, edges = np.histogram(flat, bins=bins, range=(flat.min(), flat.max()))
    cdf = hist.cumsum().astype(np.float32)
    cdf = cdf / (cdf[-1] + 1e-8)
    indices = np.clip(np.digitize(a, edges[1:-1]), 0, bins - 1)
    return cdf[indices]


def load_image(path: str) -> np.ndarray:
    """Load a PNG/JPEG image as a numpy array."""
    return np.asarray(Image.open(path))
