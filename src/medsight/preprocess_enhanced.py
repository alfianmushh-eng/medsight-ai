Enhanced preprocessing: adaptive equalization, denoising.
from __future__ import annotations
import numpy as np
from scipy import ndimage, signal

def adaptive_equalization(img: np.ndarray, *, tile_size: int = 8, clip_limit: float = 3.0) -> np.ndarray:
    img = img.astype(np.float32)
    tiles_y = img.shape[0] // tile_size
    tiles_x = img.shape[1] // tile_size
    result = img.copy()
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y1, y2 = ty * tile_size, (ty + 1) * tile_size
            x1, x2 = tx * tile_size, (tx + 1) * tile_size
            tile = img[y1:y2, x1:x2]
            hist, bins = np.histogram(tile, bins=256, range=(tile.min(), tile.max()))
            clip = int(clip_limit * tile.size / 256)
            hist = np.clip(hist, 0, clip)
            cdf = hist.cumsum().astype(np.float32)
            cdf = cdf / (cdf[-1] + 1e-8)
            indices = np.clip(np.digitize(tile, bins[1:-1]), 0, 255)
            result[y1:y2, x1:x2] = cdf[indices] * (bins[-1] - bins[0]) + bins[0]
    return result

def gaussian_denoise(img: np.ndarray, *, sigma: float = 1.0) -> np.ndarray:
    return ndimage.gaussian_filter(img, sigma)

def median_denoise(img: np.ndarray, *, size: int = 3) -> np.ndarray:
    return ndimage.median_filter(img, size)
