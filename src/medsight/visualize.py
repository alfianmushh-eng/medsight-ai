Medical image visualization helpers.
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def overlay_mask(image: np.ndarray, mask: np.ndarray, *, alpha: float = 0.4, color: tuple = (1, 0, 0)) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    image_n = (image - image.min()) / (image.max() - image.min() + 1e-8)
    overlay = image_n.copy()
    if mask.ndim == 3:
        mask = mask.squeeze()
    for c in range(3):
        overlay[..., c] = np.clip(image_n[..., c] + (mask > 0) * (color[c] - image_n[..., c]) * alpha, 0, 1)
    return overlay

def montage(images: list[np.ndarray], *, cols: int = 4, figsize=(12, 12)) -> plt.Figure:
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    return fig
