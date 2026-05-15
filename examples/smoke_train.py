"""End-to-end smoke test on a synthetic two-class dataset.

This example does not require torch — it walks through dataset inspection,
preprocessing, and metric computation using a dummy nearest-mean classifier.
It exists so a fresh checkout can confirm the toolkit works on machines
without GPU access.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from medsight import augment, datasets, metrics, preprocess


def _synthesize(root: Path, *, per_class: int = 30, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for cls, mean in [("normal", 90), ("abnormal", 160)]:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            arr = np.clip(rng.normal(mean, 20, size=(32, 32)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{cls}_{i:03d}.png")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "synth"
        _synthesize(root)

        ds = datasets.ImageFolderDataset(root)
        print(f"Loaded {len(ds)} images across {ds.classes}")
        print(f"Counts: {ds.class_counts()}")

        # Train a trivial nearest-mean baseline on mean pixel intensity
        means: dict[str, list[float]] = {c: [] for c in ds.classes}
        for sample in ds:
            img = preprocess.to_grayscale(sample.image)
            img = preprocess.min_max_normalize(img)
            means[sample.label].append(float(img.mean()))

        class_centroids = {c: float(np.mean(v)) for c, v in means.items()}
        print(f"Centroids: {class_centroids}")

        # Predict on the same set (smoke check — not a real evaluation protocol)
        y_true, y_pred = [], []
        positive = "abnormal"
        for sample in ds:
            img = preprocess.min_max_normalize(preprocess.to_grayscale(sample.image))
            score = float(img.mean())
            pred = positive if abs(score - class_centroids[positive]) < abs(score - class_centroids["normal"]) else "normal"
            y_true.append(1 if sample.label == positive else 0)
            y_pred.append(1 if pred == positive else 0)

        m = metrics.binary_classification(np.array(y_true), np.array(y_pred))
        print(f"Smoke metrics: acc={m.accuracy:.3f} f1={m.f1:.3f} recall={m.recall:.3f}")

        # Sanity check augmentation
        sample_img = preprocess.to_grayscale(next(iter(ds)).image)
        pipe = augment.Compose([augment.horizontal_flip, augment.random_brightness])
        out = pipe(sample_img)
        assert out.shape == sample_img.shape
        print("Augmentation pipeline OK")


if __name__ == "__main__":
    main()
