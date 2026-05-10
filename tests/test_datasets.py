from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medsight.datasets import ImageFolderDataset


def _make_dummy(tmp: Path) -> Path:
    for cls, n in [("normal", 2), ("pneumonia", 1)]:
        d = tmp / cls
        d.mkdir(parents=True)
        for i in range(n):
            arr = np.full((8, 8), 100 + i * 30, dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{cls}_{i}.png")
    return tmp


def test_image_folder(tmp_path: Path) -> None:
    root = _make_dummy(tmp_path)
    ds = ImageFolderDataset(root)
    assert len(ds) == 3
    assert ds.classes == ["normal", "pneumonia"]
    assert ds.class_counts() == {"normal": 2, "pneumonia": 1}
    sample = next(iter(ds))
    assert sample.image.shape == (8, 8)


def test_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ImageFolderDataset(tmp_path / "nope")
