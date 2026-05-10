"""Dataset abstractions over directories of medical images."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from medsight import preprocess


_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    image: np.ndarray
    label: str
    path: Path


class ImageFolderDataset:
    """A simple class-per-folder image dataset.

    Expected layout::

        root/
          normal/
            img001.png
            img002.png
          pneumonia/
            img003.png

    Each subdirectory of ``root`` is a class label; files matching common image
    extensions are yielded as :class:`Sample` instances.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if not self.classes:
            raise ValueError(f"no class subdirectories in {self.root}")
        self._index: list[tuple[Path, str]] = []
        for cls in self.classes:
            for p in (self.root / cls).iterdir():
                if p.suffix.lower() in _IMG_EXT:
                    self._index.append((p, cls))

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[Sample]:
        for path, label in self._index:
            img = preprocess.load_image(str(path))
            yield Sample(image=img, label=label, path=path)

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {c: 0 for c in self.classes}
        for _, lbl in self._index:
            counts[lbl] += 1
        return counts
