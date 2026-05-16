CSV-annotated image dataset.
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

class CsvImageDataset:
    def __init__(self, csv_path: str | Path, root: str | Path, img_col: str = "filename", label_col: str = "label"):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.img_col = img_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, str]:
        row = self.df.iloc[idx]
        img = np.asarray(Image.open(self.root / row[self.img_col]))
        return img, str(row[self.label_col])
