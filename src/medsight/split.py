Data splitting utilities for medical datasets.
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import random

def split_image_folder(root: str | Path, *, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42) -> pd.DataFrame:
    root = Path(root)
    rows = []
    rng = random.Random(seed)
    for cls_dir in root.iterdir():
        if not cls_dir.is_dir(): continue
        images = list(cls_dir.iterdir())
        rng.shuffle(images)
        n = len(images)
        n_val = int(n * val_frac); n_test = int(n * test_frac)
        for i, img in enumerate(images):
            split = "train" if i < n - n_val - n_test else ("val" if i < n - n_test else "test")
            rows.append({"image": str(img), "label": cls_dir.name, "split": split})
    return pd.DataFrame(rows)

def split_csv(csv_path: str | Path, *, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * val_frac); n_test = int(len(df) * test_frac)
    splits = ["train"] * (len(df) - n_val - n_test) + ["val"] * n_val + ["test"] * n_test
    df["split"] = [splits[i] for i in idx]
    return df
