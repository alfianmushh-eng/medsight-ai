DICOM series loader.
from __future__ import annotations
from pathlib import Path
import numpy as np
import pydicom

def load_series(directory: str | Path) -> np.ndarray:
    d = Path(directory)
    slices = sorted(pydicom.dcmread(str(f)) for f in d.iterdir() if f.suffix.lower() in (".dcm", ""))
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]) if hasattr(s, "ImagePositionPatient") else 0.0)
    return np.stack([s.pixel_array.astype(np.float32) for s in slices])

def load_dicom(path: str | Path) -> tuple[np.ndarray, dict]:
    ds = pydicom.dcmread(str(path))
    return ds.pixel_array.astype(np.float32), {k: str(v) for k, v in ds.items() if k != "PixelData"}
