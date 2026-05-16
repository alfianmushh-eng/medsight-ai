NIfTI volume loader.
from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib

def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    affine = img.affine
    header = dict(img.header)
    return data, affine, header

def load_nifti_pair(img_path: str | Path, seg_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(img_path)).get_fdata().astype(np.float32)
    seg = nib.load(str(seg_path)).get_fdata().astype(np.int16)
    return img, seg
