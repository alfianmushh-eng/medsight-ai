Inference helpers for classification and segmentation.
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

def load_model(model_class, checkpoint: str | Path, *, num_classes: int = 2, device: str = "cpu", **kwargs):
    model = model_class(num_classes=num_classes, **kwargs)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.to(device).eval()
    return model

def predict(model, image: np.ndarray, device: str = "cpu") -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(device)
        out = model(x)
        return F.softmax(out, dim=1).squeeze().cpu().numpy() if out.size(1) > 1 else torch.sigmoid(out).squeeze().cpu().numpy()
