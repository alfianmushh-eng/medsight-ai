LR scheduler wrappers.
from __future__ import annotations
import torch.optim as optim

def get_scheduler(optimizer, *, name: str = "plateau", **kwargs):
    schedulers = {
        "plateau": optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=kwargs.get("patience", 5), factor=kwargs.get("factor", 0.5)),
        "cosine": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get("t_max", 50)),
        "step": optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get("step_size", 10), gamma=kwargs.get("gamma", 0.1)),
    }
    if name not in schedulers:
        raise ValueError(f"unknown scheduler: {name} (available: {list(schedulers)})")
    return schedulers[name]
