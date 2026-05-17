Training callbacks.
from __future__ import annotations
import torch
from typing import Any

class EarlyStopping:
    def __init__(self, *, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.counter = 0; self.best_loss = float("inf"); self.early_stop = False

    def __call__(self, trainer, epoch):
        current = trainer.history["val_loss"][-1]
        if current < self.best_loss - self.min_delta:
            self.best_loss = current; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class ModelCheckpoint:
    def __init__(self, filepath: str = "checkpoints/best.pt", monitor: str = "val_loss", mode: str = "min"):
        self.filepath = filepath; self.monitor = monitor; self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")

    def __call__(self, trainer, epoch):
        val = trainer.history.get(self.monitor, [float("inf")])[-1]
        improved = (self.mode == "min" and val < self.best) or (self.mode == "max" and val > self.best)
        if improved:
            self.best = val
            torch.save({"model_state": trainer.model.state_dict(), "epoch": epoch, "history": trainer.history}, self.filepath)
