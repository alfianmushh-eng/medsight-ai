Base trainer for classification and segmentation.
from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

class Trainer:
    def __init__(self, model: nn.Module, device: str = "cuda", checkpoint_dir: str = "checkpoints"):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history: dict[str, list] = {"train_loss": [], "val_loss": []}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, *, callbacks: list | None = None):
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            val_loss = self._evaluate(val_loader, criterion)
            self.history["train_loss"].append(train_loss / len(train_loader))
            self.history["val_loss"].append(val_loss)
            if callbacks:
                for cb in callbacks:
                    cb(self, epoch)
        return self.history

    def _evaluate(self, loader, criterion):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                total += criterion(self.model(x), y).item()
        return total / len(loader)

    def save(self, name: str = "last.pt"):
        torch.save({"model_state": self.model.state_dict(), "history": self.history}, self.checkpoint_dir / name)
