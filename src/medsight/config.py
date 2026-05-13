"""Experiment configuration loader.

Configs are YAML files describing dataset paths, model hyperparameters and
training schedules. Loading happens once at the start of a run; from then on
the dataclass is read-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_root: str
    val_root: str
    image_size: int = 224
    batch_size: int = 32


@dataclass(frozen=True)
class ModelConfig:
    name: str = "resnet18"
    num_classes: int = 2
    pretrained: bool = True


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adam"
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    extras: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML from disk."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"expected YAML mapping, got {type(raw).__name__}")
    return ExperimentConfig(
        name=raw["name"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        extras=raw.get("extras", {}),
    )
