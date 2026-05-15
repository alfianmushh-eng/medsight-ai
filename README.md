# medsight-ai

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-oriented Python framework for medical image classification and
segmentation experiments. Built around small, composable pieces that you can
read in an afternoon and remix in a notebook.

> Research use only. Not a medical device. Do not use for clinical decisions.

## What is inside

| Module                | Purpose                                                       |
| --------------------- | ------------------------------------------------------------- |
| `medsight.preprocess` | Grayscale conversion, min-max normalize, histogram equalize   |
| `medsight.datasets`   | `ImageFolderDataset` over class-per-folder image directories  |
| `medsight.augment`    | Numpy-only flips, brightness jitter, rotation, `Compose`      |
| `medsight.metrics`    | Binary classification, Dice, IoU                              |
| `medsight.config`     | Frozen-dataclass experiment configs loaded from YAML          |
| `medsight.cli`        | `medsight inspect` and `medsight validate-config`             |

## Install

```bash
pip install -e .
# with torch / torchvision:
pip install -e ".[torch]"
# with DICOM and NIfTI support:
pip install -e ".[dicom,nifti]"
```

## Quickstart

```python
from medsight import datasets, preprocess, metrics

ds = datasets.ImageFolderDataset("data/processed/chestxray/train")
print(ds.class_counts())

for sample in ds:
    img = preprocess.min_max_normalize(preprocess.to_grayscale(sample.image))
    ...
```

```bash
medsight inspect data/processed/chestxray/train
medsight validate-config configs/chestxray_baseline.yaml
```

For a runnable end-to-end smoke test on synthetic data:

```bash
python examples/smoke_train.py
```

## Experiment configs

Configurations live under `configs/` as YAML files and are loaded as frozen
dataclasses. A baseline lives at [`configs/chestxray_baseline.yaml`](configs/chestxray_baseline.yaml).

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## Roadmap

- Torch training loop with mixed precision
- Optional torchvision backbones (ResNet, EfficientNet, ConvNeXt)
- Segmentation: U-Net trainer, Dice loss
- DICOM windowing and NIfTI volume slicing
- Inference CLI with checkpoint loading

## License

MIT — see [LICENSE](LICENSE).
