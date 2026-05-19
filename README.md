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
| `medsight.preprocess_enhanced` | Adaptive equalization, Gaussian/median denoising        |
| `medsight.datasets`   | `ImageFolderDataset`, DICOM series, NIfTI volume, CSV loader  |
| `medsight.augment`    | Numpy-only flips, brightness jitter, rotation, Compose        |
| `medsight.augment_advanced` | Random rotation, crop, cutout                            |
| `medsight.metrics`    | Binary classification, Dice, IoU                              |
| `medsight.eval_`      | Confusion matrix, per-class metrics, ROC AUC                  |
| `medsight.losses`     | DiceLoss, FocalLoss, DiceBCELoss                              |
| `medsight.models`     | SimpleCNN, ResNet wrapper, UNet                               |
| `medsight.training`   | Trainer, LR schedulers, EarlyStopping, ModelCheckpoint        |
| `medsight.config`     | Frozen-dataclass experiment configs from YAML                 |
| `medsight.visualize`  | Overlay mask, image montage                                   |
| `medsight.inference`  | Model loading and prediction                                  |
| `medsight.split`      | Dataset splitting (train/val/test)                            |
| `medsight.crossval`   | K-fold cross-validation with stratification                   |
| `medsight.cli`        | `medsight inspect`, `validate-config`, `dicom-info`           |

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
