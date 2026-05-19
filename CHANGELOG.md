# Changelog

## [0.2.0] - 2026-05-28

### Added
- DICOM series and NIfTI volume loaders
- CSV-annotated image dataset
- Adaptive equalization and denoising filters
- Random rotation, crop, and cutout augmentations
- SimpleCNN, ResNet wrapper, and UNet models
- Trainer class with fit/eval/save
- LR schedulers: plateau, cosine, step
- EarlyStopping and ModelCheckpoint callbacks
- DiceLoss, FocalLoss, DiceBCELoss
- Confusion matrix, per-class metrics, ROC AUC
- Overlay mask and montage visualization
- Inference helpers (model loading, prediction)
- Dataset splitting and k-fold cross-validation
- Additional configs: retinal and mammography baselines
- dicom-info CLI command
All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-05-24

### Added
- `preprocess` module: grayscale conversion, min-max normalization, light histogram equalization
- `datasets.ImageFolderDataset` for class-per-folder image layouts
- `augment` module: horizontal/vertical flip, brightness jitter, 90-degree rotation, Compose
- `metrics` module: binary classification metrics, Dice score, IoU score
- `config` loader: YAML-backed frozen-dataclass experiment configs
- Baseline `configs/chestxray_baseline.yaml`
- `medsight` CLI with `inspect` and `validate-config` subcommands
- `examples/smoke_train.py` synthetic end-to-end smoke test
