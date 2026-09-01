# Data Augmentation Module Design

**Date:** 2026-06-13
**Status:** Approved
**Scope:** MRI data augmentation pipeline for knee osteoarthritis (KOA) dataset

## Motivation

The project has only 404 training MRI images (`.nii.gz` format) and 103 test images for knee osteoarthritis analysis. Data augmentation is needed to:
1. Expand the training dataset for segmentation model training (U-Mamba, LKM-UNet, LightM-UNet)
2. Improve robustness of feature extraction (MEAS, LDSE, radiomics) via online augmentation

## Architecture

```
src/algorithm/data_augmentation/
├── __init__.py               # exports: augment_offline, augment_online
├── transforms.py             # MONAI transform pipeline definitions
├── offline_augmentor.py      # OfflineAugmentor class
├── online_augmentor.py       # OnlineAugmentor class
└── utils.py                  # nii.gz I/O, path helpers
```

### Layer 1: Core Module (`src/algorithm/data_augmentation/`)

Standalone Python module independent of MCP/Gradio. Two classes:

**OfflineAugmentor** — generates augmented `.nii.gz` files to disk:
- Input: `labels_train.csv` (with ImagePath, MaskPath, KLGrade columns)
- Output: augmented `imagesTr_aug/`, `labelsTr_aug/`, and updated `labels_train_aug.csv`
- Configurable augmentation multiplier (e.g., 5x turns 404 → 2020 images)
- Image and mask are transformed synchronously; KL grade label is preserved unchanged
- Generates one unique ID per augmented sample (e.g., `oaizib_001_aug0_0000.nii.gz`)

**OnlineAugmentor** — on-the-fly augmentation for feature extraction:
- Input: single image + mask (SimpleITK Image or numpy array)
- Output: list of augmented (image, mask) pairs for multi-pass feature extraction
- Integrates with existing MEAS/LDSE/radiomics extractors via composition
- Supports averaging features across multiple augmented views for robustness

### Layer 2: MCP Tool Interface

Two new `@mcp.tool()` functions in `src/server/auto_experiment_server.py`:

- `augment_dataset_offline(csv_path, output_dir, multiplier, ...)` — Agent-schedulable offline augmentation
- `augment_online_extract_features(csv_path, output_dir, n_augmentations, ...)` — online augmentation + feature extraction combined

### MONAI Transform Pipeline

```
Geometric transforms (image + mask, synchronized):
├─ RandFlip (p=0.5, axes=[0,1])       # left-right + anterior-posterior
├─ RandRotate (±10°, prob=0.5)         # simulate patient positioning variation
├─ RandZoom (0.9~1.1x, prob=0.5)       # simulate body size variation
├─ RandAffine (translate ±10 voxels, prob=0.5)
└─ RandElasticDeform (p=0.2, sigma=5)  # mild soft tissue deformation

Intensity transforms (image only):
├─ RandGaussianNoise (p=0.3, std=0.05) # MRI inherent noise
├─ RandBiasField (p=0.4, coeff=0.1)   # MRI-specific B0/B1 inhomogeneity
├─ RandAdjustContrast (p=0.3, gamma=(0.8, 1.2))
└─ RandGaussianSmooth (p=0.2, sigma=(0.5, 1.5))
```

## Constraints & Safety

1. **KneeSide handling**: before augmentation, normalize all images to the same side based on the `KneeSide` CSV field, to avoid generating anatomically impossible orientations
2. **KL Grade invariance**: geometric and intensity transforms do not alter KL grade (0-4). All augmented samples retain the original label
3. **Mask sync**: all geometric transforms use MONAI's `*d` (dictionary) variants to apply identical transforms to image and mask simultaneously
4. **No test leakage**: augmentation is applied ONLY to training set images. Test set (imagesTs) is never augmented
5. **Offline vs online**: offline mode generates files once; online mode applies random augmentation each call. No cross-contamination between modes

## New Dependencies

Added to `pyproject.toml`:
- `monai>=1.3.0` — medical imaging augmentation framework
- `torch>=2.0.0` — MONAI's underlying tensor library

Both are compatible with the project's existing `requires-python = ">=3.12"` constraint.

## Integration Points

- **Feature extraction**: `OnlineAugmentor` wraps existing `OAIZIB_MEAS_FeatureExtractor.extract_features()` and `OAIZIB_Light_LDSE_FeatureExtractor.extract_ldse_features()` to run multiple augmented passes
- **MCP server**: `auto_experiment_server.py` imports the core module and exposes it as MCP tools
- **CLI/script usage**: the module is directly importable for standalone scripts and notebooks
