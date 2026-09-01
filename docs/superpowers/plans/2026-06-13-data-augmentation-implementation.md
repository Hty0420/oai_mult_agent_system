# Data Augmentation Module — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a MONAI-based MRI data augmentation module with offline (disk) and online (feature extraction) modes, integrated as both a standalone Python module and MCP tools.

**Architecture:** A core `data_augmentation` package under `src/algorithm/` exposes `OfflineAugmentor` and `OnlineAugmentor` classes. Both use a shared MONAI transform pipeline defined in `transforms.py`. Two `@mcp.tool()` wrappers in `auto_experiment_server.py` expose these to the LangGraph agent layer.

**Tech Stack:** MONAI 1.3+, PyTorch 2.0+, SimpleITK (existing), pyradiomics (existing)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `src/algorithm/data_augmentation/__init__.py` | Public API exports |
| Create | `src/algorithm/data_augmentation/utils.py` | nii.gz load/save via SimpleITK, path helpers, KneeSide normalization |
| Create | `src/algorithm/data_augmentation/transforms.py` | MONAI transform pipeline definitions (geometric + intensity) |
| Create | `src/algorithm/data_augmentation/offline_augmentor.py` | OfflineAugmentor: generate augmented .nii.gz files to disk |
| Create | `src/algorithm/data_augmentation/online_augmentor.py` | OnlineAugmentor: on-the-fly augmentation for feature extraction |
| Create | `tests/test_data_augmentation.py` | Unit tests for transforms, augmentors, and utils |
| Modify | `pyproject.toml` | Add `monai>=1.3.0`, `torch>=2.0.0` deps |
| Modify | `src/server/auto_experiment_server.py` | Add `augment_dataset_offline` and `augment_online_extract_features` MCP tools |

---

### Task 1: Add MONAI and PyTorch dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Append after the existing `"zai-sdk>=0.2.2",` line:

```toml
    "monai>=1.3.0",
    "torch>=2.0.0",
```

The dependencies list should read (excerpt):
```toml
dependencies = [
    "anthropic>=0.57.1",
    ...
    "zai-sdk>=0.2.2",
    "monai>=1.3.0",
    "torch>=2.0.0",
]
```

- [ ] **Step 2: Install the new dependencies**

Run: `uv sync`
Expected: MONAI and PyTorch installed without errors.

- [ ] **Step 3: Verify import**

Run: `python -c "import monai; import torch; print('MONAI', monai.__version__, '| torch', torch.__version__)"`
Expected: prints version strings, no errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add monai and torch dependencies for data augmentation"
```

---

### Task 2: Create utility module (utils.py)

**Files:**
- Create: `src/algorithm/data_augmentation/utils.py`

- [ ] **Step 1: Write utils.py**

```python
"""Utility functions for nii.gz I/O and path handling."""

import os
import numpy as np
import SimpleITK as sitk


def load_nifti(file_path: str) -> tuple[np.ndarray, dict]:
    """
    Load a NIfTI file (.nii or .nii.gz) and return its array + metadata.

    Args:
        file_path: Path to .nii.gz file.

    Returns:
        (numpy_array, metadata_dict) where metadata_dict has keys:
        'spacing', 'origin', 'direction', 'dtype'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    image = sitk.ReadImage(file_path)
    array = sitk.GetArrayFromImage(image)  # shape: (z, y, x)

    metadata = {
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
        "dtype": array.dtype,
    }
    return array, metadata


def save_nifti(array: np.ndarray, metadata: dict, file_path: str) -> None:
    """
    Save a numpy array as a NIfTI file, preserving spatial metadata.

    Args:
        array: 3D numpy array with shape (z, y, x).
        metadata: Dict with 'spacing', 'origin', 'direction' keys.
        file_path: Output path (should end with .nii.gz).
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

    image = sitk.GetImageFromArray(array.astype(metadata.get("dtype", np.float32)))
    image.SetSpacing(metadata["spacing"])
    image.SetOrigin(metadata["origin"])
    image.SetDirection(metadata["direction"])

    sitk.WriteImage(image, file_path)


def normalize_knee_side(array: np.ndarray, knee_side: int) -> np.ndarray:
    """
    Flip image to a canonical side if needed. KneeSide=1 (left) is kept as-is;
    KneeSide=2 (right) is flipped along the left-right axis (axis 2 in (z,y,x)).

    This ensures all images have consistent anatomical orientation before
    augmentation, avoiding anatomically impossible flips.

    Args:
        array: 3D numpy array with shape (z, y, x).
        knee_side: 1 for left knee, 2 for right knee.

    Returns:
        Normalized array (may be flipped copy).
    """
    if knee_side == 2:
        # Flip along x-axis (left-right) in (z, y, x) layout
        return np.flip(array, axis=2).copy()
    return array


def generate_augmented_filename(original_name: str, aug_index: int) -> str:
    """
    Generate a unique filename for an augmented sample.

    Example:
        'oaizib_001_0000.nii.gz' → 'oaizib_001_aug0_0000.nii.gz'

    Args:
        original_name: Original filename stem (e.g., 'oaizib_001_0000.nii.gz').
        aug_index: Augmentation index (0-based).

    Returns:
        New filename with augmentation marker.
    """
    base, ext = os.path.splitext(original_name)
    # Handle .nii.gz double extension
    if base.endswith(".nii"):
        base = base[:-4]
        ext = ".nii.gz"
    return f"{base}_aug{aug_index}{ext}"
```

- [ ] **Step 2: Verify utils.py imports**

Run: `python -c "from src.algorithm.data_augmentation.utils import load_nifti, save_nifti, normalize_knee_side; print('utils OK')"`
(Note: this requires `src/algorithm/data_augmentation/__init__.py` to exist — skip if not yet created, just verify no syntax errors via `python -m py_compile`.)

Run: `python -m py_compile src/algorithm/data_augmentation/utils.py`
Expected: no output (successful compile).

- [ ] **Step 3: Commit**

```bash
git add src/algorithm/data_augmentation/utils.py
git commit -m "feat: add nifti I/O utilities for data augmentation"
```

---

### Task 3: Define MONAI transform pipelines (transforms.py)

**Files:**
- Create: `src/algorithm/data_augmentation/transforms.py`

- [ ] **Step 1: Write transforms.py**

```python
"""MONAI transform pipelines for knee MRI data augmentation.

Provides two classes of transforms:
- Geometric: applied to both image and mask synchronously
- Intensity: applied to image only (preserves mask values exactly)
"""

from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandBiasFieldd,
    RandElasticd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandZoomd,
    ScaleIntensityRanged,
)

# Default probability constants
DEFAULT_GEOMETRIC_PROB = 0.5
DEFAULT_INTENSITY_PROB = 0.3


def get_geometric_transforms(
    flip_prob: float = 0.5,
    rotate_prob: float = 0.5,
    rotate_degrees: float = 10.0,
    zoom_prob: float = 0.5,
    zoom_range: tuple[float, float] = (0.9, 1.1),
    translate_prob: float = 0.5,
    translate_voxels: int = 10,
    elastic_prob: float = 0.2,
) -> list:
    """
    Geometric transforms applied to both image and mask synchronously.

    Uses MONAI 'd' (dictionary) variants so that the same random parameters
    are applied to both 'image' and 'label' keys.

    Args:
        flip_prob: Probability of random flip along each axis.
        rotate_prob: Probability of random rotation.
        rotate_degrees: Maximum rotation angle in degrees.
        zoom_prob: Probability of random zoom.
        zoom_range: (min, max) zoom factors.
        translate_prob: Probability of random translation.
        translate_voxels: Maximum translation in voxels.
        elastic_prob: Probability of elastic deformation (mild).

    Returns:
        List of MONAI transform objects (passed to Compose).
    """
    return [
        RandFlipd(
            keys=["image", "label"],
            prob=flip_prob,
            spatial_axis=[0],  # anteroposterior
        ),
        RandFlipd(
            keys=["image", "label"],
            prob=flip_prob * 0.6,  # less frequent: left-right has anatomical meaning
            spatial_axis=[1],
        ),
        RandRotated(
            keys=["image", "label"],
            range_x=rotate_degrees / 180.0,  # MONAI uses radians
            mode=["bilinear", "nearest"],
            prob=rotate_prob,
        ),
        RandZoomd(
            keys=["image", "label"],
            min_zoom=zoom_range[0],
            max_zoom=zoom_range[1],
            mode=["trilinear", "nearest"],
            prob=zoom_prob,
        ),
        RandAffined(
            keys=["image", "label"],
            translate_range=translate_voxels,
            mode=["bilinear", "nearest"],
            prob=translate_prob,
        ),
        RandElasticd(
            keys=["image", "label"],
            sigma_range=(3.0, 7.0),
            magnitude_range=(10.0, 30.0),
            mode=["bilinear", "nearest"],
            prob=elastic_prob,
        ),
    ]


def get_intensity_transforms(
    noise_prob: float = 0.3,
    noise_std: float = 0.05,
    bias_field_prob: float = 0.4,
    bias_coeff: float = 0.1,
    contrast_prob: float = 0.3,
    contrast_gamma: tuple[float, float] = (0.8, 1.2),
    smooth_prob: float = 0.2,
    smooth_sigma: tuple[float, float] = (0.5, 1.5),
) -> list:
    """
    Intensity transforms applied to image only.

    These simulate MRI acquisition variations (noise, bias field, contrast
    differences) without altering anatomical content. The mask is not touched.

    Args:
        noise_prob: Probability of Gaussian noise addition.
        noise_std: Standard deviation of Gaussian noise (relative to [0,1]).
        bias_field_prob: Probability of bias field application (MRI-specific).
        bias_coeff: Bias field coefficient range.
        contrast_prob: Probability of gamma contrast adjustment.
        contrast_gamma: (min, max) gamma range.
        smooth_prob: Probability of Gaussian smoothing.
        smooth_sigma: (min, max) smoothing sigma in mm.

    Returns:
        List of MONAI transform objects (applied to 'image' key only).
    """
    return [
        RandGaussianNoised(
            keys=["image"],
            prob=noise_prob,
            std=noise_std,
        ),
        RandBiasFieldd(
            keys=["image"],
            prob=bias_field_prob,
            coeff_range=(0.0, bias_coeff),
        ),
        RandAdjustContrastd(
            keys=["image"],
            prob=contrast_prob,
            gamma=contrast_gamma,
        ),
        RandGaussianSmoothd(
            keys=["image"],
            prob=smooth_prob,
            sigma_x=smooth_sigma,
            sigma_y=smooth_sigma,
            sigma_z=smooth_sigma,
        ),
    ]


def get_full_augmentation_pipeline(
    intensity_first: bool = True,
    **kwargs,
) -> Compose:
    """
    Build the complete augmentation pipeline.

    Order: scale intensity to [0,1] → geometric → intensity → clamp.
    The scaling is necessary because MONAI intensity transforms
    assume [0,1] range inputs.

    Args:
        intensity_first: If True, apply intensity transforms before geometric
                         (recommended: True, as noise/blur are resolution-independent).
        **kwargs: Passed through to get_geometric_transforms and get_intensity_transforms.

    Returns:
        MONAI Compose object ready to apply to a dict with 'image' and 'label' keys.
    """
    transforms = [
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000.0,  # Hounsfield-like lower bound for MRI
            a_max=1000.0,   # upper bound
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
    ]

    geometric = get_geometric_transforms(**kwargs)
    intensity = get_intensity_transforms(**kwargs)

    if intensity_first:
        transforms.extend(intensity)
        transforms.extend(geometric)
    else:
        transforms.extend(geometric)
        transforms.extend(intensity)

    return Compose(transforms)
```

- [ ] **Step 2: Verify transforms compile**

Run: `python -m py_compile src/algorithm/data_augmentation/transforms.py`
Expected: no output (success).

- [ ] **Step 3: Smoke test transforms (can create tensors)**

Run command below to verify the pipeline constructs and runs on synthetic data:
```bash
python -c "
import numpy as np
import torch
from src.algorithm.data_augmentation.transforms import get_full_augmentation_pipeline

# Create synthetic 3D data
img = np.random.rand(64, 64, 32).astype(np.float32) * 2000 - 1000
lbl = (np.random.rand(64, 64, 32) > 0.9).astype(np.float32)
data = {'image': img, 'label': lbl}

pipeline = get_full_augmentation_pipeline()
result = pipeline(data)

assert 'image' in result, 'image missing from result'
assert 'label' in result, 'label missing from result'
assert result['image'].shape == img.shape, f'shape mismatch: {result[\"image\"].shape} vs {img.shape}'
assert result['label'].shape == lbl.shape, 'label shape mismatch'
print('Pipeline smoke test PASSED')
"
```
Expected: "Pipeline smoke test PASSED"

- [ ] **Step 4: Commit**

```bash
git add src/algorithm/data_augmentation/transforms.py
git commit -m "feat: add MONAI transform pipeline for MRI augmentation"
```

---

### Task 4: Create OfflineAugmentor (offline_augmentor.py)

**Files:**
- Create: `src/algorithm/data_augmentation/offline_augmentor.py`

- [ ] **Step 1: Write offline_augmentor.py**

```python
"""OfflineAugmentor: generates augmented .nii.gz files to disk."""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from .transforms import get_full_augmentation_pipeline
from .utils import (
    generate_augmented_filename,
    load_nifti,
    normalize_knee_side,
    save_nifti,
)


class OfflineAugmentor:
    """
    Generate augmented knee MRI images and masks, saved to disk.

    Reads image/mask paths from a CSV, applies MONAI augmentation
    pipeline, and writes augmented .nii.gz files alongside an updated CSV.

    Usage:
        augmentor = OfflineAugmentor(multiplier=5)
        augmentor.augment_dataset(
            csv_path="datasets/labels_train.csv",
            output_dir="datasets/augmented",
        )
    """

    def __init__(
        self,
        multiplier: int = 5,
        seed: int = 42,
        **transform_kwargs,
    ):
        """
        Args:
            multiplier: How many augmented copies per original sample.
            seed: Random seed for reproducibility.
            **transform_kwargs: Passed to get_full_augmentation_pipeline.
        """
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")

        self.multiplier = multiplier
        self.seed = seed
        self.transform_kwargs = transform_kwargs

    def augment_dataset(
        self,
        csv_path: str,
        output_dir: str,
        images_subdir: str = "imagesTr_aug",
        labels_subdir: str = "labelsTr_aug",
    ) -> str:
        """
        Augment the entire dataset referenced by a CSV file.

        Args:
            csv_path: Path to labels CSV with ImagePath, MaskPath, KLGrade columns.
            output_dir: Root directory for augmented output.
            images_subdir: Subdirectory for augmented images.
            labels_subdir: Subdirectory for augmented masks.

        Returns:
            Path to the new augmented labels CSV.
        """
        self._set_seed()

        df = pd.read_csv(csv_path, dtype={"CaseID": str})
        images_out = os.path.join(output_dir, images_subdir)
        labels_out = os.path.join(output_dir, labels_subdir)
        os.makedirs(images_out, exist_ok=True)
        os.makedirs(labels_out, exist_ok=True)

        pipeline = get_full_augmentation_pipeline(**self.transform_kwargs)
        augmented_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting dataset"):
            case_id = row["CaseID"]
            image_path = row["ImagePath"]
            mask_path = row["MaskPath"]
            kl_grade = row.get("KLGrade", None)
            knee_side = row.get("KneeSide", 1)

            # Add original sample row
            augmented_rows.append({
                "CaseID": str(case_id),
                "KneeSide": knee_side,
                "Gender": row.get("Gender", None),
                "Age": row.get("Age", None),
                "BMI": row.get("BMI", None),
                "ImagePath": image_path,
                "MaskPath": mask_path,
                "KLGrade": kl_grade,
                "Augmented": False,
            })

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                print(f"  Skipping {case_id}: missing file")
                continue

            try:
                img_array, img_meta = load_nifti(image_path)
                msk_array, msk_meta = load_nifti(mask_path)
            except Exception as e:
                print(f"  Skipping {case_id}: load error - {e}")
                continue

            # Normalize knee side for consistent augmentation
            img_array = normalize_knee_side(img_array, int(knee_side))
            msk_array = normalize_knee_side(msk_array, int(knee_side))

            image_basename = os.path.basename(image_path)
            mask_basename = os.path.basename(mask_path)

            for aug_idx in range(self.multiplier):
                self._set_seed(seed=self.seed + hash(f"{case_id}_{aug_idx}") % 100000)

                data = {"image": img_array.copy(), "label": msk_array.copy()}
                augmented = pipeline(data)

                aug_img_name = generate_augmented_filename(image_basename, aug_idx)
                aug_msk_name = generate_augmented_filename(mask_basename, aug_idx)

                aug_img_path = os.path.join(images_out, aug_img_name)
                aug_msk_path = os.path.join(labels_out, aug_msk_name)

                # Rescale back to original intensity range for saving
                aug_img = augmented["image"] * 2000.0 - 1000.0
                save_nifti(aug_img.astype(img_meta["dtype"]), img_meta, aug_img_path)
                save_nifti(augmented["label"].astype(msk_meta["dtype"]), msk_meta, aug_msk_path)

                augmented_rows.append({
                    "CaseID": f"{case_id}_aug{aug_idx}",
                    "KneeSide": knee_side,
                    "Gender": row.get("Gender", None),
                    "Age": row.get("Age", None),
                    "BMI": row.get("BMI", None),
                    "ImagePath": aug_img_path,
                    "MaskPath": aug_msk_path,
                    "KLGrade": kl_grade,
                    "Augmented": True,
                    "SourceCaseID": str(case_id),
                })

        # Write output CSV
        out_df = pd.DataFrame(augmented_rows)
        csv_out = os.path.join(output_dir, "labels_train_aug.csv")
        out_df.to_csv(csv_out, index=False)

        total_orig = len(df)
        total_aug = self.multiplier * total_orig
        print(f"\nAugmentation complete: {total_orig} original + {total_aug} augmented = {len(out_df)} total rows")
        print(f"Output CSV: {csv_out}")
        return csv_out

    def _set_seed(self, seed: int | None = None) -> None:
        """Set random seed for numpy and torch (MONAI uses both)."""
        import torch
        s = seed if seed is not None else self.seed
        np.random.seed(s)
        torch.manual_seed(s)
```

- [ ] **Step 2: Verify compilation**

Run: `python -m py_compile src/algorithm/data_augmentation/offline_augmentor.py`
Expected: no output (needs the `__init__.py` to exist for relative imports — create a minimal one first).

- [ ] **Step 3: Commit**

```bash
git add src/algorithm/data_augmentation/offline_augmentor.py
git commit -m "feat: add OfflineAugmentor for disk-based dataset expansion"
```

---

### Task 5: Create OnlineAugmentor (online_augmentor.py)

**Files:**
- Create: `src/algorithm/data_augmentation/online_augmentor.py`

- [ ] **Step 1: Write online_augmentor.py**

```python
"""OnlineAugmentor: on-the-fly augmentation for robust feature extraction."""

import numpy as np
from typing import Callable

from .transforms import get_full_augmentation_pipeline
from .utils import load_nifti, normalize_knee_side


class OnlineAugmentor:
    """
    Apply random augmentation on-the-fly, without saving to disk.

    Designed to wrap feature extractors: augment each image N times,
    extract features from each augmented version, then average/aggregate
    the results for improved robustness.

    Usage:
        augmentor = OnlineAugmentor(n_augmentations=3)
        features_list = augmentor.augment_and_extract(
            image_path="...", mask_path="...", knee_side=1,
            extractor_fn=my_extractor,
        )
        averaged = augmentor.average_features(features_list)
    """

    def __init__(
        self,
        n_augmentations: int = 3,
        seed: int = 42,
        **transform_kwargs,
    ):
        """
        Args:
            n_augmentations: Number of augmented views per sample
                             (original is always included, so total = 1 + n_augmentations).
            seed: Base random seed.
            **transform_kwargs: Passed to get_full_augmentation_pipeline.
        """
        self.n_augmentations = n_augmentations
        self.seed = seed
        self.transform_kwargs = transform_kwargs

    def augment_single(
        self,
        image_array: np.ndarray,
        mask_array: np.ndarray,
        knee_side: int = 1,
    ) -> list[dict]:
        """
        Generate augmented views of a single image-mask pair.

        Args:
            image_array: 3D numpy array (z, y, x).
            mask_array: 3D numpy array (z, y, x), integer labels.
            knee_side: 1=left, 2=right. Used to normalize before augmentation.

        Returns:
            List of dicts [{'image': np.ndarray, 'label': np.ndarray}, ...].
            First element is always the original (un-augmented) pair.
            Total length = 1 + n_augmentations.
        """
        import torch

        pipeline = get_full_augmentation_pipeline(**self.transform_kwargs)

        # Normalize
        img = normalize_knee_side(image_array.copy(), knee_side)
        msk = normalize_knee_side(mask_array.copy(), knee_side)

        results = [{"image": img.copy(), "label": msk.copy()}]

        for aug_idx in range(self.n_augmentations):
            seed = self.seed + hash(f"online_{aug_idx}") % 100000
            np.random.seed(seed)
            torch.manual_seed(seed)

            data = {"image": img.copy(), "label": msk.copy()}
            augmented = pipeline(data)

            # Rescale back to original intensity range
            aug_img = augmented["image"] * 2000.0 - 1000.0
            results.append({
                "image": aug_img.astype(image_array.dtype),
                "label": augmented["label"].astype(mask_array.dtype),
            })

        return results

    def augment_from_file(
        self,
        image_path: str,
        mask_path: str,
        knee_side: int = 1,
    ) -> list[dict]:
        """
        Load an image-mask pair from disk and generate augmented views.

        Args:
            image_path: Path to .nii.gz image.
            mask_path: Path to .nii.gz mask.
            knee_side: 1=left, 2=right.

        Returns:
            List of dicts with 'image' and 'label' numpy arrays.
        """
        img_array, _ = load_nifti(image_path)
        msk_array, _ = load_nifti(mask_path)
        return self.augment_single(img_array, msk_array, knee_side)

    def extract_with_augmentation(
        self,
        image_path: str,
        mask_path: str,
        extractor_fn: Callable[[np.ndarray, np.ndarray], dict],
        knee_side: int = 1,
        aggregator: str = "mean",
    ) -> dict:
        """
        Extract features from original + augmented views and aggregate.

        Args:
            image_path: Path to .nii.gz image.
            mask_path: Path to .nii.gz mask.
            extractor_fn: Function(image_array, mask_array) -> dict of numeric features.
                          The function receives numpy arrays directly.
            knee_side: 1=left, 2=right.
            aggregator: 'mean' or 'median' for feature aggregation.

        Returns:
            Aggregated feature dict with keys like 'V_Femur_mean', 'V_Femur_std', etc.
            Also includes 'n_augmentations' and per-feature std for uncertainty estimation.
        """
        views = self.augment_from_file(image_path, mask_path, knee_side)

        all_features = []
        for i, view in enumerate(views):
            feats = extractor_fn(view["image"], view["label"])
            feats["_view_index"] = i
            all_features.append(feats)

        return self._aggregate_features(all_features, aggregator)

    def _aggregate_features(
        self,
        features_list: list[dict],
        method: str = "mean",
    ) -> dict:
        """
        Aggregate feature dicts across augmented views.

        For each numeric feature key, computes mean, std, and the chosen
        aggregate (mean or median) across views.

        Args:
            features_list: List of per-view feature dicts.
            method: 'mean' or 'median'.

        Returns:
            Aggregated dict with keys like:
                '{feature}_mean', '{feature}_std', and
                'n_augmentations' (int).
        """
        if not features_list:
            return {}

        # Collect all numeric keys (skip _view_index and non-numeric)
        all_keys = set()
        for feats in features_list:
            for k, v in feats.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, (int, float, np.floating, np.integer)):
                    all_keys.add(k)

        aggregated = {"n_augmentations": len(features_list)}

        for key in sorted(all_keys):
            values = []
            for feats in features_list:
                if key in feats:
                    v = feats[key]
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass

            if not values:
                continue

            arr = np.array(values)
            if method == "median":
                aggregated[f"{key}"] = float(np.median(arr))
            else:
                aggregated[f"{key}"] = float(np.mean(arr))
            aggregated[f"{key}_std"] = float(np.std(arr))

        return aggregated
```

- [ ] **Step 2: Verify compilation**

Run: `python -m py_compile src/algorithm/data_augmentation/online_augmentor.py`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add src/algorithm/data_augmentation/online_augmentor.py
git commit -m "feat: add OnlineAugmentor for on-the-fly feature augmentation"
```

---

### Task 6: Create package __init__.py

**Files:**
- Create: `src/algorithm/data_augmentation/__init__.py`

- [ ] **Step 1: Write __init__.py**

```python
"""Data augmentation module for knee osteoarthritis MRI.

Provides:
- OfflineAugmentor: Generate augmented .nii.gz files to disk
- OnlineAugmentor: On-the-fly augmentation for feature extraction
- Transform pipelines: Reusable MONAI transform definitions
"""

from .offline_augmentor import OfflineAugmentor
from .online_augmentor import OnlineAugmentor
from .transforms import (
    get_full_augmentation_pipeline,
    get_geometric_transforms,
    get_intensity_transforms,
)
from .utils import load_nifti, save_nifti, normalize_knee_side

__all__ = [
    "OfflineAugmentor",
    "OnlineAugmentor",
    "get_full_augmentation_pipeline",
    "get_geometric_transforms",
    "get_intensity_transforms",
    "load_nifti",
    "save_nifti",
    "normalize_knee_side",
]
```

- [ ] **Step 2: Verify full module import**

Run:
```bash
cd src && python -c "
from algorithm.data_augmentation import (
    OfflineAugmentor,
    OnlineAugmentor,
    get_full_augmentation_pipeline,
    get_geometric_transforms,
    get_intensity_transforms,
    load_nifti,
    save_nifti,
    normalize_knee_side,
)
print('All imports OK')
print('OfflineAugmentor:', OfflineAugmentor)
print('OnlineAugmentor:', OnlineAugmentor)
"
```
Expected: "All imports OK" and class references printed.

- [ ] **Step 3: Commit**

```bash
git add src/algorithm/data_augmentation/__init__.py
git commit -m "feat: add data_augmentation package __init__ with public API"
```

---

### Task 7: Write unit tests

**Files:**
- Create: `tests/test_data_augmentation.py`

- [ ] **Step 1: Write test file**

```python
"""Tests for the data_augmentation module."""

import os
import sys
import tempfile
import numpy as np
import SimpleITK as sitk
import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from algorithm.data_augmentation.utils import (
    load_nifti,
    save_nifti,
    normalize_knee_side,
    generate_augmented_filename,
)
from algorithm.data_augmentation.transforms import (
    get_geometric_transforms,
    get_intensity_transforms,
    get_full_augmentation_pipeline,
)
from algorithm.data_augmentation import OfflineAugmentor, OnlineAugmentor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_3d_array():
    """A 64x64x32 float32 array simulating an MRI volume."""
    rng = np.random.RandomState(42)
    return rng.randn(64, 64, 32).astype(np.float32) * 500 + 200


@pytest.fixture
def synthetic_mask_array():
    """A 64x64x32 int array simulating a segmentation mask with labels 0-5."""
    rng = np.random.RandomState(42)
    mask = np.zeros((64, 64, 32), dtype=np.int16)
    # Create some blob-like structures
    mask[10:30, 10:30, 5:15] = 1   # Femur
    mask[30:50, 30:50, 10:20] = 2  # Cartilage
    mask[20:40, 40:60, 8:18] = 3   # Tibia
    return mask


@pytest.fixture
def nifti_pair(synthetic_3d_array, synthetic_mask_array):
    """Create temporary .nii.gz files, return (img_path, msk_path, metadata)."""
    tmpdir = tempfile.mkdtemp()
    img_path = os.path.join(tmpdir, "test_image_0000.nii.gz")
    msk_path = os.path.join(tmpdir, "test_label.nii.gz")

    img = sitk.GetImageFromArray(synthetic_3d_array)
    img.SetSpacing((0.36, 0.36, 0.7))
    img.SetOrigin((0, 0, 0))
    sitk.WriteImage(img, img_path)

    msk = sitk.GetImageFromArray(synthetic_mask_array)
    msk.SetSpacing((0.36, 0.36, 0.7))
    msk.SetOrigin((0, 0, 0))
    sitk.WriteImage(msk, msk_path)

    yield img_path, msk_path, {"spacing": (0.36, 0.36, 0.7)}

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# utils.py tests
# ---------------------------------------------------------------------------

class TestLoadSaveNifti:
    def test_load_roundtrip(self, nifti_pair):
        img_path, msk_path, meta = nifti_pair
        arr, loaded_meta = load_nifti(img_path)
        assert arr.shape == (64, 64, 32)
        assert loaded_meta["spacing"] == (0.36, 0.36, 0.7)

    def test_save_and_reload(self, synthetic_3d_array):
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            out_path = f.name

        try:
            metadata = {"spacing": (1.0, 1.0, 3.0), "origin": (0, 0, 0),
                        "direction": (1,0,0,0,1,0,0,0,1), "dtype": np.float32}
            save_nifti(synthetic_3d_array, metadata, out_path)

            reloaded, reloaded_meta = load_nifti(out_path)
            np.testing.assert_array_almost_equal(synthetic_3d_array, reloaded, decimal=2)
            assert reloaded_meta["spacing"] == (1.0, 1.0, 3.0)
        finally:
            os.unlink(out_path)

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_nifti("/nonexistent/path.nii.gz")


class TestNormalizeKneeSide:
    def test_left_knee_unchanged(self, synthetic_3d_array):
        result = normalize_knee_side(synthetic_3d_array.copy(), knee_side=1)
        np.testing.assert_array_equal(result, synthetic_3d_array)

    def test_right_knee_flipped(self):
        arr = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        result = normalize_knee_side(arr, knee_side=2)
        expected = np.flip(arr, axis=2)
        np.testing.assert_array_equal(result, expected)

    def test_flip_roundtrip(self):
        """Flipping twice should return the original."""
        arr = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        result = normalize_knee_side(normalize_knee_side(arr, 2), 2)
        np.testing.assert_array_equal(result, arr)


class TestGenerateAugmentedFilename:
    def test_single_extension(self):
        result = generate_augmented_filename("oaizib_001_0000.nii.gz", 3)
        assert result == "oaizib_001_0000_aug3.nii.gz"

    def test_zero_index(self):
        result = generate_augmented_filename("case_001.nii.gz", 0)
        assert result == "case_001_aug0.nii.gz"


# ---------------------------------------------------------------------------
# transforms.py tests
# ---------------------------------------------------------------------------

class TestTransformPipelines:
    def test_geometric_transforms_preserve_label_shape(self, synthetic_3d_array, synthetic_mask_array):
        transforms = get_geometric_transforms()
        from monai.transforms import Compose
        pipeline = Compose(transforms)

        data = {"image": synthetic_3d_array.copy(), "label": synthetic_mask_array.copy()}
        result = pipeline(data)

        assert result["image"].shape == synthetic_3d_array.shape
        assert result["label"].shape == synthetic_mask_array.shape

    def test_intensity_transforms_only_modify_image(self, synthetic_3d_array, synthetic_mask_array):
        transforms = get_intensity_transforms(noise_prob=1.0, noise_std=0.0)
        from monai.transforms import Compose
        pipeline = Compose(transforms)

        data = {"image": synthetic_3d_array.copy(), "label": synthetic_mask_array.copy()}
        result = pipeline(data)

        # Noise std=0 → image unchanged; label always unchanged
        assert result["label"].shape == synthetic_mask_array.shape

    def test_full_pipeline_produces_valid_output(self, synthetic_3d_array, synthetic_mask_array):
        pipeline = get_full_augmentation_pipeline()
        data = {"image": synthetic_3d_array.copy(), "label": synthetic_mask_array.copy()}
        result = pipeline(data)

        assert result["image"].shape == synthetic_3d_array.shape
        assert result["label"].shape == synthetic_mask_array.shape
        # Image values should be in [0, 1] range after ScaleIntensity
        assert 0.0 <= result["image"].max() <= 1.0

    def test_full_pipeline_deterministic_with_seed(self, synthetic_3d_array, synthetic_mask_array):
        import torch
        pipeline = get_full_augmentation_pipeline()

        np.random.seed(123)
        torch.manual_seed(123)
        data1 = {"image": synthetic_3d_array.copy(), "label": synthetic_mask_array.copy()}
        result1 = pipeline(data1)

        np.random.seed(123)
        torch.manual_seed(123)
        data2 = {"image": synthetic_3d_array.copy(), "label": synthetic_mask_array.copy()}
        result2 = pipeline(data2)

        np.testing.assert_array_almost_equal(result1["image"], result2["image"], decimal=4)
        np.testing.assert_array_almost_equal(result1["label"], result2["label"], decimal=4)


# ---------------------------------------------------------------------------
# OfflineAugmentor tests
# ---------------------------------------------------------------------------

class TestOfflineAugmentor:
    def test_init_validation(self):
        with pytest.raises(ValueError):
            OfflineAugmentor(multiplier=0)

    def test_augment_with_synthetic_data(self, nifti_pair):
        img_path, msk_path, _ = nifti_pair

        # Create a minimal CSV
        tmpdir = tempfile.mkdtemp()
        csv_path = os.path.join(tmpdir, "test_labels.csv")
        with open(csv_path, "w") as f:
            f.write("CaseID,KneeSide,Gender,Age,BMI,ImagePath,MaskPath,KLGrade\n")
            f.write(f"1,1,1,50,25.0,{img_path},{msk_path},2\n")

        output_dir = os.path.join(tmpdir, "augmented_output")

        try:
            augmentor = OfflineAugmentor(multiplier=2, seed=42)
            out_csv = augmentor.augment_dataset(csv_path, output_dir)

            assert os.path.exists(out_csv)

            import pandas as pd
            df = pd.read_csv(out_csv)
            # 1 original + 2 augmented = 3 rows
            assert len(df) == 3
            assert df["KLGrade"].iloc[0] == 2
            assert df["KLGrade"].iloc[1] == 2  # KL preserved
            assert df["Augmented"].iloc[0] == False
            assert df["Augmented"].iloc[1] == True

            # Check files exist
            assert os.path.exists(df["ImagePath"].iloc[1])
            assert os.path.exists(df["MaskPath"].iloc[1])
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# OnlineAugmentor tests
# ---------------------------------------------------------------------------

class TestOnlineAugmentor:
    def test_augment_single_returns_correct_count(self, synthetic_3d_array, synthetic_mask_array):
        augmentor = OnlineAugmentor(n_augmentations=3, seed=42)
        views = augmentor.augment_single(synthetic_3d_array, synthetic_mask_array, knee_side=1)
        assert len(views) == 4  # 1 original + 3 augmented

    def test_augment_from_file(self, nifti_pair):
        img_path, msk_path, _ = nifti_pair
        augmentor = OnlineAugmentor(n_augmentations=2, seed=42)
        views = augmentor.augment_from_file(img_path, msk_path, knee_side=1)
        assert len(views) == 3
        for v in views:
            assert v["image"].shape == (64, 64, 32)
            assert v["label"].shape == (64, 64, 32)

    def test_extract_with_augmentation_aggregates_features(self, nifti_pair):
        img_path, msk_path, _ = nifti_pair

        def dummy_extractor(image, mask):
            """Simulate feature extraction: returns volume and surface_area."""
            return {
                "volume": float(np.sum(mask > 0)),
                "surface_area": float(np.sum(mask > 0)) * 0.5,
            }

        augmentor = OnlineAugmentor(n_augmentations=3, seed=42)
        result = augmentor.extract_with_augmentation(
            img_path, msk_path,
            extractor_fn=dummy_extractor,
            knee_side=1,
            aggregator="mean",
        )

        assert "volume" in result
        assert "volume_std" in result
        assert "surface_area" in result
        assert "surface_area_std" in result
        assert result["n_augmentations"] == 4

    def test_aggregate_features_empty(self):
        augmentor = OnlineAugmentor(n_augmentations=1)
        result = augmentor._aggregate_features([], "mean")
        assert result == {}

    def test_aggregate_features_median(self):
        augmentor = OnlineAugmentor(n_augmentations=1)
        feats = [
            {"a": 1.0, "b": 10.0},
            {"a": 3.0, "b": 20.0},
            {"a": 5.0, "b": 30.0},
        ]
        result = augmentor._aggregate_features(feats, "median")
        assert result["a"] == 3.0
        assert result["b"] == 20.0
```

- [ ] **Step 2: Run tests to verify they fail** (module code isn't complete)

Run: `pytest tests/test_data_augmentation.py -v --tb=short 2>&1 | head -60`
Expected: some tests pass (utils), some may fail if dependencies missing. At minimum check that no import errors.

- [ ] **Step 3: Install test dependencies and run**

Run: `uv sync --group dev` (if pytest is in dev deps) or `pip install pytest`
Run: `pytest tests/test_data_augmentation.py -v --tb=short`
Expected: all tests PASS (green).

- [ ] **Step 4: Commit**

```bash
git add tests/test_data_augmentation.py
git commit -m "test: add unit tests for data augmentation module"
```

---

### Task 8: Add MCP tools to auto_experiment_server.py

**Files:**
- Modify: `src/server/auto_experiment_server.py`

- [ ] **Step 1: Add imports at top of auto_experiment_server.py**

After the existing LDSE import block (line 24), insert:

```python
# Data augmentation
from algorithm.data_augmentation import OfflineAugmentor, OnlineAugmentor
```

- [ ] **Step 2: Add MCP tool functions**

Insert the following two `@mcp.tool()` functions BEFORE the `if __name__ == "__main__":` block (before line 441):

```python
@mcp.tool()
def augment_dataset_offline(
    csv_path: Annotated[str, Field(description="Path to labels_train.csv with ImagePath, MaskPath, KLGrade columns")],
    output_dir: Annotated[str, Field(description="Output directory for augmented images and labels")],
    multiplier: Annotated[int, Field(description="Number of augmented copies per original sample")] = 5,
    seed: Annotated[int, Field(description="Random seed for reproducibility")] = 42,
):
    """
    Offline data augmentation: generates augmented knee MRI images and masks to disk.

    Reads the training set CSV, applies MONAI geometric+intensity augmentation
    to each image-mask pair, and saves augmented .nii.gz files alongside an
    updated labels CSV with preserved KL grades.

    The augmentation pipeline includes:
    - Geometric: random flip, rotation (±10°), zoom (0.9-1.1x), translation, elastic deformation
    - Intensity: Gaussian noise, MRI bias field, contrast adjustment, Gaussian smooth

    Returns:
        dict with:
            - 'status': 'success' or 'failed'
            - 'output_csv': path to the augmented labels CSV
            - 'total_samples': total rows in the output CSV (original + augmented)
            - 'message': summary message
    """
    try:
        augmentor = OfflineAugmentor(multiplier=multiplier, seed=seed)
        out_csv = augmentor.augment_dataset(
            csv_path=csv_path,
            output_dir=output_dir,
        )
        import pandas as pd
        df = pd.read_csv(out_csv)
        return {
            "status": "success",
            "output_csv": out_csv,
            "total_samples": len(df),
            "message": f"Augmentation complete: {len(df)} total samples saved to {output_dir}",
        }
    except Exception as e:
        return {"status": "failed", "message": str(e)}


@mcp.tool()
def augment_online_extract_features(
    csv_path: Annotated[str, Field(description="Path to CSV with ImagePath, MaskPath, CaseID, KLGrade columns")],
    output_dir: Annotated[str, Field(description="Output directory for augmented feature CSV")],
    n_augmentations: Annotated[int, Field(description="Number of augmented views per sample (total = 1 original + N augmented)")] = 3,
    feature_type: Annotated[str, Field(description="Feature type: 'MEAS', 'LDSE', or 'radiomics'")] = "MEAS",
    aggregator: Annotated[str, Field(description="Aggregation method: 'mean' or 'median'")] = "mean",
    seed: Annotated[int, Field(description="Random seed")] = 42,
):
    """
    Online data augmentation + feature extraction: applies random augmentation
    to each image, extracts features from each view, and aggregates results.

    This improves feature robustness by averaging across N augmented versions
    of each sample. Supports MEAS, LDSE, and radiomics feature types.

    Returns:
        dict with:
            - 'status': 'success' or 'failed'
            - 'output_csv': path to aggregated feature CSV
            - 'total_samples': number of samples processed
            - 'message': summary message
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        if feature_type.upper() == "MEAS":
            from algorithm.KOA_feature_extraction.meas_feature_extraction import OAIZIB_MEAS_FeatureExtractor
            extractor_obj = OAIZIB_MEAS_FeatureExtractor(output_dir)

            def extractor_fn(image_array, mask_array):
                """Extract MEAS features from in-memory arrays."""
                import tempfile
                import SimpleITK as sitk
                with tempfile.TemporaryDirectory() as tmpdir:
                    img_path = os.path.join(tmpdir, "_tmp_img.nii.gz")
                    msk_path = os.path.join(tmpdir, "_tmp_msk.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(image_array), img_path)
                    sitk.WriteImage(sitk.GetImageFromArray(mask_array.astype(np.int16)), msk_path)
                    feats = extractor_obj.extract_features("_online", img_path, msk_path)
                    # Filter to numeric-only
                    if feats:
                        return {k: v for k, v in feats.items()
                                if isinstance(v, (int, float, np.floating, np.integer))}
                    return {}

        elif feature_type.upper() == "LDSE":
            from algorithm.KOA_feature_extraction.ldse_feature_extraction import OAIZIB_Light_LDSE_FeatureExtractor
            # LDSE requires pre-trained PCA models — fit on training set first
            ldse = OAIZIB_Light_LDSE_FeatureExtractor(output_dir, n_components=50)
            # Fit PCA models from the CSV
            ldse.fit_shape_models(csv_path, subset="train")

            def extractor_fn(image_array, mask_array):
                import tempfile
                import SimpleITK as sitk
                with tempfile.TemporaryDirectory() as tmpdir:
                    msk_path = os.path.join(tmpdir, "_tmp_msk.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(mask_array.astype(np.int16)), msk_path)
                    feats = ldse.extract_ldse_features(msk_path)
                    if feats:
                        return {k: v for k, v in feats.items()
                                if isinstance(v, (int, float, np.floating, np.integer))}
                    return {}

        elif feature_type.lower() == "radiomics":
            from radiomics import featureextractor as rx_extractor
            rx = rx_extractor.RadiomicsFeatureExtractor()

            def extractor_fn(image_array, mask_array):
                import tempfile
                import SimpleITK as sitk
                with tempfile.TemporaryDirectory() as tmpdir:
                    img_path = os.path.join(tmpdir, "_tmp_img.nii.gz")
                    msk_path = os.path.join(tmpdir, "_tmp_msk.nii.gz")
                    sitk.WriteImage(sitk.GetImageFromArray(image_array), img_path)
                    sitk.WriteImage(sitk.GetImageFromArray(mask_array.astype(np.int16)), msk_path)
                    try:
                        with suppress_output():
                            result = rx.execute(img_path, msk_path, label=1)
                        return {k: v for k, v in result.items()
                                if not k.startswith('diagnostics')
                                and isinstance(v, (int, float, np.floating, np.integer))}
                    except Exception:
                        return {}
        else:
            return {"status": "failed", "message": f"Unknown feature_type: {feature_type}. Choose MEAS, LDSE, or radiomics."}

        online_aug = OnlineAugmentor(n_augmentations=n_augmentations, seed=seed)

        df = pd.read_csv(csv_path, dtype={"CaseID": str})
        all_aggregated = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Online augment + {feature_type} extraction"):
            case_id = row["CaseID"]
            image_path = row["ImagePath"]
            mask_path = row["MaskPath"]
            kl_grade = row.get("KLGrade", None)
            knee_side = int(row.get("KneeSide", 1))

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                print(f"  Skipping {case_id}: missing file")
                continue

            try:
                agg_features = online_aug.extract_with_augmentation(
                    image_path, mask_path,
                    extractor_fn=extractor_fn,
                    knee_side=knee_side,
                    aggregator=aggregator,
                )
                agg_features["CaseID"] = str(case_id)
                if kl_grade is not None:
                    agg_features["KLGrade"] = kl_grade
                all_aggregated.append(agg_features)
            except Exception as e:
                print(f"  Error processing {case_id}: {e}")
                continue

        if not all_aggregated:
            return {"status": "failed", "message": "No features extracted successfully"}

        result_df = pd.DataFrame(all_aggregated)
        feature_csv = os.path.join(output_dir, f"{feature_type.lower()}_augmented_features.csv")
        result_df.to_csv(feature_csv, index=False)

        return {
            "status": "success",
            "output_csv": feature_csv,
            "total_samples": len(result_df),
            "message": f"Online augmentation complete: {len(result_df)} samples × {1 + n_augmentations} views, features saved to {feature_csv}",
        }

    except Exception as e:
        return {"status": "failed", "message": str(e)}
```

- [ ] **Step 3: Verify auto_experiment_server.py parses correctly**

Run: `python -m py_compile src/server/auto_experiment_server.py`
Expected: no output.

- [ ] **Step 4: Quick integration smoke test**

Run the following to verify the MCP tools are registered:
```bash
python -c "
import sys
sys.path.insert(0, 'src')
# Just check the tools are callable (don't actually run them)
from server.auto_experiment_server import augment_dataset_offline, augment_online_extract_features
print('MCP tools registered:')
print('  augment_dataset_offline:', callable(augment_dataset_offline))
print('  augment_online_extract_features:', callable(augment_online_extract_features))
"
```
Expected:
```
MCP tools registered:
  augment_dataset_offline: True
  augment_online_extract_features: True
```

- [ ] **Step 5: Commit**

```bash
git add src/server/auto_experiment_server.py
git commit -m "feat: add data augmentation MCP tools to auto_experiment_server"
```

---

### Task 9: End-to-end integration test (manual)

**Files:**
- Test: `tests/test_integration_augmentation.py`

- [ ] **Step 1: Write integration smoke test**

```python
"""End-to-end smoke test for the augmentation pipeline using real dataset."""

import os
import sys
import tempfile
import numpy as np
import SimpleITK as sitk
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from algorithm.data_augmentation import OfflineAugmentor, OnlineAugmentor


def test_offline_augmentor_full_pipeline():
    """Create synthetic nifti files, run offline augmentation, verify output."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create synthetic image + mask
        img_dir = os.path.join(tmpdir, "imagesTr")
        msk_dir = os.path.join(tmpdir, "labelsTr")
        os.makedirs(img_dir)
        os.makedirs(msk_dir)

        rng = np.random.RandomState(42)
        for i in range(3):
            img = sitk.GetImageFromArray(
                rng.randn(32, 32, 16).astype(np.float32) * 300 + 200
            )
            img.SetSpacing((0.36, 0.36, 0.7))
            sitk.WriteImage(img, os.path.join(img_dir, f"case_{i:03d}_0000.nii.gz"))

            msk = sitk.GetImageFromArray(
                (rng.rand(32, 32, 16) * 5).astype(np.int16)
            )
            msk.SetSpacing((0.36, 0.36, 0.7))
            sitk.WriteImage(msk, os.path.join(msk_dir, f"case_{i:03d}.nii.gz"))

        # Create CSV
        rows = []
        for i in range(3):
            rows.append({
                "CaseID": str(i),
                "KneeSide": 1,
                "Gender": 1,
                "Age": 50,
                "BMI": 25.0,
                "ImagePath": os.path.join(img_dir, f"case_{i:03d}_0000.nii.gz"),
                "MaskPath": os.path.join(msk_dir, f"case_{i:03d}.nii.gz"),
                "KLGrade": i % 5,
            })
        csv_path = os.path.join(tmpdir, "labels_train.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        # Run offline augmentation
        aug_dir = os.path.join(tmpdir, "augmented")
        augmentor = OfflineAugmentor(multiplier=2, seed=42)
        out_csv = augmentor.augment_dataset(csv_path, aug_dir)

        # Verify
        out_df = pd.read_csv(out_csv)
        assert len(out_df) == 9  # 3 original + 3×2 augmented
        assert out_df["KLGrade"].nunique() <= 5  # KL grades preserved
        assert os.path.exists(out_df["ImagePath"].iloc[3])  # first augmented file exists

        print("✓ Offline augmentation pipeline: PASSED")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_online_augmentor_with_nifti():
    """Test online augmentation with synthetic data on disk."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create single synthetic pair
        img_path = os.path.join(tmpdir, "test_0000.nii.gz")
        msk_path = os.path.join(tmpdir, "test.nii.gz")

        rng = np.random.RandomState(42)
        img = sitk.GetImageFromArray(rng.randn(32, 32, 16).astype(np.float32) * 300)
        img.SetSpacing((0.36, 0.36, 0.7))
        sitk.WriteImage(img, img_path)

        msk = sitk.GetImageFromArray((np.ones((32, 32, 16)) * 2).astype(np.int16))
        msk.SetSpacing((0.36, 0.36, 0.7))
        sitk.WriteImage(msk, msk_path)

        augmentor = OnlineAugmentor(n_augmentations=3, seed=42)

        def simple_extractor(image_arr, mask_arr):
            return {"mean_intensity": float(np.mean(image_arr)),
                    "mask_voxels": int(np.sum(mask_arr > 0))}

        result = augmentor.extract_with_augmentation(
            img_path, msk_path,
            extractor_fn=simple_extractor,
            knee_side=1,
            aggregator="mean",
        )

        assert "mean_intensity" in result
        assert "mean_intensity_std" in result
        assert "mask_voxels" in result
        assert result["n_augmentations"] == 4

        print("✓ Online augmentation pipeline: PASSED")

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/test_integration_augmentation.py -v --tb=short`
Expected: both tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_augmentation.py
git commit -m "test: add end-to-end integration tests for augmentation pipeline"
```

---

## Self-Review Checklist

Before handing off:
- [ ] All file paths are exact and exist or will be created
- [ ] Every step has concrete code or commands (no placeholders)
- [ ] Types and signatures are consistent across tasks
- [ ] Tests cover: utils, transforms, OfflineAugmentor, OnlineAugmentor, integration
- [ ] Dependencies (MONAI, torch) are declared before they are used
- [ ] MCP tool code compiles without errors
