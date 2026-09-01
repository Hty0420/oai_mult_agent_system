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
