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
