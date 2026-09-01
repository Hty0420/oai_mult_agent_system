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
    Rand3DElasticd,
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
        Rand3DElasticd(
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

    Order: scale intensity to [0,1] -> geometric -> intensity -> clamp.
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
