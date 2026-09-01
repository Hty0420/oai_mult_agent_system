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
            # MONAI 1.5 returns MetaTensor; convert to numpy for .astype()
            if hasattr(aug_img, "numpy"):
                aug_img = aug_img.numpy()
            aug_label = augmented["label"]
            if hasattr(aug_label, "numpy"):
                aug_label = aug_label.numpy()
            results.append({
                "image": aug_img.astype(image_array.dtype),
                "label": aug_label.astype(mask_array.dtype),
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
            knee_side: 1=left, 2=right.
            aggregator: 'mean' or 'median' for feature aggregation.

        Returns:
            Aggregated feature dict with keys like 'V_Femur', 'V_Femur_std', etc.
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
                '{feature}', '{feature}_std', and 'n_augmentations' (int).
        """
        if not features_list:
            return {}

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
