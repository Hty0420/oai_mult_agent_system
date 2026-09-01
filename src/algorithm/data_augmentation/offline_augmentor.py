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
        progress_callback=None,
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
        successful_cases = 0

        def report(processed, case_id=None, status=None):
            if progress_callback is None:
                return
            try:
                progress_callback(
                    {
                        "phase": "augment",
                        "processed_cases": int(processed),
                        "total_cases": int(len(df)),
                        "successful_cases": int(successful_cases),
                        "failed_cases": int(processed - successful_cases),
                        "case_id": str(case_id) if case_id is not None else None,
                        "case_status": status,
                    }
                )
            except Exception:
                pass

        report(0)
        for processed, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc="Augmenting dataset"),
            start=1,
        ):
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
                report(processed, case_id, "missing_input")
                continue

            try:
                img_array, img_meta = load_nifti(image_path)
                msk_array, msk_meta = load_nifti(mask_path)
            except Exception as e:
                print(f"  Skipping {case_id}: load error - {e}")
                report(processed, case_id, "load_failed")
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
                # MONAI 1.5 returns MetaTensor; convert to numpy for save_nifti
                if hasattr(aug_img, "numpy"):
                    aug_img = aug_img.numpy()
                aug_label = augmented["label"]
                if hasattr(aug_label, "numpy"):
                    aug_label = aug_label.numpy()
                save_nifti(aug_img.astype(img_meta["dtype"]), img_meta, aug_img_path)
                save_nifti(aug_label.astype(msk_meta["dtype"]), msk_meta, aug_msk_path)

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
            successful_cases += 1
            report(processed, case_id, "success")

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
