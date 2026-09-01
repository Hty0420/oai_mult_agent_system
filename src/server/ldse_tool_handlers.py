"""Dependency-light, leakage-safe entry points for LDSE extraction."""

import os

from algorithm.KOA_feature_extraction.ldse_feature_extraction import (
    OAIZIB_Light_LDSE_FeatureExtractor,
    run_train_test_ldse_extraction,
    stable_mask_array_sample_key,
)


def prepare_ldse_online_augmentation_extractor(
    output_dir,
    csv_path,
    dataset_role,
    model_path=None,
    n_components=50,
    random_seed=42,
) -> OAIZIB_Light_LDSE_FeatureExtractor:
    """Prepare online LDSE augmentation with an explicit fit/load boundary."""
    if dataset_role not in {"train", "test"}:
        raise ValueError("dataset_role must be explicitly set to 'train' or 'test'")
    if dataset_role == "test" and not model_path:
        raise ValueError("dataset_role='test' requires model_path")

    extractor = OAIZIB_Light_LDSE_FeatureExtractor(
        output_dir, n_components=n_components, random_seed=random_seed
    )
    if dataset_role == "train":
        extractor.fit_shape_models(csv_path)
    else:
        extractor.load_shape_models(model_path)
    return extractor


def execute_ldse_train_test_extraction(
    output_dir,
    train_csv_path,
    test_csv_path,
    n_components=50,
    random_seed=42,
    progress_callback=None,
) -> dict[str, str]:
    """Fit PCA/scalers only on training data and transform the held-out CSV."""
    return run_train_test_ldse_extraction(
        output_dir,
        train_csv_path,
        test_csv_path,
        n_components=n_components,
        random_seed=random_seed,
        progress_callback=progress_callback,
    )


def execute_ldse_single_extraction(
    output_dir,
    csv_path,
    mode,
    dataset_role=None,
    model_path=None,
    n_components=50,
    random_seed=42,
    progress_callback=None,
) -> str:
    """Explicitly fit training data or transform with a verified training artifact."""
    if mode == "fit_transform" and dataset_role != "train":
        raise ValueError("mode='fit_transform' requires dataset_role='train'")
    if mode == "transform" and not model_path:
        raise ValueError("mode='transform' requires model_path")

    normalized_output_dir = os.path.abspath(os.path.normpath(output_dir))
    fixed_output_paths = (
        os.path.join(normalized_output_dir, "ldse_features_summary.csv"),
        os.path.join(normalized_output_dir, "ldse_features_summary.pkl"),
    )
    for fixed_output_path in fixed_output_paths:
        if os.path.exists(fixed_output_path):
            raise FileExistsError(
                f"LDSE single output already exists: {fixed_output_path}"
            )

    extractor = OAIZIB_Light_LDSE_FeatureExtractor(
        normalized_output_dir, n_components=n_components, random_seed=random_seed
    )
    extractor.process_from_csv(
        csv_path,
        mode=mode,
        dataset_role=dataset_role,
        model_path=model_path,
        progress_callback=progress_callback,
    )
    summary_path = os.path.join(normalized_output_dir, "ldse_features_summary.csv")
    if not os.path.isfile(summary_path):
        raise RuntimeError(f"LDSE extraction did not produce a summary CSV: {summary_path}")
    return summary_path
