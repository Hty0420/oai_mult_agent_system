"""Leakage-safe LDSE shape feature extraction."""

import hashlib
import io
import json
import os
import pickle
import re
import uuid
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..klgrade_policy import LabeledKLGradeRows, select_labeled_klgrade_rows


MODEL_ARTIFACT_VERSION = 2
PCA_SOLVER = "full"
MAX_MODEL_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_MODEL_ARRAY_BYTES = 64 * 1024 * 1024
MAX_MODEL_METADATA_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _LDSECSVSnapshot:
    path: str
    payload: bytes
    sha256: str
    dataframe: pd.DataFrame


def stable_mask_array_sample_key(mask_array):
    """Return a path-independent identity for an in-memory augmented mask."""
    contiguous = np.ascontiguousarray(mask_array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return "mask-sha256:" + digest.hexdigest()


class OAIZIB_Light_LDSE_FeatureExtractor:
    """Extract LDSE features with scaler/PCA models fitted only on train data."""

    def __init__(self, output_dir, n_components=50, random_seed=42):
        self.output_dir = output_dir or "extracted_LDSE_features"
        os.makedirs(self.output_dir, exist_ok=True)
        self.structures = {
            1: "Femur",
            2: "Femoral_Cartilage",
            3: "Tibia",
            4: "Medial_Tibial_Cartilage",
            5: "Lateral_Tibial_Cartilage",
        }
        self.n_components = n_components
        self.random_seed = random_seed
        self.pca_models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_path = None
        self.train_csv_path = None
        self.train_csv_sha256 = None
        self.last_transform_counts = {}
        self._csv_snapshots = {}

    def load_image(self, path):
        """Load an image or mask, returning None when SimpleITK cannot read it."""
        try:
            return sitk.ReadImage(path)
        except Exception:
            return None

    @staticmethod
    def _normalize_mask_identity(mask_path):
        return os.path.normcase(os.path.abspath(os.path.normpath(str(mask_path))))

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _capture_csv_snapshot(csv_path):
        if not csv_path or not os.path.isfile(csv_path):
            raise FileNotFoundError("LDSE input CSV does not exist")
        normalized_path = os.path.abspath(csv_path)
        with open(normalized_path, "rb") as source:
            payload = source.read()
        dataframe = pd.read_csv(io.BytesIO(payload), dtype={"CaseID": str})
        missing_columns = sorted({"CaseID", "MaskPath"}.difference(dataframe.columns))
        if missing_columns:
            raise ValueError(
                "LDSE input CSV is missing required columns: " + ", ".join(missing_columns)
            )
        return _LDSECSVSnapshot(
            path=normalized_path,
            payload=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            dataframe=dataframe,
        )

    @staticmethod
    def _read_csv_with_required_columns(csv_path):
        return OAIZIB_Light_LDSE_FeatureExtractor._capture_csv_snapshot(
            csv_path
        ).dataframe

    def _snapshot_for_csv(self, csv_path):
        normalized_path = os.path.abspath(csv_path)
        snapshot = self._csv_snapshots.get(normalized_path)
        if snapshot is not None:
            return snapshot
        return self._capture_csv_snapshot(normalized_path)

    def _child_rng(self, sample_key, label):
        normalized_key = self._normalize_mask_identity(sample_key or "")
        material = f"{self.random_seed}\0{normalized_key}\0{label}".encode("utf-8")
        child_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return np.random.default_rng(child_seed)

    def get_boundary_points(self, mask_array, label, num_points=1000, sample_key=None):
        """Extract and deterministically subsample boundary voxels for one structure."""
        binary = (mask_array == label).astype(np.uint8)
        if np.sum(binary) == 0:
            return None
        dilated = ndimage.binary_dilation(binary, iterations=2)
        boundary = dilated - binary
        boundary_points = np.argwhere(boundary > 0)
        if len(boundary_points) > num_points:
            indices = self._child_rng(sample_key, label).choice(
                len(boundary_points), num_points, replace=False
            )
            return boundary_points[indices]
        return boundary_points

    def _feature_columns_from_models(self, pca_models, structures=None):
        structures = structures if structures is not None else self.structures
        columns = []
        for sid, structure_name in structures.items():
            if sid not in pca_models:
                continue
            columns.extend(
                f"LDSE_{structure_name}_PC{component_index}"
                for component_index in range(1, int(pca_models[sid].n_components_) + 1)
            )
        return columns

    def _write_new_model_artifact(self, metadata, arrays):
        """Serialize to a private temporary file and atomically publish one NPZ."""
        os.makedirs(self.output_dir, exist_ok=True)
        artifact_id = uuid.uuid4().hex
        model_path = os.path.abspath(
            os.path.join(self.output_dir, f"light_ldse_models_{artifact_id}.npz")
        )
        temporary_path = os.path.join(
            self.output_dir, f".light_ldse_models_{artifact_id}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(temporary_path, "xb") as artifact_file:
                np.savez_compressed(
                    artifact_file,
                    metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
                    **arrays,
                )
                artifact_file.flush()
                os.fsync(artifact_file.fileno())
            os.replace(temporary_path, model_path)
            return model_path
        except Exception:
            try:
                if os.path.isfile(temporary_path):
                    os.remove(temporary_path)
            except OSError:
                pass
            raise

    def fit_shape_models(self, train_csv_path, progress_callback=None):
        """Fit models from training CSV data and return the saved artifact path."""
        snapshot = self._snapshot_for_csv(train_csv_path)
        selection = select_labeled_klgrade_rows(
            snapshot.dataframe,
            dataset_role="LDSE training input",
        )
        dataframe = selection.frame
        normalized_csv_path = snapshot.path
        structure_points = {sid: [] for sid in self.structures}

        total_cases = int(len(dataframe))
        valid_cases = 0

        def report_fit(processed, case_id=None, status=None):
            if progress_callback is None:
                return
            try:
                progress_callback(
                    {
                        "phase": "fit",
                        "processed_cases": int(processed),
                        "total_cases": total_cases,
                        "successful_cases": int(valid_cases),
                        "failed_cases": int(processed - valid_cases),
                        "case_id": str(case_id) if case_id is not None else None,
                        "case_status": status,
                    }
                )
            except Exception:
                pass

        report_fit(0)
        for processed, raw_mask_path in enumerate(
            tqdm(dataframe["MaskPath"].tolist(), desc="Collecting shape data"),
            start=1,
        ):
            mask_path = self._normalize_mask_identity(raw_mask_path)
            if not (
                os.path.isfile(mask_path)
                and mask_path.lower().endswith((".nii", ".nii.gz"))
            ):
                report_fit(processed, mask_path, "missing_or_invalid_mask")
                continue
            mask = self.load_image(mask_path)
            if mask is None:
                report_fit(processed, mask_path, "load_failed")
                continue
            mask_array = sitk.GetArrayFromImage(mask)
            case_has_points = False
            for sid in self.structures:
                points = self.get_boundary_points(mask_array, sid, sample_key=mask_path)
                if points is not None and len(points) >= 100:
                    structure_points[sid].append(points.ravel())
                    case_has_points = True
            if case_has_points:
                valid_cases += 1
            report_fit(
                processed,
                mask_path,
                "success" if case_has_points else "no_valid_structure",
            )

        fitted_pca_models = {}
        fitted_scalers = {}
        actual_n_components = {}
        for sid in self.structures:
            points_list = structure_points[sid]
            if len(points_list) < 5:
                continue
            scaler = StandardScaler()
            scaled_points = scaler.fit_transform(points_list)
            actual_n = min(self.n_components, len(points_list), scaled_points.shape[1])
            pca = PCA(n_components=actual_n, svd_solver=PCA_SOLVER)
            pca.fit(scaled_points)
            fitted_pca_models[sid] = pca
            fitted_scalers[sid] = scaler
            actual_n_components[sid] = actual_n

        if not fitted_pca_models:
            raise RuntimeError("No structure has enough valid training samples to fit LDSE models")

        feature_columns = self._feature_columns_from_models(fitted_pca_models)
        train_csv_sha256 = snapshot.sha256
        metadata = {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "fitted_dataset_role": "train",
            "train_csv_path": normalized_csv_path,
            "train_csv_sha256": train_csv_sha256,
            "train_input_rows": selection.input_rows,
            "train_labeled_rows": selection.labeled_rows,
            "train_excluded_unlabeled_rows": selection.excluded_unlabeled_rows,
            "n_components_requested": int(self.n_components),
            "n_components_actual": {
                str(sid): int(count) for sid, count in actual_n_components.items()
            },
            "random_seed": int(self.random_seed),
            "pca_solver": PCA_SOLVER,
            "feature_columns": feature_columns,
            "structures": {str(sid): name for sid, name in self.structures.items()},
        }
        arrays = {}
        for sid in fitted_pca_models:
            scaler = fitted_scalers[sid]
            pca = fitted_pca_models[sid]
            arrays[f"scaler_mean_{sid}"] = np.asarray(scaler.mean_, dtype=np.float64)
            arrays[f"scaler_scale_{sid}"] = np.asarray(scaler.scale_, dtype=np.float64)
            arrays[f"scaler_var_{sid}"] = np.asarray(scaler.var_, dtype=np.float64)
            arrays[f"pca_components_{sid}"] = np.asarray(pca.components_, dtype=np.float64)
            arrays[f"pca_mean_{sid}"] = np.asarray(pca.mean_, dtype=np.float64)
        model_path = self._write_new_model_artifact(metadata, arrays)

        self.pca_models = fitted_pca_models
        self.scalers = fitted_scalers
        self.feature_columns = feature_columns
        self.model_path = model_path
        self.train_csv_path = normalized_csv_path
        self.train_csv_sha256 = train_csv_sha256
        return model_path

    def load_shape_models(self, model_path):
        """Load a validated numeric NPZ and reconstruct only known sklearn classes."""
        if not model_path or not os.path.isfile(model_path):
            raise FileNotFoundError("LDSE model artifact does not exist")
        model_path = os.path.abspath(model_path)
        if not model_path.lower().endswith(".npz"):
            raise ValueError(
                "Legacy or unsupported LDSE model artifact; a version-2 NPZ is required"
            )
        if os.path.getsize(model_path) > MAX_MODEL_ARTIFACT_BYTES:
            raise ValueError("LDSE model NPZ exceeds the safe artifact size limit")
        try:
            with zipfile.ZipFile(model_path) as archive_file:
                members = archive_file.infolist()
                if not members or any(member.is_dir() for member in members):
                    raise ValueError("LDSE model NPZ has an invalid archive layout")
                if any(member.file_size > MAX_MODEL_ARRAY_BYTES for member in members):
                    raise ValueError("LDSE model NPZ contains an oversized array")
                if sum(member.file_size for member in members) > MAX_MODEL_ARTIFACT_BYTES:
                    raise ValueError("LDSE model NPZ exceeds the safe expanded size limit")
            with np.load(model_path, allow_pickle=False) as archive:
                loaded_arrays = {key: archive[key].copy() for key in archive.files}
        except Exception as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("LDSE model NPZ"):
                raise
            raise ValueError(
                "LDSE model NPZ is corrupt, contains object arrays, or is unreadable"
            ) from exc

        metadata_array = loaded_arrays.pop("metadata", None)
        if (
            metadata_array is None
            or metadata_array.size != 1
            or metadata_array.dtype.kind not in {"U", "S"}
            or metadata_array.nbytes > MAX_MODEL_METADATA_BYTES
        ):
            raise ValueError("LDSE model NPZ metadata must be one bounded string array")
        metadata_text = metadata_array.item()
        if isinstance(metadata_text, bytes):
            try:
                metadata_text = metadata_text.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError("LDSE model NPZ metadata is not valid UTF-8") from exc
        try:
            artifact = json.loads(metadata_text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("LDSE model NPZ metadata is not valid JSON") from exc

        required_metadata = {
            "artifact_version",
            "fitted_dataset_role",
            "train_csv_path",
            "train_csv_sha256",
            "n_components_requested",
            "n_components_actual",
            "random_seed",
            "pca_solver",
            "feature_columns",
            "structures",
        }
        if not isinstance(artifact, dict) or not required_metadata.issubset(artifact):
            raise ValueError("LDSE model artifact is unverified or missing required metadata")
        if artifact["artifact_version"] != MODEL_ARTIFACT_VERSION:
            raise ValueError("Unsupported LDSE model artifact_version")
        if artifact["fitted_dataset_role"] != "train":
            raise ValueError("LDSE model artifact was not fitted on the train dataset")
        if artifact["pca_solver"] != PCA_SOLVER:
            raise ValueError("LDSE model artifact has unsupported PCA solver metadata")

        requested_components = artifact["n_components_requested"]
        actual_n_components = artifact["n_components_actual"]
        structures_metadata = artifact["structures"]
        if (
            not isinstance(requested_components, int)
            or isinstance(requested_components, bool)
            or requested_components <= 0
            or not isinstance(actual_n_components, dict)
            or not actual_n_components
            or not isinstance(structures_metadata, dict)
            or not structures_metadata
        ):
            raise ValueError("LDSE model artifact has mismatched PCA/scaler structure keys")
        if not all(
            isinstance(key, str)
            and key.isdigit()
            and isinstance(value, str)
            and value
            for key, value in structures_metadata.items()
        ):
            raise ValueError("LDSE model artifact has invalid structures metadata")
        if not set(actual_n_components).issubset(structures_metadata):
            raise ValueError("LDSE model artifact has mismatched PCA/scaler structure keys")
        if not all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and 0 < value <= requested_components
            for value in actual_n_components.values()
        ):
            raise ValueError("LDSE model artifact has invalid component counts")

        training_csv_path = artifact["train_csv_path"]
        training_csv_sha256 = artifact["train_csv_sha256"]
        if not isinstance(training_csv_path, str) or not os.path.isabs(training_csv_path):
            raise ValueError("LDSE model artifact training CSV path must be absolute")
        if not isinstance(training_csv_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", training_csv_sha256
        ):
            raise ValueError("LDSE model artifact has an invalid training CSV SHA-256 digest")
        if os.path.isfile(training_csv_path) and self._file_sha256(training_csv_path) != training_csv_sha256:
            raise ValueError("LDSE model artifact training CSV SHA-256 does not match the referenced file")
        if not isinstance(artifact["random_seed"], (int, np.integer)):
            raise ValueError("LDSE model artifact has invalid deterministic sampling metadata")
        if isinstance(artifact["random_seed"], bool):
            raise ValueError("LDSE model artifact has invalid deterministic sampling metadata")

        expected_array_keys = {
            f"{prefix}_{sid}"
            for sid in actual_n_components
            for prefix in (
                "scaler_mean", "scaler_scale", "scaler_var",
                "pca_components", "pca_mean",
            )
        }
        if set(loaded_arrays) != expected_array_keys:
            raise ValueError("LDSE model NPZ is missing required arrays or has unknown arrays")

        pca_models = {}
        scalers = {}
        for sid_text, component_count in actual_n_components.items():
            numeric = {}
            for prefix in (
                "scaler_mean", "scaler_scale", "scaler_var",
                "pca_components", "pca_mean",
            ):
                array = loaded_arrays[f"{prefix}_{sid_text}"]
                if array.dtype.kind != "f" or not np.isfinite(array).all():
                    raise ValueError("LDSE model estimator arrays must be finite numeric floats")
                numeric[prefix] = array

            scaler_mean = numeric["scaler_mean"]
            scaler_scale = numeric["scaler_scale"]
            scaler_var = numeric["scaler_var"]
            pca_components = numeric["pca_components"]
            pca_mean = numeric["pca_mean"]
            feature_dimension = scaler_mean.size
            if (
                scaler_mean.ndim != 1
                or feature_dimension <= 0
                or scaler_scale.shape != (feature_dimension,)
                or scaler_var.shape != (feature_dimension,)
                or pca_mean.shape != (feature_dimension,)
            ):
                raise ValueError("LDSE model scaler/PCA vectors have incompatible sizes")
            if np.any(scaler_scale <= 0):
                raise ValueError("LDSE model scaler scale values must be positive and nonzero")
            if np.any(scaler_var < 0):
                raise ValueError("LDSE model scaler variance values must be nonnegative")
            if pca_components.shape != (component_count, feature_dimension):
                raise ValueError("LDSE model PCA component matrix shape is incompatible")

            scaler = StandardScaler()
            scaler.mean_ = scaler_mean
            scaler.scale_ = scaler_scale
            scaler.var_ = scaler_var
            scaler.n_features_in_ = feature_dimension
            pca = PCA(n_components=component_count, svd_solver=PCA_SOLVER)
            pca.components_ = pca_components
            pca.mean_ = pca_mean
            pca.n_components_ = component_count
            pca.n_features_in_ = feature_dimension
            sid = int(sid_text)
            scalers[sid] = scaler
            pca_models[sid] = pca

        structures = {int(key): value for key, value in structures_metadata.items()}
        expected_columns = self._feature_columns_from_models(pca_models, structures)
        if (
            not isinstance(artifact["feature_columns"], list)
            or not all(isinstance(column, str) for column in artifact["feature_columns"])
        ):
            raise ValueError("LDSE model artifact feature columns must be strings")
        if artifact["feature_columns"] != expected_columns:
            raise ValueError("LDSE model artifact feature columns are inconsistent with PCA models")

        self.pca_models = pca_models
        self.scalers = scalers
        self.structures = structures
        self.feature_columns = artifact["feature_columns"]
        self.random_seed = int(artifact["random_seed"])
        self.n_components = int(requested_components)
        self.model_path = model_path
        self.train_csv_path = training_csv_path
        self.train_csv_sha256 = training_csv_sha256

    def extract_ldse_features(self, mask_path, sample_key=None):
        """Extract LDSE features for one mask using already-fitted models."""
        mask = self.load_image(mask_path)
        if mask is None:
            return None
        mask_array = sitk.GetArrayFromImage(mask)
        features = {}
        for sid, structure_name in self.structures.items():
            if sid not in self.pca_models:
                continue
            points = self.get_boundary_points(
                mask_array, sid, num_points=1000, sample_key=sample_key or mask_path
            )
            if points is None or len(points) < 100:
                continue
            scaled_points = self.scalers[sid].transform(points.ravel().reshape(1, -1))
            # Only the parameters required for projection are persisted in the
            # safe numeric NPZ artifact.  Newer scikit-learn releases access
            # additional fitted attributes (for example ``explained_variance_``)
            # even when PCA whitening is disabled, which makes reconstructed
            # estimators fail despite the projection itself needing only the
            # fitted mean and components.  Compute that projection explicitly
            # so version-compatible artifacts remain minimal and pickle-free.
            pca = self.pca_models[sid]
            ldse = ((scaled_points - pca.mean_) @ pca.components_.T)[0]
            for index, value in enumerate(ldse, start=1):
                features[f"LDSE_{structure_name}_PC{index}"] = value
        return features

    def transform_from_csv(
        self,
        csv_path,
        output_dir=None,
        save=True,
        progress_callback=None,
        progress_phase="transform",
    ):
        """Transform only with models already fitted or explicitly loaded."""
        if not self.pca_models or not self.scalers or not self.feature_columns:
            raise RuntimeError("LDSE models must be fitted or loaded before transforming")
        snapshot = self._snapshot_for_csv(csv_path)
        selection = select_labeled_klgrade_rows(
            snapshot.dataframe,
            dataset_role="LDSE transform input",
        )
        dataframe = selection.frame
        all_features = []
        total_cases = int(len(dataframe))

        def report_transform(processed, case_id=None, status=None):
            if progress_callback is None:
                return
            try:
                progress_callback(
                    {
                        "phase": progress_phase,
                        "processed_cases": int(processed),
                        "total_cases": total_cases,
                        "successful_cases": int(len(all_features)),
                        "failed_cases": int(processed - len(all_features)),
                        "case_id": str(case_id) if case_id is not None else None,
                        "case_status": status,
                    }
                )
            except Exception:
                pass

        report_transform(0)
        for processed, (_, row) in enumerate(
            tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing CSV entries"),
            start=1,
        ):
            mask_path = self._normalize_mask_identity(row["MaskPath"])
            if not os.path.isfile(mask_path):
                report_transform(processed, row["CaseID"], "missing_mask")
                continue
            ldse_features = self.extract_ldse_features(mask_path, sample_key=mask_path)
            if ldse_features is None:
                report_transform(processed, row["CaseID"], "no_features")
                continue
            case_features = {"CaseID": row["CaseID"]}
            if ldse_features:
                case_features.update(ldse_features)
            if "KLGrade" in dataframe.columns:
                case_features["KLGrade"] = row["KLGrade"]
            all_features.append(case_features)
            report_transform(processed, row["CaseID"], "success")

        output_columns = ["CaseID"] + self.feature_columns
        if "KLGrade" in dataframe.columns:
            output_columns.append("KLGrade")
        features_dataframe = pd.DataFrame(all_features).reindex(columns=output_columns)
        if save:
            target_dir = output_dir or self.output_dir
            os.makedirs(target_dir, exist_ok=True)
            features_dataframe.to_csv(
                os.path.join(target_dir, "ldse_features_summary.csv"), index=False
            )
            with open(os.path.join(target_dir, "ldse_features_summary.pkl"), "wb") as output:
                pickle.dump(features_dataframe, output)
        self.last_transform_counts = {
            "input_rows": selection.input_rows,
            "labeled_rows": selection.labeled_rows,
            "excluded_unlabeled_rows": selection.excluded_unlabeled_rows,
            "output_rows": int(len(features_dataframe)),
            "other_skipped_rows": int(selection.labeled_rows - len(features_dataframe)),
        }
        return features_dataframe

    def process_from_csv(
        self,
        csv_path,
        save=True,
        *,
        mode,
        dataset_role=None,
        model_path=None,
        progress_callback=None,
    ):
        """Explicitly fit on training data or transform with an explicit artifact."""
        if mode not in {"fit_transform", "transform"}:
            raise ValueError("mode must be 'fit_transform' or 'transform'")
        if mode == "fit_transform":
            if dataset_role != "train":
                raise ValueError("mode='fit_transform' requires dataset_role='train'")
            snapshot = self._capture_csv_snapshot(csv_path)
            previous_snapshots = self._csv_snapshots
            self._csv_snapshots = {snapshot.path: snapshot}
            try:
                self.fit_shape_models(csv_path, progress_callback=progress_callback)
                return self.transform_from_csv(
                    csv_path,
                    save=save,
                    progress_callback=progress_callback,
                    progress_phase="train_transform",
                )
            finally:
                self._csv_snapshots = previous_snapshots
        if not model_path:
            raise ValueError("mode='transform' requires model_path")
        self.load_shape_models(model_path)
        return self.transform_from_csv(
            csv_path,
            save=save,
            progress_callback=progress_callback,
            progress_phase="test_transform" if dataset_role == "test" else "transform",
        )


def _read_paired_ldse_input(
    csv_path, dataset_role
) -> tuple[_LDSECSVSnapshot, LabeledKLGradeRows]:
    """Read a paired-workflow input after enforcing its classification schema."""
    snapshot = OAIZIB_Light_LDSE_FeatureExtractor._capture_csv_snapshot(csv_path)
    selection = select_labeled_klgrade_rows(
        snapshot.dataframe,
        dataset_role=f"LDSE {dataset_role} input",
    )
    return snapshot, selection


def _paired_ldse_paths(output_dir):
    """Return the fixed, absolute artifact locations for one paired LDSE run."""
    output_root = os.path.abspath(output_dir)
    model_dir = os.path.join(output_root, "model")
    train_dir = os.path.join(output_root, "train")
    test_dir = os.path.join(output_root, "test")
    return {
        "output_root": output_root,
        "model_dir": model_dir,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_features_path": os.path.join(train_dir, "ldse_features_summary.csv"),
        "train_features_pkl_path": os.path.join(train_dir, "ldse_features_summary.pkl"),
        "test_features_path": os.path.join(test_dir, "ldse_features_summary.csv"),
        "test_features_pkl_path": os.path.join(test_dir, "ldse_features_summary.pkl"),
        "manifest_path": os.path.join(output_root, "ldse_run_manifest.json"),
    }


def _ensure_new_paired_ldse_output(paths):
    """Reject all reused paired targets before model fitting can start."""
    fixed_targets = (
        paths["train_features_path"],
        paths["train_features_pkl_path"],
        paths["test_features_path"],
        paths["test_features_pkl_path"],
        paths["manifest_path"],
    )
    for target_path in fixed_targets:
        if os.path.exists(target_path):
            raise FileExistsError("LDSE paired output target already exists")

    if os.path.isdir(paths["model_dir"]):
        model_artifacts = [
            name
            for name in os.listdir(paths["model_dir"])
            if name.startswith("light_ldse_models_") and name.endswith(".npz")
        ]
        if model_artifacts:
            raise FileExistsError(
                "LDSE paired model directory already contains an LDSE model artifact"
            )


def _validate_paired_feature_schema(dataframe, expected_columns, dataset_role):
    """Ensure transformed output exactly follows the train-fitted feature schema."""
    actual_columns = list(dataframe.columns)
    missing_columns = [column for column in expected_columns if column not in actual_columns]
    extra_columns = [column for column in actual_columns if column not in expected_columns]
    if missing_columns or extra_columns:
        details = []
        if missing_columns:
            details.append("missing columns: " + ", ".join(missing_columns))
        if extra_columns:
            details.append("extra columns: " + ", ".join(extra_columns))
        raise ValueError(
            f"LDSE {dataset_role} feature schema differs from the fitted training schema: "
            + "; ".join(details)
        )
    if actual_columns != expected_columns:
        raise ValueError(
            f"LDSE {dataset_role} feature schema differs from the fitted training schema: "
            "missing columns: none; extra columns: none; column order differs"
        )


def _create_paired_ldse_directory(path, created_directories):
    """Create a directory and remember only the path components new to this run."""
    missing_directories = []
    candidate = path
    while not os.path.exists(candidate):
        missing_directories.append(candidate)
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
    os.makedirs(path, exist_ok=True)
    created_directories.extend(reversed(missing_directories))


def _write_new_paired_feature_files(dataframe, csv_path, pkl_path, created_paths):
    """Write one paired feature table without replacing an existing target."""
    with open(csv_path, "x", encoding="utf-8", newline="") as csv_file:
        created_paths.append(csv_path)
        dataframe.to_csv(csv_file, index=False)
    with open(pkl_path, "xb") as pickle_file:
        created_paths.append(pkl_path)
        pickle.dump(dataframe, pickle_file)


def _cleanup_paired_ldse_run(created_paths, created_directories):
    """Remove only artifacts and empty directories created by the failed invocation."""
    for path in reversed(created_paths):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass
    for directory in reversed(created_directories):
        try:
            os.rmdir(directory)
        except OSError:
            pass


def _verify_paired_csv_hash(csv_path, expected_hash, dataset_role):
    """Abort a paired transaction when an input changes during processing."""
    current_hash = OAIZIB_Light_LDSE_FeatureExtractor._file_sha256(csv_path)
    if current_hash != expected_hash:
        raise ValueError(f"LDSE {dataset_role} CSV changed during paired run")


def run_train_test_ldse_extraction(
    output_dir,
    train_csv_path,
    test_csv_path,
    n_components=50,
    random_seed=42,
    progress_callback=None,
):
    """Fit LDSE once on train and transform an identically-shaped test output."""
    train_snapshot, train_selection = _read_paired_ldse_input(train_csv_path, "train")
    test_snapshot, test_selection = _read_paired_ldse_input(test_csv_path, "test")
    train_csv_path = train_snapshot.path
    test_csv_path = test_snapshot.path
    train_csv_sha256 = train_snapshot.sha256
    test_csv_sha256 = test_snapshot.sha256
    paths = _paired_ldse_paths(output_dir)
    _ensure_new_paired_ldse_output(paths)
    created_paths = []
    created_directories = []
    try:
        _create_paired_ldse_directory(paths["model_dir"], created_directories)
        _create_paired_ldse_directory(paths["train_dir"], created_directories)
        _create_paired_ldse_directory(paths["test_dir"], created_directories)
        extractor = OAIZIB_Light_LDSE_FeatureExtractor(
            paths["model_dir"], n_components=n_components, random_seed=random_seed
        )
        extractor._csv_snapshots = {
            train_snapshot.path: train_snapshot,
            test_snapshot.path: test_snapshot,
        }
        model_path = os.path.abspath(
            extractor.fit_shape_models(
                train_csv_path,
                progress_callback=progress_callback,
            )
        )
        created_paths.append(model_path)
        _verify_paired_csv_hash(train_csv_path, train_csv_sha256, "train")
        if extractor.train_csv_sha256 != train_csv_sha256:
            raise ValueError(
                "LDSE train model artifact hash does not match the paired-run input hash"
            )
        train_features = extractor.transform_from_csv(
            train_csv_path,
            save=False,
            progress_callback=progress_callback,
            progress_phase="train_transform",
        )
        test_features = extractor.transform_from_csv(
            test_csv_path,
            save=False,
            progress_callback=progress_callback,
            progress_phase="test_transform",
        )
        _verify_paired_csv_hash(train_csv_path, train_csv_sha256, "train")
        _verify_paired_csv_hash(test_csv_path, test_csv_sha256, "test")

        expected_columns = ["CaseID"] + list(extractor.feature_columns) + ["KLGrade"]
        _validate_paired_feature_schema(train_features, expected_columns, "train")
        _validate_paired_feature_schema(test_features, expected_columns, "test")

        actual_n_components = {
            str(structure_id): int(model.n_components_)
            for structure_id, model in extractor.pca_models.items()
        }
        manifest = {
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "fitted_dataset_role": "train",
            "train_csv_path": train_csv_path,
            "test_csv_path": test_csv_path,
            "train_features_path": os.path.abspath(paths["train_features_path"]),
            "test_features_path": os.path.abspath(paths["test_features_path"]),
            "model_path": model_path,
            "train_csv_sha256": train_csv_sha256,
            "test_csv_sha256": test_csv_sha256,
            "random_seed": int(random_seed),
            "n_components_requested": int(n_components),
            "n_components_actual": actual_n_components,
            "feature_columns": list(extractor.feature_columns),
            "feature_columns_match": True,
            "train_input_rows": train_selection.input_rows,
            "train_labeled_rows": train_selection.labeled_rows,
            "train_excluded_unlabeled_rows": train_selection.excluded_unlabeled_rows,
            "train_output_rows": int(len(train_features)),
            "train_skipped_rows": int(train_selection.input_rows - len(train_features)),
            "train_other_skipped_rows": int(
                train_selection.labeled_rows - len(train_features)
            ),
            "test_input_rows": test_selection.input_rows,
            "test_labeled_rows": test_selection.labeled_rows,
            "test_excluded_unlabeled_rows": test_selection.excluded_unlabeled_rows,
            "test_output_rows": int(len(test_features)),
            "test_skipped_rows": int(test_selection.input_rows - len(test_features)),
            "test_other_skipped_rows": int(
                test_selection.labeled_rows - len(test_features)
            ),
        }
        _write_new_paired_feature_files(
            train_features,
            paths["train_features_path"],
            paths["train_features_pkl_path"],
            created_paths,
        )
        _write_new_paired_feature_files(
            test_features,
            paths["test_features_path"],
            paths["test_features_pkl_path"],
            created_paths,
        )
        with open(paths["manifest_path"], "x", encoding="utf-8") as manifest_file:
            created_paths.append(paths["manifest_path"])
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)

        return {
            "train_features_path": os.path.abspath(paths["train_features_path"]),
            "test_features_path": os.path.abspath(paths["test_features_path"]),
            "model_path": model_path,
            "manifest_path": os.path.abspath(paths["manifest_path"]),
        }
    except Exception:
        _cleanup_paired_ldse_run(created_paths, created_directories)
        raise


def koa_lasso_classification(
    train_path=r"D:\\oai_mcp\\datasets\\ldse_extracted_features\\train_ldse_features_summary.csv",
    test_path=r"D:\\oai_mcp\\datasets\\ldse_extracted_features\\test_ldse_features_summary.csv",
    output_dir=r"D:\\oai_mcp\\KOA_LASSO_Results\\ldse",
    by_region=False,
):
    """Run the existing LDSE LASSO classifier without importing it for extraction."""
    from algorithm.feature_class_algorithm.koa_lasso_classifier import KOALassoClassifier

    classifier = KOALassoClassifier(
        train_path=train_path,
        test_path=test_path,
        output_dir=output_dir,
        by_region=by_region,
    )
    classifier.run()
    return output_dir
