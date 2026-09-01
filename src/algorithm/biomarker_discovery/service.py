"""File-oriented service for cross-sectional MRI candidate biomarker discovery.

The service is deterministic and research-only.  Discovery is restricted to the
training table; the frozen test table is used only to confirm a candidate set
whose identifiers and preprocessing parameters have already been hashed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import scipy

from .statistics import (
    adjust_bh,
    adjust_holm,
    feature_association,
    fit_ordered_logit,
)


SERVICE_SCHEMA_VERSION = "1.0"
SERVICE_VERSION = "1.1.0"
DEFAULT_COVARIATES = ("Age", "Gender", "BMI", "KneeSide")
DEFAULT_ENDPOINT = "KLGrade"
DEFAULT_ID_COLUMN = "CaseID"
MAX_FEATURES = 5000

METHOD_REFERENCES = [
    {
        "source_id": "SRC-MCCULLAGH-1980",
        "title": "Regression Models for Ordinal Data",
        "doi": "10.1111/j.2517-6161.1980.tb01109.x",
        "url": "https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1980.tb01109.x",
        "role": "proportional-odds ordinal regression",
    },
    {
        "source_id": "SRC-BH-1995",
        "title": "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing",
        "doi": "10.1111/j.2517-6161.1995.tb02031.x",
        "url": "https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x",
        "role": "Benjamini-Hochberg false-discovery-rate control",
    },
    {
        "source_id": "SRC-IBSI-2020",
        "title": "The Image Biomarker Standardization Initiative",
        "doi": "10.1148/radiol.2020191145",
        "pmid": "32154773",
        "url": "https://pubmed.ncbi.nlm.nih.gov/32154773/",
        "role": "quantitative imaging feature standardisation",
    },
    {
        "source_id": "SRC-CLEAR-2023",
        "title": "CheckList for EvaluAtion of Radiomics research (CLEAR)",
        "doi": "10.1186/s13244-023-01415-8",
        "pmid": "37142815",
        "url": "https://pubmed.ncbi.nlm.nih.gov/37142815/",
        "role": "transparent and reproducible radiomics reporting",
    },
    {
        "source_id": "SRC-ECKSTEIN-2015",
        "title": "Cartilage Thickness Change as an Imaging Biomarker of Knee Osteoarthritis Progression",
        "doi": "10.1002/art.39324",
        "pmid": "26316262",
        "url": "https://pubmed.ncbi.nlm.nih.gov/26316262/",
        "role": "knee-cartilage imaging-biomarker context",
    },
    {
        "source_id": "SRC-WIRTH-2008",
        "title": "A technique for regional analysis of femorotibial cartilage thickness based on quantitative magnetic resonance imaging",
        "doi": "10.1109/TMI.2007.907323",
        "pmid": "18541481",
        "url": "https://pubmed.ncbi.nlm.nih.gov/18541481/",
        "role": "anatomically based femorotibial cartilage subregions",
    },
    {
        "source_id": "SRC-FROBELL-2010",
        "title": "Femorotibial subchondral bone area and regional cartilage thickness across radiographic stages of osteoarthritis",
        "doi": "10.1002/acr.20262",
        "pmid": "20496431",
        "url": "https://pubmed.ncbi.nlm.nih.gov/20496431/",
        "role": "cross-sectional regional cartilage-thickness and KL-stage context",
    },
]

_COLUMN_ALIASES = {
    "caseid": "CaseID",
    "case_id": "CaseID",
    "sourcecaseid": "SourceCaseID",
    "source_case_id": "SourceCaseID",
    "klgrade": "KLGrade",
    "kl_grade": "KLGrade",
    "koa_grade": "KLGrade",
    "age": "Age",
    "gender": "Gender",
    "sex": "Gender",
    "bmi": "BMI",
    "kneeside": "KneeSide",
    "knee_side": "KneeSide",
}

_NON_FEATURE_COLUMNS = {
    "CaseID", "SourceCaseID", "KLGrade", "Age", "Gender", "BMI", "KneeSide",
    "ImagePath", "MaskPath", "Region", "image_id", "patient_id", "split",
}


class BiomarkerDiscoveryError(ValueError):
    """Expected, user-actionable service error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolved_input(path: str, suffix: str) -> Path:
    candidate = Path(str(path)).expanduser().resolve()
    if not candidate.is_file() or candidate.suffix.lower() != suffix:
        raise BiomarkerDiscoveryError(f"输入文件不存在或格式不正确: {candidate}")
    if not _inside(candidate, _project_root()):
        raise BiomarkerDiscoveryError("输入文件必须位于当前项目工作区内。")
    return candidate


def _prepare_output_dir(path: str) -> Path:
    destination = Path(str(path)).expanduser().resolve()
    root = _project_root()
    if not _inside(destination, root) or destination == root:
        raise BiomarkerDiscoveryError("输出目录必须位于项目工作区内且不能是项目根目录。")
    if destination.exists() and any(destination.iterdir()):
        raise BiomarkerDiscoveryError(f"输出目录必须不存在或为空: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _resolved_output_path(path: str) -> Path:
    destination = Path(str(path)).expanduser().resolve()
    root = _project_root()
    if not _inside(destination, root) or destination == root:
        raise BiomarkerDiscoveryError("输出目录必须位于项目工作区内且不能是项目根目录。")
    return destination


def _reuse_discovery_result(
    output_dir: str,
    train_path: Path,
    test_path: Path,
) -> dict[str, Any] | None:
    """Return a complete hash-verified prior result for an idempotent retry."""
    destination = _resolved_output_path(output_dir)
    manifest_path = destination / "biomarker_manifest.json"
    if not destination.is_dir() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        manifest.get("status") != "success"
        or manifest.get("service_version") != SERVICE_VERSION
        or manifest.get("train_sha256") != _sha256_file(train_path)
        or manifest.get("test_sha256") != _sha256_file(test_path)
    ):
        return None
    artifact_paths: list[Path] = []
    for filename, expected_hash in dict(manifest.get("artifact_hashes", {})).items():
        artifact = (destination / str(filename)).resolve()
        if artifact.parent != destination or not artifact.is_file():
            return None
        if _sha256_file(artifact) != str(expected_hash):
            return None
        artifact_paths.append(artifact)
    by_name = {path.name: path for path in artifact_paths}
    return {
        "status": "success",
        "message": "复用同一运行已完成且哈希校验通过的生物标志物结果。",
        "scientific_outcome": manifest.get("scientific_outcome"),
        "output_dir": str(destination),
        "manifest_path": str(manifest_path),
        "report_path": str(by_name.get("biomarker_report.md", destination / "biomarker_report.md")),
        "candidate_manifest_path": str(by_name.get("candidate_manifest.json", destination / "candidate_manifest.json")),
        "discovery_results_path": str(by_name.get("discovery_results.csv", destination / "discovery_results.csv")),
        "confirmation_results_path": str(by_name.get("confirmation_results.csv", destination / "confirmation_results.csv")),
        "candidate_count": int(manifest.get("candidate_count", 0)),
        "confirmed_count": int(manifest.get("confirmed_count", 0)),
        "files": [str(path) for path in [*artifact_paths, manifest_path]],
        "idempotent_reuse": True,
    }


def _reuse_audit_result(
    output_dir: str,
    manifest_path: Path,
) -> dict[str, Any] | None:
    destination = _resolved_output_path(output_dir)
    audit_path = destination / "biomarker_audit.json"
    report_path = destination / "biomarker_audit.md"
    if not audit_path.is_file() or not report_path.is_file():
        return None
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        audit.get("service_version") != SERVICE_VERSION
        or audit.get("manifest_sha256") != _sha256_file(manifest_path)
    ):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    run_dir = manifest_path.parent.resolve()
    for filename, expected_hash in dict(manifest.get("artifact_hashes", {})).items():
        artifact = (run_dir / str(filename)).resolve()
        if (
            artifact.parent != run_dir
            or not artifact.is_file()
            or _sha256_file(artifact) != str(expected_hash)
        ):
            return None
    valid = audit.get("valid") is True
    return {
        "status": "success" if valid else "failed",
        "message": (
            "复用同一运行已完成的独立审计。"
            if valid
            else "复用同一运行未通过的独立审计。"
        ),
        "valid": valid,
        "audit_path": str(audit_path),
        "report_path": str(report_path),
        "candidate_count": int(audit.get("candidate_count", 0)),
        "confirmed_count": int(audit.get("confirmed_count", 0)),
        "scientific_outcome": audit.get("scientific_outcome"),
        "dataset_summary": dict(audit.get("dataset_summary", {})),
        "confirmed_candidates": list(audit.get("confirmed_candidates", [])),
        "source_report_path": audit.get("source_report_path"),
        "files": [str(audit_path), str(report_path)],
        "gates": dict(audit.get("gates", {})),
        "idempotent_reuse": True,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed: dict[str, str] = {}
    for column in frame.columns:
        clean = str(column).strip().lstrip("\ufeff")
        renamed[column] = _COLUMN_ALIASES.get(clean.lower(), clean)
    normalized = frame.rename(columns=renamed)
    if normalized.columns.duplicated().any():
        duplicates = normalized.columns[normalized.columns.duplicated()].tolist()
        raise BiomarkerDiscoveryError(f"标准化后存在重复列: {duplicates}")
    return normalized


def _parse_covariates(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return list(DEFAULT_COVARIATES)
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",")]
    else:
        raw = [str(item).strip() for item in value]
    return list(dict.fromkeys(item for item in raw if item))


def _select_labeled_rows(frame: pd.DataFrame, endpoint: str, role: str) -> tuple[pd.DataFrame, int]:
    if endpoint not in frame.columns:
        raise BiomarkerDiscoveryError(f"{role} 特征表缺少结局列 {endpoint}")
    numeric = pd.to_numeric(frame[endpoint], errors="coerce")
    invalid = numeric.notna() & ~numeric.isin([-1, 0, 1, 2, 3, 4])
    if invalid.any():
        raise BiomarkerDiscoveryError(
            f"{role} {endpoint} 包含 {int(invalid.sum())} 个 0-4/-1 之外的值。"
        )
    excluded = int((numeric.isna() | (numeric == -1)).sum())
    selected = frame.loc[numeric.notna() & (numeric != -1)].copy()
    selected[endpoint] = numeric.loc[selected.index].astype("int64")
    if len(selected) < 20 or selected[endpoint].nunique() < 3:
        raise BiomarkerDiscoveryError(
            f"{role} 有效样本或 KL 等级不足，无法进行有序关联分析。"
        )
    return selected.reset_index(drop=True), excluded


def _validate_identifiers(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_column: str,
) -> dict[str, Any]:
    if id_column not in train.columns or id_column not in test.columns:
        raise BiomarkerDiscoveryError(f"训练/测试特征表必须包含 {id_column}")
    if train[id_column].astype(str).duplicated().any():
        raise BiomarkerDiscoveryError("训练特征表不是患者/膝关节级：CaseID 存在重复。")
    if test[id_column].astype(str).duplicated().any():
        raise BiomarkerDiscoveryError("测试特征表不是患者/膝关节级：CaseID 存在重复。")
    train_ids = set(train[id_column].astype(str))
    test_ids = set(test[id_column].astype(str))
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise BiomarkerDiscoveryError(
            f"训练/测试存在 {len(overlap)} 个重叠 {id_column}，已阻断分析。"
        )
    group_column = next(
        (
            name for name in ("SourceCaseID", "OAI_SubjectID", "SubjectID", "patient_id")
            if name in train.columns and name in test.columns
        ),
        None,
    )
    group_overlap_count = 0
    if group_column:
        train_groups = set(train[group_column].dropna().astype(str))
        test_groups = set(test[group_column].dropna().astype(str))
        group_overlap_count = len(train_groups & test_groups)
        if group_overlap_count:
            raise BiomarkerDiscoveryError(
                f"训练/测试存在 {group_overlap_count} 个重叠 {group_column}，可能为同一受试者双膝。"
            )
    return {
        "id_column": id_column,
        "id_overlap_count": 0,
        "group_column": group_column,
        "group_overlap_count": group_overlap_count,
    }


def _encode_covariates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    requested: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    train_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    expanded_names: list[str] = []
    definitions: dict[str, Any] = {}
    missing: list[str] = []

    for name in requested:
        if name not in train.columns or name not in test.columns:
            missing.append(name)
            continue
        train_numeric = pd.to_numeric(train[name], errors="coerce")
        test_numeric = pd.to_numeric(test[name], errors="coerce")
        numeric_fraction = float(train_numeric.notna().mean())
        if numeric_fraction >= 0.8 and train_numeric.nunique(dropna=True) > 2:
            median = float(train_numeric.median())
            filled_train = train_numeric.fillna(median).to_numpy(dtype=np.float64)
            filled_test = test_numeric.fillna(median).to_numpy(dtype=np.float64)
            mean = float(np.mean(filled_train))
            std = float(np.std(filled_train, ddof=0))
            if not math.isfinite(std) or std <= 1e-12:
                missing.append(f"{name}:constant")
                continue
            train_columns.append((filled_train - mean) / std)
            test_columns.append((filled_test - mean) / std)
            expanded_names.append(name)
            definitions[name] = {
                "type": "numeric",
                "imputation": "training_median",
                "median": median,
                "mean": mean,
                "std": std,
            }
            continue

        train_text = train[name].astype("string").str.strip()
        test_text = test[name].astype("string").str.strip()
        observed = train_text.dropna()
        if observed.empty:
            missing.append(f"{name}:all_missing")
            continue
        mode = str(observed.mode().iloc[0])
        train_text = train_text.fillna(mode)
        test_text = test_text.fillna(mode)
        categories = sorted(str(value) for value in train_text.unique())
        if len(categories) < 2:
            missing.append(f"{name}:constant")
            continue
        reference = categories[0]
        for category in categories[1:]:
            train_columns.append((train_text == category).to_numpy(dtype=np.float64))
            test_columns.append((test_text == category).to_numpy(dtype=np.float64))
            expanded_names.append(f"{name}={category}")
        definitions[name] = {
            "type": "categorical",
            "imputation": "training_mode",
            "mode": mode,
            "categories": categories,
            "reference": reference,
            "unseen_test_values": sorted(
                set(str(value) for value in test_text.unique()) - set(categories)
            ),
        }

    train_matrix = (
        np.column_stack(train_columns)
        if train_columns else np.empty((len(train), 0), dtype=np.float64)
    )
    test_matrix = (
        np.column_stack(test_columns)
        if test_columns else np.empty((len(test), 0), dtype=np.float64)
    )
    return train_matrix, test_matrix, expanded_names, {
        "requested": requested,
        "expanded_columns": expanded_names,
        "definitions": definitions,
        "unavailable": missing,
    }


def _feature_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    endpoint: str,
    id_column: str,
    covariates: Iterable[str],
) -> tuple[list[str], list[str]]:
    excluded = set(_NON_FEATURE_COLUMNS) | {endpoint, id_column} | set(covariates)
    common = [column for column in train.columns if column in test.columns and column not in excluded]
    eligible: list[str] = []
    rejected: list[str] = []
    for column in common:
        train_numeric = pd.to_numeric(train[column], errors="coerce")
        test_numeric = pd.to_numeric(test[column], errors="coerce")
        if train_numeric.notna().mean() < 0.8 or test_numeric.notna().mean() < 0.8:
            rejected.append(f"{column}:non_numeric_or_missing")
            continue
        if train_numeric.nunique(dropna=True) < 3:
            rejected.append(f"{column}:near_constant")
            continue
        eligible.append(column)
    if not eligible:
        raise BiomarkerDiscoveryError("训练/测试表没有共同的可分析数值特征。")
    if len(eligible) > MAX_FEATURES:
        raise BiomarkerDiscoveryError(
            f"候选特征数 {len(eligible)} 超过安全上限 {MAX_FEATURES}。"
        )
    return eligible, rejected


def _feature_arrays(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    train_numeric = pd.to_numeric(train[feature], errors="coerce")
    test_numeric = pd.to_numeric(test[feature], errors="coerce")
    median = float(train_numeric.median())
    train_values = train_numeric.fillna(median).to_numpy(dtype=np.float64)
    test_values = test_numeric.fillna(median).to_numpy(dtype=np.float64)
    mean = float(np.mean(train_values))
    std = float(np.std(train_values, ddof=0))
    if not math.isfinite(std) or std <= 1e-12:
        raise BiomarkerDiscoveryError(f"特征 {feature} 在训练集中为常量。")
    return (
        (train_values - mean) / std,
        (test_values - mean) / std,
        {
            "training_median": median,
            "training_mean": mean,
            "training_std": std,
            "train_missing_fraction": float(train_numeric.isna().mean()),
            "test_missing_fraction": float(test_numeric.isna().mean()),
        },
    )


def _bootstrap_direction_support(
    y: np.ndarray,
    covariates: np.ndarray,
    feature: np.ndarray,
    reference_beta: float,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    estimates: list[float] = []
    sample_count = len(y)
    for _ in range(int(iterations)):
        indexes = rng.integers(0, sample_count, sample_count)
        sampled_y = y[indexes]
        if np.unique(sampled_y).size < 3:
            continue
        sampled_x = np.column_stack([covariates[indexes], feature[indexes]])
        try:
            fitted = fit_ordered_logit(sampled_y, sampled_x, max_iterations=300)
        except (ValueError, FloatingPointError):
            continue
        if fitted.converged and fitted.coefficients.size:
            estimates.append(float(fitted.coefficients[-1]))
    if not estimates:
        return {
            "requested_iterations": int(iterations),
            "successful_iterations": 0,
            "directional_support": 0.0,
            "bootstrap_beta_ci_low": None,
            "bootstrap_beta_ci_high": None,
        }
    values = np.asarray(estimates, dtype=np.float64)
    direction = 1.0 if reference_beta >= 0 else -1.0
    return {
        "requested_iterations": int(iterations),
        "successful_iterations": int(values.size),
        "directional_support": float(np.mean(np.sign(values) == direction)),
        "bootstrap_beta_ci_low": float(np.quantile(values, 0.025)),
        "bootstrap_beta_ci_high": float(np.quantile(values, 0.975)),
    }


def _prune_correlated_candidates(
    ranked: list[dict[str, Any]],
    feature_vectors: Mapping[str, np.ndarray],
    *,
    correlation_threshold: float,
    maximum: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    selected: list[dict[str, Any]] = []
    redundant: dict[str, str] = {}
    for candidate in ranked:
        feature = str(candidate["feature"])
        representative = None
        for retained in selected:
            other = str(retained["feature"])
            correlation = float(
                pd.Series(feature_vectors[feature]).corr(
                    pd.Series(feature_vectors[other]), method="spearman"
                )
            )
            if math.isfinite(correlation) and abs(correlation) >= correlation_threshold:
                representative = other
                break
        if representative:
            redundant[feature] = representative
            continue
        selected.append(candidate)
        if len(selected) >= maximum:
            break
    return selected, redundant


def _build_report(
    *,
    train_count: int,
    test_count: int,
    feature_count: int,
    candidate_count: int,
    confirmed_count: int,
    scientific_outcome: str,
    candidate_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# KL 0–4 横断面 MRI 候选生物标志物发现报告",
        "",
        "> 研究用途：本结果是横断面关联性候选发现，不构成临床诊断、因果机制或临床可用生物标志物认证。",
        "",
        "## 运行摘要",
        "",
        f"- 训练集有效膝关节：{train_count}",
        f"- 冻结测试集有效膝关节：{test_count}",
        f"- 接受有序关联检验的特征：{feature_count}",
        f"- 训练集冻结候选：{candidate_count}",
        f"- 测试集确认候选：{confirmed_count}",
        f"- 科学结局：`{scientific_outcome}`",
        "",
        "## 方法",
        "",
        "KL 0–4 作为有序结局，使用 proportional-odds cumulative-logit 模型。每个影像特征按训练集均值和标准差标准化，并在可用时调整 Age、Gender、BMI 和 KneeSide。训练集全特征检验采用 Benjamini–Hochberg FDR；候选集冻结后，测试集采用 Holm 校正确认，禁止依据测试结果重新筛选。",
        "",
        "## 已确认候选",
        "",
    ]
    confirmed = [row for row in candidate_rows if row.get("confirmation_status") == "confirmed"]
    if not confirmed:
        lines.append("没有候选满足预先规定的测试集确认规则。这是有效的阴性结果，不应通过放宽阈值改写。")
    else:
        lines.append("| 候选 | 训练 OR/SD | 训练 q | 测试 OR/SD | Holm P | 方向一致 |")
        lines.append("|---|---:|---:|---:|---:|:---:|")
        for row in confirmed:
            lines.append(
                f"| {row['feature']} | {row['train_odds_ratio_per_sd']:.4g} | "
                f"{row['train_q_value']:.4g} | {row['test_odds_ratio_per_sd']:.4g} | "
                f"{row['test_holm_p_value']:.4g} | 是 |"
            )
    lines.extend(["", "## 方法依据", ""])
    for reference in METHOD_REFERENCES:
        lines.append(
            f"- {reference['title']} — {reference['role']}; "
            f"DOI: {reference.get('doi', 'N/A')}; {reference['url']}"
        )
    lines.extend(
        [
            "",
            "## 主要限制",
            "",
            "- 当前确认集来自同一数据资源的预留测试集，不等同于独立中心外部验证。",
            "- 横断面 KL 关联不能证明疾病进展预测能力或生物学因果机制。",
            "- 若没有重复分割或短期重测数据，特征的读者间/重测重复性仍需另行验证。",
            "",
        ]
    )
    return "\n".join(lines)


def run_biomarker_discovery(
    train_feature_path: str,
    test_feature_path: str,
    output_dir: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    id_column: str = DEFAULT_ID_COLUMN,
    covariates: str | Iterable[str] | None = None,
    fdr_alpha: float = 0.05,
    confirmation_alpha: float = 0.05,
    correlation_threshold: float = 0.90,
    maximum_candidates: int = 20,
    bootstrap_iterations: int = 100,
    bootstrap_support_threshold: float = 0.70,
    random_seed: int = 42,
    progress_callback=None,
) -> dict[str, Any]:
    """Discover adjusted ordinal-KL associations and confirm a frozen set."""
    def report_progress(progress: float, message: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(float(progress), 100.0, str(message))
        except Exception:
            # Observability must never change a scientific result.
            pass

    report_progress(0.0, "正在校验分析参数和训练/测试输入")
    if not 0.0 < float(fdr_alpha) < 1.0:
        raise BiomarkerDiscoveryError("fdr_alpha 必须位于 0 和 1 之间。")
    if not 0.0 < float(confirmation_alpha) < 1.0:
        raise BiomarkerDiscoveryError("confirmation_alpha 必须位于 0 和 1 之间。")
    if not 0.0 < float(correlation_threshold) <= 1.0:
        raise BiomarkerDiscoveryError("correlation_threshold 必须位于 (0, 1]。")
    if not 1 <= int(maximum_candidates) <= 100:
        raise BiomarkerDiscoveryError("maximum_candidates 必须位于 1-100。")
    if not 10 <= int(bootstrap_iterations) <= 2000:
        raise BiomarkerDiscoveryError("bootstrap_iterations 必须位于 10-2000。")

    train_path = _resolved_input(train_feature_path, ".csv")
    test_path = _resolved_input(test_feature_path, ".csv")
    if train_path == test_path:
        raise BiomarkerDiscoveryError("训练和测试特征表不能是同一个文件。")
    reused = _reuse_discovery_result(output_dir, train_path, test_path)
    if reused is not None:
        return reused
    destination = _prepare_output_dir(output_dir)
    requested_covariates = _parse_covariates(covariates)

    try:
        raw_train = _normalize_columns(pd.read_csv(train_path, low_memory=False))
        raw_test = _normalize_columns(pd.read_csv(test_path, low_memory=False))
        train, excluded_train = _select_labeled_rows(raw_train, endpoint, "训练集")
        test, excluded_test = _select_labeled_rows(raw_test, endpoint, "测试集")
        identifier_qc = _validate_identifiers(train, test, id_column)
        train_covariates, test_covariates, covariate_names, covariate_plan = _encode_covariates(
            train, test, requested_covariates
        )
        features, rejected_features = _feature_columns(
            train, test, endpoint, id_column, requested_covariates
        )
        report_progress(
            8.0,
            f"数据质控完成：训练 {len(train)} 例，测试 {len(test)} 例，可检验特征 {len(features)} 个",
        )

        train_y = train[endpoint].to_numpy(dtype=np.int64)
        test_y = test[endpoint].to_numpy(dtype=np.int64)
        reduced_train = fit_ordered_logit(train_y, train_covariates)
        if not reduced_train.converged:
            raise BiomarkerDiscoveryError(
                f"训练集协变量基线有序模型未收敛: {reduced_train.message}"
            )
        report_progress(12.0, "训练集协变量基线模型已拟合")

        discovery_rows: list[dict[str, Any]] = []
        train_feature_vectors: dict[str, np.ndarray] = {}
        test_feature_vectors: dict[str, np.ndarray] = {}
        preprocessing: dict[str, dict[str, float]] = {}
        for feature_index, feature in enumerate(features, start=1):
            train_values, test_values, feature_plan = _feature_arrays(train, test, feature)
            train_feature_vectors[feature] = train_values
            test_feature_vectors[feature] = test_values
            preprocessing[feature] = feature_plan
            try:
                evidence = feature_association(
                    train_y,
                    train_covariates,
                    train_values,
                    reduced_fit=reduced_train,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                evidence = {"converged": False, "error": str(exc), "iterations": 0}
            discovery_rows.append(
                {
                    "feature": feature,
                    **feature_plan,
                    **evidence,
                }
            )
            report_progress(
                12.0 + 38.0 * feature_index / max(1, len(features)),
                f"训练集特征扫描 {feature_index}/{len(features)}；当前特征={feature}",
            )

        converged_indexes = [
            index for index, row in enumerate(discovery_rows) if row.get("converged")
        ]
        if not converged_indexes:
            raise BiomarkerDiscoveryError("所有候选特征的有序模型均未收敛。")
        q_values = adjust_bh(discovery_rows[index]["p_value"] for index in converged_indexes)
        for row in discovery_rows:
            row["q_value"] = None
            row["fdr_significant"] = False
            row["candidate_selected"] = False
            row["redundant_with"] = None
        for index, q_value in zip(converged_indexes, q_values.tolist()):
            discovery_rows[index]["q_value"] = float(q_value)
            discovery_rows[index]["fdr_significant"] = bool(q_value < fdr_alpha)

        significant = sorted(
            [row for row in discovery_rows if row.get("fdr_significant")],
            key=lambda row: (float(row["q_value"]), float(row["p_value"]), str(row["feature"])),
        )
        selected, redundant = _prune_correlated_candidates(
            significant,
            train_feature_vectors,
            correlation_threshold=float(correlation_threshold),
            maximum=int(maximum_candidates),
        )
        selected_names = {str(row["feature"]) for row in selected}
        for row in discovery_rows:
            feature = str(row["feature"])
            row["candidate_selected"] = feature in selected_names
            row["redundant_with"] = redundant.get(feature)
        report_progress(
            55.0,
            f"训练集多重校正与去冗余完成：冻结候选 {len(selected)} 个",
        )

        frozen_candidates: list[dict[str, Any]] = []
        for index, row in enumerate(selected, start=1):
            feature = str(row["feature"])
            support = _bootstrap_direction_support(
                train_y,
                train_covariates,
                train_feature_vectors[feature],
                float(row["beta_per_sd"]),
                iterations=int(bootstrap_iterations),
                seed=int(random_seed) + index,
            )
            frozen_candidates.append(
                {
                    "candidate_id": f"BIO-{index:04d}",
                    "feature": feature,
                    "train_beta_per_sd": float(row["beta_per_sd"]),
                    "train_odds_ratio_per_sd": float(row["odds_ratio_per_sd"]),
                    "train_odds_ratio_ci_low": float(row["odds_ratio_ci_low"]),
                    "train_odds_ratio_ci_high": float(row["odds_ratio_ci_high"]),
                    "train_p_value": float(row["p_value"]),
                    "train_q_value": float(row["q_value"]),
                    "preprocessing": preprocessing[feature],
                    "bootstrap": support,
                }
            )
            report_progress(
                55.0 + 30.0 * index / max(1, len(selected)),
                f"候选稳定性 bootstrap {index}/{len(selected)}；当前特征={feature}",
            )
        if not selected:
            report_progress(85.0, "训练集没有候选进入 bootstrap 阶段")

        candidate_set_payload = [
            {
                "candidate_id": row["candidate_id"],
                "feature": row["feature"],
                "preprocessing": row["preprocessing"],
            }
            for row in frozen_candidates
        ]
        candidate_set_sha256 = _sha256_json(candidate_set_payload)
        candidate_manifest = {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "created_at": _utc_now(),
            "discovery_split": "train",
            "confirmation_split": "test",
            "test_data_used_for_discovery": False,
            "endpoint": endpoint,
            "endpoint_scale": "ordinal_0_4",
            "fdr_method": "Benjamini-Hochberg",
            "fdr_alpha": float(fdr_alpha),
            "correlation_pruning_threshold": float(correlation_threshold),
            "bootstrap_support_threshold": float(bootstrap_support_threshold),
            "candidate_set_sha256": candidate_set_sha256,
            "candidate_count": len(frozen_candidates),
            "candidates": frozen_candidates,
        }

        confirmation_rows: list[dict[str, Any]] = []
        if frozen_candidates:
            reduced_test = fit_ordered_logit(test_y, test_covariates)
            if not reduced_test.converged:
                raise BiomarkerDiscoveryError(
                    f"测试集协变量基线有序模型未收敛: {reduced_test.message}"
                )
            for confirmation_index, candidate in enumerate(frozen_candidates, start=1):
                feature = str(candidate["feature"])
                try:
                    evidence = feature_association(
                        test_y,
                        test_covariates,
                        test_feature_vectors[feature],
                        reduced_fit=reduced_test,
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    evidence = {"converged": False, "error": str(exc), "p_value": 1.0}
                confirmation_rows.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "feature": feature,
                        **evidence,
                    }
                )
                report_progress(
                    85.0 + 10.0 * confirmation_index / max(1, len(frozen_candidates)),
                    f"冻结测试集确认 {confirmation_index}/{len(frozen_candidates)}；当前特征={feature}",
                )
            holm_values = adjust_holm(
                float(row.get("p_value", 1.0)) if row.get("converged") else 1.0
                for row in confirmation_rows
            )
            for row, adjusted in zip(confirmation_rows, holm_values.tolist()):
                row["holm_p_value"] = float(adjusted)
                source = next(
                    item for item in frozen_candidates
                    if item["candidate_id"] == row["candidate_id"]
                )
                direction_concordant = bool(
                    row.get("converged")
                    and np.sign(float(row.get("beta_per_sd", 0.0)))
                    == np.sign(float(source["train_beta_per_sd"]))
                )
                bootstrap_ok = bool(
                    float(source["bootstrap"].get("directional_support", 0.0))
                    >= float(bootstrap_support_threshold)
                )
                confirmed = bool(
                    direction_concordant
                    and bootstrap_ok
                    and float(adjusted) < float(confirmation_alpha)
                )
                row["direction_concordant"] = direction_concordant
                row["bootstrap_support_pass"] = bootstrap_ok
                row["confirmation_status"] = "confirmed" if confirmed else "not_confirmed"
        else:
            report_progress(95.0, "没有冻结候选需要进入测试集确认")

        confirmation_by_id = {row["candidate_id"]: row for row in confirmation_rows}
        combined_candidates: list[dict[str, Any]] = []
        for candidate in frozen_candidates:
            confirmation = confirmation_by_id.get(candidate["candidate_id"], {})
            combined_candidates.append(
                {
                    **candidate,
                    "test_beta_per_sd": confirmation.get("beta_per_sd"),
                    "test_odds_ratio_per_sd": confirmation.get("odds_ratio_per_sd"),
                    "test_odds_ratio_ci_low": confirmation.get("odds_ratio_ci_low"),
                    "test_odds_ratio_ci_high": confirmation.get("odds_ratio_ci_high"),
                    "test_p_value": confirmation.get("p_value"),
                    "test_holm_p_value": confirmation.get("holm_p_value"),
                    "direction_concordant": confirmation.get("direction_concordant", False),
                    "confirmation_status": confirmation.get(
                        "confirmation_status", "not_tested"
                    ),
                }
            )
        confirmed_count = sum(
            row["confirmation_status"] == "confirmed" for row in combined_candidates
        )
        if confirmed_count:
            scientific_outcome = "completed_confirmed_candidates"
        elif frozen_candidates:
            scientific_outcome = "completed_no_confirmed_biomarker"
        else:
            scientific_outcome = "completed_no_discovery_candidates"

        dataset_qc_path = destination / "dataset_qc.json"
        analysis_plan_path = destination / "analysis_plan.json"
        discovery_path = destination / "discovery_results.csv"
        candidate_path = destination / "candidate_manifest.json"
        confirmation_path = destination / "confirmation_results.csv"
        report_path = destination / "biomarker_report.md"
        manifest_path = destination / "biomarker_manifest.json"

        dataset_qc = {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "status": "pass",
            "train_source": str(train_path),
            "test_source": str(test_path),
            "train_sha256": _sha256_file(train_path),
            "test_sha256": _sha256_file(test_path),
            "train_input_rows": int(len(raw_train)),
            "test_input_rows": int(len(raw_test)),
            "train_labeled_rows": int(len(train)),
            "test_labeled_rows": int(len(test)),
            "train_unlabeled_excluded": excluded_train,
            "test_unlabeled_excluded": excluded_test,
            "train_kl_distribution": {
                str(key): int(value)
                for key, value in train[endpoint].value_counts().sort_index().items()
            },
            "test_kl_distribution": {
                str(key): int(value)
                for key, value in test[endpoint].value_counts().sort_index().items()
            },
            "identifier_qc": identifier_qc,
            "eligible_feature_count": len(features),
            "rejected_features": rejected_features,
            "covariate_plan": covariate_plan,
        }
        analysis_plan = {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "plan_version": "KL_ORDINAL_BIOMARKER_V1",
            "task": "cross-sectional MRI candidate biomarker discovery",
            "endpoint": endpoint,
            "endpoint_scale": "ordinal_0_4",
            "unit_of_analysis": "knee",
            "model": "proportional_odds_cumulative_logit",
            "effect": "odds ratio for higher KL grade per training SD increase",
            "covariates_requested": requested_covariates,
            "covariates_used": covariate_names,
            "discovery_multiplicity": "Benjamini-Hochberg FDR",
            "confirmation_multiplicity": "Holm FWER",
            "fdr_alpha": float(fdr_alpha),
            "confirmation_alpha": float(confirmation_alpha),
            "correlation_threshold": float(correlation_threshold),
            "maximum_candidates": int(maximum_candidates),
            "bootstrap_iterations": int(bootstrap_iterations),
            "bootstrap_support_threshold": float(bootstrap_support_threshold),
            "random_seed": int(random_seed),
            "test_data_policy": "confirmation_only_after_candidate_freeze",
            "method_references": METHOD_REFERENCES,
        }
        discovery_frame = pd.DataFrame(discovery_rows).sort_values(
            ["converged", "q_value", "p_value", "feature"],
            ascending=[False, True, True, True],
            na_position="last",
        )
        confirmation_columns = [
            "candidate_id", "feature", "converged", "beta_per_sd", "standard_error",
            "odds_ratio_per_sd", "odds_ratio_ci_low", "odds_ratio_ci_high",
            "likelihood_ratio", "p_value", "holm_p_value", "direction_concordant",
            "bootstrap_support_pass", "confirmation_status", "error", "iterations",
        ]
        confirmation_frame = pd.DataFrame(confirmation_rows)
        for column in confirmation_columns:
            if column not in confirmation_frame.columns:
                confirmation_frame[column] = pd.Series(dtype="object")
        confirmation_frame = confirmation_frame[confirmation_columns]
        report = _build_report(
            train_count=len(train),
            test_count=len(test),
            feature_count=len(converged_indexes),
            candidate_count=len(frozen_candidates),
            confirmed_count=confirmed_count,
            scientific_outcome=scientific_outcome,
            candidate_rows=combined_candidates,
        )

        _atomic_json(dataset_qc_path, dataset_qc)
        _atomic_json(analysis_plan_path, analysis_plan)
        _atomic_csv(discovery_path, discovery_frame)
        _atomic_json(candidate_path, candidate_manifest)
        _atomic_csv(confirmation_path, confirmation_frame)
        _atomic_text(report_path, report)
        artifact_paths = [
            dataset_qc_path, analysis_plan_path, discovery_path,
            candidate_path, confirmation_path, report_path,
        ]
        manifest = {
            "schema_version": SERVICE_SCHEMA_VERSION,
            "service_version": SERVICE_VERSION,
            "created_at": _utc_now(),
            "status": "success",
            "scientific_outcome": scientific_outcome,
            "research_only": True,
            "clinical_use_prohibited": True,
            "train_source": str(train_path),
            "test_source": str(test_path),
            "train_sha256": _sha256_file(train_path),
            "test_sha256": _sha256_file(test_path),
            "candidate_set_sha256": candidate_set_sha256,
            "candidate_count": len(frozen_candidates),
            "confirmed_count": confirmed_count,
            "test_data_used_for_discovery": False,
            "test_data_used_for_confirmation": True,
            "artifact_hashes": {
                path.name: _sha256_file(path) for path in artifact_paths
            },
            "artifacts": {path.stem: path.name for path in artifact_paths},
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
            "method_references": METHOD_REFERENCES,
        }
        _atomic_json(manifest_path, manifest)
        files = [str(path) for path in [*artifact_paths, manifest_path]]
        report_progress(
            100.0,
            f"分析完成：冻结候选 {len(frozen_candidates)} 个，测试集确认 {confirmed_count} 个",
        )
        return {
            "status": "success",
            "message": (
                f"生物标志物发现流程完成：训练集冻结 {len(frozen_candidates)} 个候选，"
                f"测试集确认 {confirmed_count} 个。"
            ),
            "scientific_outcome": scientific_outcome,
            "output_dir": str(destination),
            "manifest_path": str(manifest_path),
            "report_path": str(report_path),
            "candidate_manifest_path": str(candidate_path),
            "discovery_results_path": str(discovery_path),
            "confirmation_results_path": str(confirmation_path),
            "candidate_count": len(frozen_candidates),
            "confirmed_count": confirmed_count,
            "files": files,
        }
    except Exception:
        # Do not delete partial evidence automatically.  A failed run remains
        # inspectable and the non-empty output directory prevents accidental
        # reuse as a supposedly clean run.
        raise


def audit_biomarker_discovery(
    manifest_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Independently verify hashes, frozen candidates, multiplicity and claims."""
    manifest_file = _resolved_input(manifest_path, ".json")
    reused = _reuse_audit_result(output_dir, manifest_file)
    if reused is not None:
        return reused
    destination = _prepare_output_dir(output_dir)
    run_dir = manifest_file.parent.resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BiomarkerDiscoveryError(f"无法读取生物标志物运行清单: {exc}") from exc

    gates: dict[str, dict[str, Any]] = {}
    artifact_hashes = manifest.get("artifact_hashes", {})
    hash_failures: list[str] = []
    for filename, expected in dict(artifact_hashes).items():
        path = (run_dir / str(filename)).resolve()
        if path.parent != run_dir or not path.is_file():
            hash_failures.append(f"{filename}:missing_or_outside_run")
        elif _sha256_file(path) != str(expected):
            hash_failures.append(f"{filename}:sha256_mismatch")
    gates["artifact_integrity"] = {
        "status": "pass" if not hash_failures else "fail",
        "details": hash_failures,
    }

    no_test_leakage = manifest.get("test_data_used_for_discovery") is False
    gates["test_not_used_for_discovery"] = {
        "status": "pass" if no_test_leakage else "fail",
        "details": {"observed": manifest.get("test_data_used_for_discovery")},
    }

    candidate_file = run_dir / "candidate_manifest.json"
    discovery_file = run_dir / "discovery_results.csv"
    confirmation_file = run_dir / "confirmation_results.csv"
    dataset_qc_file = run_dir / "dataset_qc.json"
    analysis_plan_file = run_dir / "analysis_plan.json"
    table_errors: list[str] = []
    try:
        candidate_manifest = json.loads(candidate_file.read_text(encoding="utf-8"))
        dataset_qc = json.loads(dataset_qc_file.read_text(encoding="utf-8"))
        analysis_plan = json.loads(analysis_plan_file.read_text(encoding="utf-8"))
        discovery = pd.read_csv(discovery_file)
        confirmation = pd.read_csv(confirmation_file)
    except (OSError, UnicodeError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        candidate_manifest = {}
        dataset_qc = {}
        analysis_plan = {}
        discovery = pd.DataFrame()
        confirmation = pd.DataFrame()
        table_errors.append(type(exc).__name__)

    candidates = list(candidate_manifest.get("candidates", []))
    candidate_ids = [str(item.get("candidate_id")) for item in candidates]
    confirmation_ids = (
        [str(value) for value in confirmation.get("candidate_id", pd.Series(dtype=str)).tolist()]
        if not confirmation.empty else []
    )
    if candidate_ids != confirmation_ids:
        table_errors.append("confirmation_candidate_set_mismatch")
    payload = [
        {
            "candidate_id": item.get("candidate_id"),
            "feature": item.get("feature"),
            "preprocessing": item.get("preprocessing"),
        }
        for item in candidates
    ]
    observed_candidate_hash = _sha256_json(payload)
    if observed_candidate_hash != candidate_manifest.get("candidate_set_sha256"):
        table_errors.append("candidate_manifest_hash_mismatch")
    if observed_candidate_hash != manifest.get("candidate_set_sha256"):
        table_errors.append("run_manifest_candidate_hash_mismatch")
    gates["candidate_freeze"] = {
        "status": "pass" if not table_errors else "fail",
        "details": table_errors,
    }

    multiplicity_errors: list[str] = []
    if not discovery.empty and {"converged", "p_value", "q_value"}.issubset(discovery.columns):
        converged = discovery[discovery["converged"].astype(str).str.lower().isin({"true", "1"})]
        recalculated = adjust_bh(pd.to_numeric(converged["p_value"], errors="coerce"))
        observed = pd.to_numeric(converged["q_value"], errors="coerce").to_numpy()
        if len(recalculated) != len(observed) or not np.allclose(
            recalculated, observed, rtol=1e-6, atol=1e-8, equal_nan=False
        ):
            multiplicity_errors.append("discovery_bh_mismatch")
    elif not discovery.empty:
        multiplicity_errors.append("discovery_columns_missing")
    if not confirmation.empty and {"p_value", "holm_p_value"}.issubset(confirmation.columns):
        recalculated = adjust_holm(pd.to_numeric(confirmation["p_value"], errors="coerce").fillna(1.0))
        observed = pd.to_numeric(confirmation["holm_p_value"], errors="coerce").to_numpy()
        if len(recalculated) != len(observed) or not np.allclose(
            recalculated, observed, rtol=1e-6, atol=1e-8, equal_nan=False
        ):
            multiplicity_errors.append("confirmation_holm_mismatch")
    gates["multiplicity_recomputed"] = {
        "status": "pass" if not multiplicity_errors else "fail",
        "details": multiplicity_errors,
    }

    confirmation_decision_errors: list[str] = []
    confirmation_records = {
        str(row.get("candidate_id")): row
        for row in confirmation.to_dict(orient="records")
    }
    confirmation_alpha = float(analysis_plan.get("confirmation_alpha", 0.05))
    support_threshold = float(
        candidate_manifest.get("bootstrap_support_threshold", 0.70)
    )
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id"))
        row = confirmation_records.get(candidate_id)
        if row is None:
            confirmation_decision_errors.append(f"{candidate_id}:missing_confirmation")
            continue
        train_beta = _finite_float_or_none(candidate.get("train_beta_per_sd"))
        test_beta = _finite_float_or_none(row.get("beta_per_sd"))
        holm_p = _finite_float_or_none(row.get("holm_p_value"))
        support = _finite_float_or_none(
            dict(candidate.get("bootstrap", {})).get("directional_support")
        )
        expected_direction = bool(
            train_beta is not None
            and test_beta is not None
            and np.sign(train_beta) == np.sign(test_beta)
        )
        expected_support = bool(
            support is not None and support >= support_threshold
        )
        expected_confirmed = bool(
            expected_direction
            and expected_support
            and holm_p is not None
            and holm_p < confirmation_alpha
        )
        observed_confirmed = str(
            row.get("confirmation_status", "")
        ).strip().lower() == "confirmed"
        if expected_confirmed != observed_confirmed:
            confirmation_decision_errors.append(
                f"{candidate_id}:confirmation_status_mismatch"
            )
    gates["confirmation_decisions_recomputed"] = {
        "status": "pass" if not confirmation_decision_errors else "fail",
        "details": confirmation_decision_errors,
    }

    confirmed_count = 0
    if not confirmation.empty and "confirmation_status" in confirmation.columns:
        confirmed_count = int((confirmation["confirmation_status"] == "confirmed").sum())
    confirmation_by_id = confirmation_records
    confirmed_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        confirmation_row = confirmation_by_id.get(str(candidate.get("candidate_id")), {})
        if confirmation_row.get("confirmation_status") != "confirmed":
            continue
        train_beta = _finite_float_or_none(candidate.get("train_beta_per_sd"))
        confirmed_candidates.append(
            {
                "candidate_id": str(candidate.get("candidate_id")),
                "feature": str(candidate.get("feature")),
                "direction": (
                    "higher_feature_higher_KL"
                    if train_beta is not None and train_beta > 0
                    else "higher_feature_lower_KL"
                ),
                "train_beta_per_sd": train_beta,
                "train_odds_ratio_per_sd": _finite_float_or_none(
                    candidate.get("train_odds_ratio_per_sd")
                ),
                "train_q_value": _finite_float_or_none(
                    candidate.get("train_q_value")
                ),
                "bootstrap_directional_support": _finite_float_or_none(
                    dict(candidate.get("bootstrap", {})).get("directional_support")
                ),
                "test_beta_per_sd": _finite_float_or_none(
                    confirmation_row.get("beta_per_sd")
                ),
                "test_odds_ratio_per_sd": _finite_float_or_none(
                    confirmation_row.get("odds_ratio_per_sd")
                ),
                "test_holm_p_value": _finite_float_or_none(
                    confirmation_row.get("holm_p_value")
                ),
            }
        )
    confirmed_candidates.sort(
        key=lambda item: (
            item.get("test_holm_p_value")
            if item.get("test_holm_p_value") is not None
            else 1.0,
            item.get("feature", ""),
        )
    )
    count_ok = (
        int(manifest.get("candidate_count", -1)) == len(candidates)
        and int(manifest.get("confirmed_count", -1)) == confirmed_count
    )
    gates["reported_counts_reconcile"] = {
        "status": "pass" if count_ok else "fail",
        "details": {
            "manifest_candidate_count": manifest.get("candidate_count"),
            "observed_candidate_count": len(candidates),
            "manifest_confirmed_count": manifest.get("confirmed_count"),
            "observed_confirmed_count": confirmed_count,
        },
    }
    valid = all(gate["status"] == "pass" for gate in gates.values())
    audit = {
        "schema_version": SERVICE_SCHEMA_VERSION,
        "service_version": SERVICE_VERSION,
        "audited_at": _utc_now(),
        "manifest_path": str(manifest_file),
        "manifest_sha256": _sha256_file(manifest_file),
        "valid": valid,
        "gates": gates,
        "candidate_count": len(candidates),
        "confirmed_count": confirmed_count,
        "scientific_outcome": manifest.get("scientific_outcome"),
        "dataset_summary": {
            "train_input_rows": int(dataset_qc.get("train_input_rows", 0) or 0),
            "test_input_rows": int(dataset_qc.get("test_input_rows", 0) or 0),
            "train_labeled_rows": int(dataset_qc.get("train_labeled_rows", 0) or 0),
            "test_labeled_rows": int(dataset_qc.get("test_labeled_rows", 0) or 0),
            "train_unlabeled_excluded": int(
                dataset_qc.get("train_unlabeled_excluded", 0) or 0
            ),
            "test_unlabeled_excluded": int(
                dataset_qc.get("test_unlabeled_excluded", 0) or 0
            ),
            "eligible_feature_count": int(
                dataset_qc.get("eligible_feature_count", 0) or 0
            ),
            "covariates_used": list(
                dict(dataset_qc.get("covariate_plan", {})).get(
                    "expanded_columns", []
                )
            ),
            "covariates_unavailable": list(
                dict(dataset_qc.get("covariate_plan", {})).get("unavailable", [])
            ),
        },
        "confirmed_candidates": confirmed_candidates,
        "source_report_path": str(run_dir / "biomarker_report.md"),
        "research_only": True,
    }
    audit_json = destination / "biomarker_audit.json"
    audit_markdown = destination / "biomarker_audit.md"
    _atomic_json(audit_json, audit)
    lines = [
        "# 生物标志物发现运行审计",
        "",
        f"- 总体：{'PASS' if valid else 'FAIL'}",
        f"- 候选数：{len(candidates)}",
        f"- 确认数：{confirmed_count}",
        "",
        "| 门禁 | 状态 |",
        "|---|---|",
    ]
    for name, gate in gates.items():
        lines.append(f"| {name} | {gate['status'].upper()} |")
    _atomic_text(audit_markdown, "\n".join(lines) + "\n")
    return {
        "status": "success" if valid else "failed",
        "message": "生物标志物运行审计通过。" if valid else "生物标志物运行审计未通过。",
        "valid": valid,
        "audit_path": str(audit_json),
        "report_path": str(audit_markdown),
        "candidate_count": len(candidates),
        "confirmed_count": confirmed_count,
        "scientific_outcome": manifest.get("scientific_outcome"),
        "dataset_summary": audit["dataset_summary"],
        "confirmed_candidates": confirmed_candidates,
        "source_report_path": audit["source_report_path"],
        "files": [str(audit_json), str(audit_markdown)],
        "gates": gates,
    }
