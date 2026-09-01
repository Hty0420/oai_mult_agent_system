"""Deterministic, privacy-minimizing dataset profiling for KL workflows."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

from .path_utils import collect_csv_paths, infer_dataset_role


DATASET_PROFILE_VERSION = "1.0.0"
_MODULE_SRC_ROOT = Path(__file__).resolve().parents[1]
_COLUMN_ALIASES = {
    "caseid": "CaseID",
    "case_id": "CaseID",
    "imagepath": "ImagePath",
    "image_path": "ImagePath",
    "maskpath": "MaskPath",
    "mask_path": "MaskPath",
    "klgrade": "KLGrade",
    "kl_grade": "KLGrade",
    "koa_grade": "KLGrade",
}


def _canonical_column(name: str) -> str:
    stripped = str(name or "").strip()
    return _COLUMN_ALIASES.get(stripped.lower(), stripped)


def _resolve_existing_path(path: str, roots: Iterable[Path]) -> Path | None:
    candidate = Path(path)
    candidates = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
    for item in candidates:
        try:
            resolved = item.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _default_csv_candidates(roots: Iterable[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in roots:
        for relative in (
            Path("datasets") / "labels_train.csv",
            Path("datasets") / "labels_test.csv",
            Path("src") / "datasets" / "labels_train.csv",
            Path("src") / "datasets" / "labels_test.csv",
            Path("user_upload_file") / "labels_train.csv",
            Path("user_upload_file") / "labels_test.csv",
            Path("src") / "user_upload_file" / "labels_train.csv",
            Path("src") / "user_upload_file" / "labels_test.csv",
        ):
            candidate = (root / relative).resolve()
            if candidate.is_file() and candidate not in candidates:
                candidates.append(candidate)
    return candidates


def discover_source_csvs(
    query: str,
    *,
    search_roots: Iterable[str | Path] | None = None,
    preferred_paths: Iterable[str | Path] | None = None,
) -> list[Path]:
    """Prefer current request, then current session, then bounded defaults."""
    roots = [Path.cwd(), _MODULE_SRC_ROOT]
    if search_roots:
        roots = [Path(root) for root in search_roots] + roots
    roots = list(dict.fromkeys(root.resolve() for root in roots))

    discovered: list[Path] = []
    for raw_path in collect_csv_paths(str(query or "")):
        resolved = _resolve_existing_path(raw_path, roots)
        if resolved is not None and resolved not in discovered:
            discovered.append(resolved)
    if discovered:
        return discovered
    for raw_path in preferred_paths or []:
        resolved = _resolve_existing_path(str(raw_path), roots)
        if resolved is not None and resolved.suffix.lower() == ".csv":
            if resolved not in discovered:
                discovered.append(resolved)
    if discovered:
        return discovered
    return _default_csv_candidates(roots)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_csv(path: Path, role: str) -> dict[str, Any]:
    row_count = 0
    class_distribution: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_columns = list(reader.fieldnames or [])
        canonical_by_raw = {column: _canonical_column(column) for column in raw_columns}
        for row in reader:
            row_count += 1
            label_key = next(
                (raw for raw, canonical in canonical_by_raw.items() if canonical == "KLGrade"),
                None,
            )
            if label_key:
                label = str(row.get(label_key, "")).strip()
                if label:
                    class_distribution[label] = class_distribution.get(label, 0) + 1

    canonical_columns = list(dict.fromkeys(canonical_by_raw.values()))
    return {
        "role": role,
        "path": str(path),
        "sha256": _sha256(path),
        "row_count": row_count,
        "columns": canonical_columns,
        "class_distribution": class_distribution,
    }


def profile_kl_dataset(
    query: str,
    *,
    search_roots: Iterable[str | Path] | None = None,
    preferred_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Build a non-patient-level profile used by the autonomous planner."""
    files = discover_source_csvs(
        query,
        search_roots=search_roots,
        preferred_paths=preferred_paths,
    )
    profiles: dict[str, dict[str, Any]] = {}
    unassigned: list[str] = []
    issues: list[str] = []

    for path in files:
        role = infer_dataset_role(str(path))
        if role not in {"train", "test"}:
            unassigned.append(str(path))
            continue
        if role in profiles:
            issues.append(f"multiple_{role}_csvs")
            continue
        try:
            profiles[role] = _profile_csv(path, role)
        except (OSError, csv.Error, UnicodeError) as exc:
            issues.append(f"unreadable_{role}_csv:{type(exc).__name__}")

    for role in ("train", "test"):
        if role not in profiles:
            issues.append(f"missing_{role}_csv")
    if profiles:
        for role, profile in profiles.items():
            if "KLGrade" not in profile["columns"]:
                issues.append(f"missing_KLGrade:{role}")
            if profile["row_count"] <= 0:
                issues.append(f"empty_{role}_csv")

    hash_payload = "|".join(
        f"{role}:{profiles[role]['sha256']}" for role in sorted(profiles)
    )
    dataset_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest() if hash_payload else None
    blocking = {
        issue for issue in issues
        if issue.startswith(("missing_train", "missing_test", "unreadable", "missing_KLGrade", "empty"))
    }
    return {
        "schema_version": "1.0",
        "profiler_version": DATASET_PROFILE_VERSION,
        "status": "ready" if not blocking else "blocked",
        "dataset_hash": dataset_hash,
        "source_files": profiles,
        "unassigned_files": unassigned,
        "issues": issues,
    }
