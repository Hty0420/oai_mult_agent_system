"""Shared helpers for passing generated feature paths between agents."""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Iterable


# Preserve Windows drive prefixes while accepting the same relative paths.
_CSV_PATH_RE = re.compile(
    r"((?:[A-Za-z]:[\\/])?(?:[^\\/\s\"'`<>|,;:：{}\[\]()]+[\\/])+"
    r"[^\\/\s\"'`<>|,;:：{}\[\]()]+\.csv)",
    re.IGNORECASE,
)

_ABSOLUTE_NIFTI_PATH_RE = re.compile(
    r"((?:[A-Za-z]:[\\/]|\\\\)[^\"'\r\n<>|]+?\.nii(?:\.gz)?)",
    re.IGNORECASE,
)
_RELATIVE_NIFTI_PATH_RE = re.compile(
    r"((?:[^\\/\s\"'`<>|,;:：{}\[\]()]+[\\/])+"
    r"[^\\/\s\"'`<>|,;:：{}\[\]()]+\.nii(?:\.gz)?)",
    re.IGNORECASE,
)


_TOOL_RESULT_PATH_KEYS = {
    "train_features_path",
    "test_features_path",
    "train_features",
    "test_features",
    "output_csv",
    "filtered_csv",
    "report_path",
    "model_path",
    "manifest_path",
    "global_csv",
    "subregion_csv",
    "analysis_json",
    "subregion_mask",
    "candidate_manifest_path",
    "discovery_results_path",
    "confirmation_results_path",
    "audit_path",
    "dataset_qc_path",
    "analysis_plan_path",
}

_TOOL_RESULT_PATH_MAP_KEYS = {
    "thickness_maps",
}

_TOOL_RESULT_PATH_LIST_KEYS = {
    "files",
    "output_files",
}


def _decode_repr_escaped_separators(path: str) -> str:
    """Decode repr-escaped Windows separators without changing literal UNC paths."""
    escaped_drive_path = len(path) > 3 and path[1] == ":" and path[2:4] == "\\\\"
    leading_separators = len(path) - len(path.lstrip("\\"))
    escaped_unc_path = leading_separators >= 4
    if escaped_drive_path or escaped_unc_path:
        return path.replace("\\\\", "\\")
    return path


def _iter_csv_path_candidates(text: str) -> Iterable[str]:
    """Yield CSV-looking whitespace-delimited path candidates from tool text."""
    for token in text.split():
        path = token.strip("\"'`{}[](),;，。")
        if path.lower().endswith(".csv") and ("\\" in path or "/" in path):
            yield _decode_repr_escaped_separators(path)


def _is_placeholder(path: str) -> bool:
    lower = path.lower()
    return "path/to/" in lower or "path\\to\\" in lower or "placeholder" in lower


def _clean_path(path: str) -> str:
    return path.strip().strip("\"'`").rstrip(".,;，。)")


def _collect_csv_paths_free_text(text: str) -> list[str]:
    """Collect CSV paths from unstructured text only."""
    if not text:
        return []

    paths: list[str] = []
    seen: set[str] = set()
    candidates = list(_iter_csv_path_candidates(text))
    if not candidates:
        candidates = _CSV_PATH_RE.findall(text)

    for match in candidates:
        path = _clean_path(match)
        if not path or _is_placeholder(path):
            continue
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def parse_tool_result_paths(value: Any) -> list[tuple[str, str]]:
    """Parse structured or textual tool output without losing path boundaries."""
    parsed_value = value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                parsed_value = json.loads(stripped)
            except json.JSONDecodeError:
                try:
                    parsed_value = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    parsed_value = value

    results: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()

    def remember(key: str, path: str) -> None:
        cleaned = _clean_path(_decode_repr_escaped_separators(path))
        item = (key, cleaned)
        if cleaned and not _is_placeholder(cleaned) and item not in seen:
            seen.add(item)
            seen_paths.add(cleaned)
            results.append(item)

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for raw_key, child in item.items():
                key = str(raw_key)
                if key in _TOOL_RESULT_PATH_KEYS and isinstance(child, str):
                    remember(key, child)
                elif key in _TOOL_RESULT_PATH_MAP_KEYS and isinstance(child, dict):
                    for map_key, map_path in child.items():
                        if isinstance(map_path, str):
                            remember(f"{key}.{map_key}", map_path)
                else:
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(parsed_value)

    def visit_path_lists(item: Any) -> None:
        if isinstance(item, dict):
            for raw_key, child in item.items():
                key = str(raw_key)
                if key in _TOOL_RESULT_PATH_LIST_KEYS and isinstance(child, (list, tuple)):
                    for path in child:
                        if isinstance(path, str):
                            cleaned = _clean_path(_decode_repr_escaped_separators(path))
                            if cleaned not in seen_paths:
                                remember(key, path)
                else:
                    visit_path_lists(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit_path_lists(child)

    visit_path_lists(parsed_value)
    if results:
        return results

    if isinstance(value, str):
        key_pattern = "|".join(
            re.escape(key)
            for key in sorted(_TOOL_RESULT_PATH_KEYS, key=len, reverse=True)
        )
        line_pattern = re.compile(
            rf"^\s*(?:[-*]\s*)?({key_pattern})\s*[:=]\s*(.+?)\s*$",
            re.IGNORECASE,
        )
        for line in value.splitlines():
            match = line_pattern.match(line)
            if match:
                remember(match.group(1).lower(), match.group(2).rstrip(",;"))
        if results:
            return results

        for csv_path in _collect_csv_paths_free_text(value):
            remember("output_csv", csv_path)
    return results


def collect_csv_paths(text: str) -> list[str]:
    """Collect CSV paths, preferring structured fields that preserve spaces."""
    structured_paths = [
        path
        for _, path in parse_tool_result_paths(text)
        if path.lower().endswith(".csv")
    ]
    if structured_paths:
        return list(dict.fromkeys(structured_paths))
    return _collect_csv_paths_free_text(text)


def collect_nifti_paths(text: str) -> list[str]:
    """Collect absolute or relative .nii/.nii.gz paths from user/tool text."""
    if not text:
        return []
    paths: list[str] = []
    seen: set[str] = set()
    for pattern_index, pattern in enumerate(
        (_ABSOLUTE_NIFTI_PATH_RE, _RELATIVE_NIFTI_PATH_RE)
    ):
        for match in pattern.findall(text):
            path = _clean_path(match)
            if not path or _is_placeholder(path) or path in seen:
                continue
            # A relative-pattern match can be the tail of an already captured
            # absolute Windows path (``data\\case.nii.gz`` inside
            # ``D:\\data\\case.nii.gz``).  Keep only the authoritative path.
            if pattern_index == 1 and any(
                existing.lower().endswith(path.lower()) for existing in paths
            ):
                continue
            seen.add(path)
            paths.append(path)
    return paths


def feature_csv_score(path: str) -> int:
    """Score whether a CSV path looks like a generated feature table."""
    normalized = path.replace("\\", "/").lower()
    basename = os.path.basename(normalized)

    if basename == "filtered_features.csv":
        return 4
    if basename.endswith("_features_summary.csv"):
        return 3
    if basename.endswith("_augmented_features.csv"):
        return 3
    if "_features_" in normalized and basename.endswith(".csv"):
        return 2
    return 0


def prefer_generated_feature_paths(paths: Iterable[str]) -> list[str]:
    """Prefer generated feature CSVs over uploaded label/metadata CSVs."""
    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path or _is_placeholder(path):
            continue
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)

    generated = [path for path in unique_paths if feature_csv_score(path) > 0]
    if not generated:
        return unique_paths

    return sorted(generated, key=feature_csv_score, reverse=True)


def infer_dataset_role(path: str, label: str = "") -> str | None:
    """Infer whether a path describes train or test data."""
    label_parts = re.split(r"[\\/_.\-\s]+", label.lower()) if label else []
    if "test" in label_parts or "testing" in label_parts:
        return "test"
    if "train" in label_parts or "training" in label_parts:
        return "train"

    normalized = path.replace("\\", "/").lower()
    components = [part for part in normalized.split("/") if part]
    basename = os.path.basename(normalized)
    basename_parts = re.split(r"[_.\-\s]+", os.path.splitext(basename)[0])

    if feature_csv_score(path) == 0:
        if "test" in basename_parts or "testing" in basename_parts:
            return "test"
        if "train" in basename_parts or "training" in basename_parts:
            return "train"

    # Prefer the generated output directory over the filename: test feature
    # folders can contain files whose names mention train (for example labels_train_aug.csv).
    for component in reversed(components[:-1]):
        parts = re.split(r"[_.\-\s]+", component)
        if "test" in parts or "testing" in parts:
            return "test"
        if "train" in parts or "training" in parts:
            return "train"
    if "test" in basename_parts or "testing" in basename_parts:
        return "test"
    if "train" in basename_parts or "training" in basename_parts:
        return "train"

    text = f"{label} {basename}".lower()
    if "test" in text:
        return "test"
    if "train" in text:
        return "train"
    return None


def select_train_test_feature_paths(paths: Iterable[str]) -> tuple[str | None, str | None]:
    """Select train/test feature CSVs from collected generated paths."""
    paths = prefer_generated_feature_paths(paths)
    train_path = None
    test_path = None

    for path in paths:
        role = infer_dataset_role(path)
        if role == "train" and train_path is None:
            train_path = path
        elif role == "test" and test_path is None:
            test_path = path

    return train_path, test_path


def select_train_test_source_paths(paths: Iterable[str]) -> tuple[str | None, str | None]:
    """Select train/test source CSVs, preferring non-generated label/image tables."""
    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path or _is_placeholder(path) or path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    source_paths = [path for path in unique_paths if feature_csv_score(path) == 0]
    candidates = source_paths or unique_paths

    train_path = None
    test_path = None
    for path in candidates:
        role = infer_dataset_role(path)
        if role == "train" and train_path is None:
            train_path = path
        elif role == "test" and test_path is None:
            test_path = path

    return train_path, test_path


def resolve_feature_csv_argument(current_path: str, available_paths: Iterable[str]) -> str:
    """Replace a missing or guessed feature CSV with a matching generated path."""
    if not current_path:
        return current_path

    available = prefer_generated_feature_paths(available_paths)
    if not available:
        return current_path

    current_score = feature_csv_score(current_path)
    available_has_generated = any(feature_csv_score(path) > 0 for path in available)
    if os.path.exists(current_path) and (current_score > 0 or not available_has_generated):
        return current_path

    current_norm = current_path.replace("\\", "/").lower()
    current_dir = os.path.dirname(current_norm)
    current_role = infer_dataset_role(current_path)

    for candidate in available:
        candidate_norm = candidate.replace("\\", "/").lower()
        if current_dir and os.path.dirname(candidate_norm) == current_dir:
            return candidate

    if current_role:
        for candidate in available:
            if infer_dataset_role(candidate) == current_role:
                return candidate
        if available_has_generated:
            return current_path

    current_stem = os.path.splitext(os.path.basename(current_norm))[0]
    current_stem = current_stem.replace("_summary", "")
    for candidate in available:
        candidate_stem = os.path.splitext(os.path.basename(candidate.lower()))[0]
        if current_stem and current_stem in candidate_stem:
            return candidate

    if len(available) == 1:
        return available[0]

    return current_path


def resolve_source_csv_argument(current_path: str, available_paths: Iterable[str]) -> str:
    """Resolve an extraction input without ever substituting a feature table.

    Feature extraction tools need source metadata containing ImagePath/MaskPath.
    A generated feature CSV is a valid downstream classification input, but it is
    never a compatible replacement for such a source table.
    """
    if not current_path:
        return current_path

    source_paths = [
        path
        for path in dict.fromkeys(available_paths)
        if path and not _is_placeholder(path) and feature_csv_score(path) == 0
    ]

    # A real non-feature source supplied for this call is authoritative even when
    # conversational history contains generated LDSE/MEAS feature files.
    if feature_csv_score(current_path) == 0 and os.path.exists(current_path):
        return current_path
    if not source_paths:
        return current_path

    current_role = infer_dataset_role(current_path)
    if current_role:
        for candidate in source_paths:
            if infer_dataset_role(candidate) == current_role:
                return candidate

    if feature_csv_score(current_path) == 0 and current_path in source_paths:
        return current_path
    if len(source_paths) == 1:
        return source_paths[0]
    return current_path


def ensure_quality_output_dir(
    output_dir: str,
    feature_csv: str = "",
    used_dirs: set[str] | None = None,
) -> str:
    """Avoid overwriting quality reports when a plan reuses one output dir."""
    if not output_dir:
        return output_dir

    used_dirs = used_dirs if used_dirs is not None else set()
    normalized_used = {os.path.normcase(os.path.normpath(path)) for path in used_dirs}
    normalized_output = os.path.normcase(os.path.normpath(output_dir))
    exists_with_files = os.path.isdir(output_dir) and bool(os.listdir(output_dir))

    if normalized_output not in normalized_used and not exists_with_files:
        return output_dir

    role = infer_dataset_role(feature_csv) or infer_dataset_role(output_dir)
    suffix = role.capitalize() if role else "Run"
    candidate = f"{output_dir}_{suffix}"
    index = 2

    while (
        os.path.normcase(os.path.normpath(candidate)) in normalized_used
        or (os.path.isdir(candidate) and bool(os.listdir(candidate)))
    ):
        candidate = f"{output_dir}_{suffix}_{index}"
        index += 1

    return candidate
