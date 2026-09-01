"""Experiment results ETL — parse classification artifacts into the database.

LASSO runs produce ``classification_run_manifest.json`` (atomic JSON with all
12 patient-level metrics, input SHA-256, library versions).  Filter runs
produce only ``metrics_summary.txt`` (plain text).  This module handles both.

Usage as a library::

    from db_utils.experiment_etl import scan_and_load
    result = scan_and_load("src")  # walks src/ for manifests

Usage as a CLI::

    python -m db_utils.experiment_etl --root src
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

from db_utils.trajectory import (
    build_artifact_record,
    build_experiment_evaluations,
    build_trajectory_event,
    current_policy_id,
)
from db_utils.scientific_quality import evaluate_classification_manifest


CLASSIFICATION_TOOLS = frozenset(
    {"koa_lasso_classification", "koa_filter_classification"}
)
DEFAULT_CLASSIFICATION_OUTPUT_DIRS = {
    "koa_lasso_classification": "KOA_LASSO_Results",
    "koa_filter_classification": "KOA_Filter_Results",
}


# ---------------------------------------------------------------------------
# Feature-type inference (mirrors classification_agent._infer_feature_type)
# ---------------------------------------------------------------------------

def _infer_feature_type(text: str) -> str:
    """Guess the feature type from a path or directory name."""
    upper = text.upper()
    if "LDSE" in upper:
        return "LDSE"
    if "MEAS" in upper:
        return "MEAS"
    if "RADIOMICS" in upper or "RADIOLOGY" in upper:
        return "Radiomics"
    if "COMB" in upper:
        return "Combined"
    return "unknown"


# ---------------------------------------------------------------------------
# LASSO manifest parser
# ---------------------------------------------------------------------------

def parse_lasso_manifest(manifest_path: str) -> Optional[dict]:
    """Read a ``classification_run_manifest.json`` and return a row dict
    suitable for ``experiment_results``, or *None* if the file is unreadable.
    """
    path = Path(manifest_path)
    if not path.is_file():
        return None

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # Validate minimal expected structure
    metrics = manifest.get("metrics")
    inputs = manifest.get("inputs")
    versions = manifest.get("versions")
    if not isinstance(metrics, dict) or not isinstance(inputs, dict):
        return None

    train_info = inputs.get("train", {})
    test_info = inputs.get("test", {})

    # Suffix from filename: "classification_run_manifest.json" → ""
    # "classification_run_manifest_regionM.json" → "_regionM"
    basename = path.name
    suffix = ""
    m = re.match(r"classification_run_manifest(_region\w*)?\.json", basename)
    if m and m.group(1):
        suffix = m.group(1)

    output_dir = str(path.parent.resolve())
    train_path = train_info.get("path", "")
    feature_type = _infer_feature_type(train_path)

    row: dict[str, Any] = {
        "output_dir": output_dir,
        "suffix": suffix,
        "feature_type": feature_type,
        "classification_method": "LASSO",
        "evaluation_split": "test",
        "policy_id": current_policy_id(),
        "train_path": train_path,
        "test_path": test_info.get("path", ""),
        "train_sha256": train_info.get("sha256", ""),
        "test_sha256": test_info.get("sha256", ""),
        "NumPatients": metrics.get("NumPatients"),
        "AUC": metrics.get("AUC"),
        "PR_AUC": metrics.get("PR_AUC"),
        "Accuracy": metrics.get("Accuracy"),
        "Precision": metrics.get("Precision"),
        "Sensitivity": metrics.get("Sensitivity"),
        "Specificity": metrics.get("Specificity"),
        "F1": metrics.get("F1"),
        "TN": metrics.get("TN"),
        "FP": metrics.get("FP"),
        "FN": metrics.get("FN"),
        "TP": metrics.get("TP"),
        "selected_feature_count": manifest.get("selected_feature_count"),
        "final_c": manifest.get("final_c"),
        "random_seed": manifest.get("random_seed"),
        "python_version": versions.get("python", "") if versions else "",
        "numpy_version": versions.get("numpy", "") if versions else "",
        "pandas_version": versions.get("pandas", "") if versions else "",
        "sklearn_version": versions.get("scikit_learn", "") if versions else "",
        "manifest_sha256": _sha256_file(path),
        "scientific_quality": evaluate_classification_manifest(path),
    }

    # Counts
    counts = manifest.get("counts", {})
    for role in ("train", "test"):
        c = counts.get(role, {})
        if isinstance(c, dict):
            row[f"{role}_rows"] = c.get("rows")
            row[f"{role}_unique_patients"] = c.get("unique_patients")

    test_class_counts = counts.get("test", {}).get("class_counts", {})
    if isinstance(test_class_counts, dict) and test_class_counts:
        row["class_count"] = len(test_class_counts)
        row["task_type"] = (
            "binary_kl" if len(test_class_counts) == 2 else "ordinal_kl"
        )

    return row


# ---------------------------------------------------------------------------
# Filter metrics parser (best-effort, plain-text only)
# ---------------------------------------------------------------------------

def parse_filter_metrics_summary(txt_path: str) -> Optional[dict]:
    """Parse a Filter ``metrics_summary.txt``.

    Returns a dict keyed by method name (e.g. ``"Chi-2 (卡方检验)"``) with
    sub-keys ``auc``, ``precision``, ``recall``, or *None* on failure.
    """
    path = Path(txt_path)
    if not path.is_file():
        return None

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    results: dict[str, dict[str, float]] = {}
    # Lines like: "Chi-2 (卡方检验)            : AUC=0.7234, Precision=0.6812, Recall=0.7521"
    line_re = re.compile(
        r"^(.+?)\s*:\s*AUC\s*=\s*([\d.]+)\s*,\s*Precision\s*=\s*([\d.]+)\s*,\s*Recall\s*=\s*([\d.]+)"
    )
    for line in text.splitlines():
        m = line_re.match(line.strip())
        if m:
            results[m.group(1).strip()] = {
                "auc": float(m.group(2)),
                "precision": float(m.group(3)),
                "recall": float(m.group(4)),
            }
    return results if results else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's contents."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def find_manifests(root_dir: str) -> list[Path]:
    """Yield all ``classification_run_manifest*.json`` paths under *root_dir*,
    skipping temp files.
    """
    root = Path(root_dir)
    manifests: list[Path] = []
    for p in root.rglob("classification_run_manifest*.json"):
        if p.name.startswith(".") or p.suffix == ".tmp":
            continue
        if p.is_file():
            manifests.append(p)
    return manifests


def find_filter_summaries(root_dir: str) -> list[Path]:
    """Find Filter-classification metric summaries without matching other reports."""
    root = Path(root_dir)
    return [
        path
        for path in root.rglob("metrics_summary.txt")
        if path.is_file() and "FILTER_CLASSIFICATION" in str(path.parent).upper()
    ]


# ---------------------------------------------------------------------------
# Database loader
# ---------------------------------------------------------------------------

def _link_trace(
    row: dict,
    db: Any,  # DatabaseManager
) -> None:
    """Best-effort: look up the trace that produced this output directory."""
    try:
        output_dir = row.get("output_dir", "")
        if not output_dir:
            return
        basename = os.path.basename(output_dir)
        with db.connection.cursor() as cursor:
            cursor.execute(
                """SELECT trace_id, user_id, session_id
                   FROM tool_calls_log
                   WHERE tool_name IN ('koa_lasso_classification', 'koa_filter_classification')
                     AND tool_arguments LIKE %s
                   ORDER BY id DESC
                   LIMIT 1""",
                (f"%{basename}%",),
            )
            result = cursor.fetchone()
            if result:
                if result.get("trace_id"):
                    row["trace_id"] = result["trace_id"]
                if result.get("user_id"):
                    row["user_id"] = result["user_id"]
                if result.get("session_id"):
                    row["session_id"] = result["session_id"]
    except Exception:
        pass  # trace linking is optional


def scan_and_load(
    root_dir: str = ".",
    db: Any = None,
    trace_id: str = None,
    session_id: str = None,
    classification_tool: str = None,
) -> dict:
    """Scan *root_dir* for classification manifests and load into the database.

    Parameters:
        root_dir: Top-level directory to walk.
        db: Optional ``DatabaseManager``.  If *None*, one is created/closed
            internally.  If provided, the caller manages its lifecycle.
        trace_id: If set, stamp every imported row with this trace.
        session_id: If set, stamp every imported row with this session.
        classification_tool: When set by the online hook, ingest only the
            artifact type produced by that exact classification tool.

    Returns:
        ``{"loaded": n, "skipped": m}``
    """
    if classification_tool == "koa_lasso_classification":
        manifests = find_manifests(root_dir)
        filter_summaries = []
    elif classification_tool == "koa_filter_classification":
        manifests = []
        # This mode is only used after the caller has validated that root_dir
        # is the exact output directory returned by the Filter tool.  Custom
        # output directory names therefore do not need a naming convention.
        root = Path(root_dir)
        filter_summaries = [
            path for path in root.rglob("metrics_summary.txt") if path.is_file()
        ]
    else:
        manifests = find_manifests(root_dir)
        filter_summaries = find_filter_summaries(root_dir)
    close_db = db is None

    loaded = 0
    skipped = 0
    artifacts_registered = 0
    evaluations_loaded = 0

    try:
        if db is None:
            from db_utils.db_utils import DatabaseManager

            db = DatabaseManager()

        seen: set[tuple[str, str]] = set()  # (output_dir, suffix)

        for manifest_path in manifests:
            row = parse_lasso_manifest(str(manifest_path))
            if row is None:
                skipped += 1
                continue

            key = (row["output_dir"], row["suffix"])
            if key in seen:
                skipped += 1
                continue
            seen.add(key)

            # Attach caller-supplied context
            if trace_id:
                row["trace_id"] = trace_id
            if session_id:
                row["session_id"] = session_id

            # Best-effort trace linking
            if not row.get("trace_id"):
                _link_trace(row, db)

            try:
                db.save_experiment_result(row)
                loaded += 1
                if hasattr(db, "register_artifact") and hasattr(db, "save_evaluation_result"):
                    artifact = build_artifact_record(
                        path=manifest_path,
                        artifact_type="classification_manifest",
                        created_by="experiment_etl",
                        trace_id=row.get("trace_id"),
                        metadata={
                            "feature_type": row.get("feature_type"),
                            "classification_method": row.get("classification_method"),
                            "evaluation_split": row.get("evaluation_split"),
                        },
                    )
                    db.register_artifact(artifact)
                    artifacts_registered += 1

                    evaluations = build_experiment_evaluations(
                        row,
                        artifact_id=artifact["artifact_id"],
                    )
                    for evaluation in evaluations:
                        db.save_evaluation_result(evaluation)
                    evaluations_loaded += len(evaluations)

                    if row.get("trace_id") and hasattr(db, "save_trajectory_event"):
                        db.save_trajectory_event(
                            build_trajectory_event(
                                event_type="evaluation_recorded",
                                trace_id=row["trace_id"],
                                actor_type="evaluator",
                                actor_name="classification_manifest",
                                payload={
                                    "artifact_id": artifact["artifact_id"],
                                    "metric_count": sum(
                                        item["record_type"] == "metric"
                                        for item in evaluations
                                    ),
                                    "gate_count": sum(
                                        item["record_type"] == "gate"
                                        for item in evaluations
                                    ),
                                    "evaluator_version": evaluations[0]["evaluator_version"],
                                },
                            )
                        )
            except Exception as exc:
                print(f"[experiment_etl] 跳过 {manifest_path}: {exc}")
                skipped += 1

        for summary_path in filter_summaries:
            parsed = parse_filter_metrics_summary(str(summary_path))
            if not parsed:
                skipped += 1
                continue

            artifact = None
            for method_label, metrics in parsed.items():
                method_hash = hashlib.sha256(method_label.encode("utf-8")).hexdigest()[:12]
                row = {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "output_dir": str(summary_path.parent.resolve()),
                    "suffix": f"_filter_{method_hash}",
                    "feature_type": _infer_feature_type(str(summary_path.parent)),
                    "classification_method": "Filter",
                    "method_label": method_label,
                    "evaluation_split": "test",
                    "policy_id": current_policy_id(),
                    "task_type": "classification_unknown",
                    "AUC": metrics.get("auc"),
                    "Precision": metrics.get("precision"),
                    "Sensitivity": metrics.get("recall"),
                    "manifest_sha256": _sha256_file(summary_path),
                }
                if not row.get("trace_id"):
                    _link_trace(row, db)

                try:
                    db.save_experiment_result(row)
                    loaded += 1
                    if hasattr(db, "register_artifact") and hasattr(db, "save_evaluation_result"):
                        if artifact is None:
                            artifact = build_artifact_record(
                                path=summary_path,
                                artifact_type="filter_metrics_summary",
                                created_by="experiment_etl",
                                trace_id=row.get("trace_id"),
                                metadata={
                                    "classification_method": "Filter",
                                    "evaluation_split": "test",
                                    "structured_manifest": False,
                                },
                            )
                            db.register_artifact(artifact)
                            artifacts_registered += 1
                        evaluations = build_experiment_evaluations(
                            row,
                            artifact_id=artifact["artifact_id"],
                            evaluator_version="filter-text-v1",
                        )
                        for evaluation in evaluations:
                            db.save_evaluation_result(evaluation)
                        evaluations_loaded += len(evaluations)
                except Exception as exc:
                    print(f"[experiment_etl] Filter import skipped: {exc}")
                    skipped += 1

    finally:
        if close_db and db is not None:
            try:
                db.close()
            except Exception:
                pass

    return {
        "loaded": loaded,
        "skipped": skipped,
        "artifacts_registered": artifacts_registered,
        "evaluations_loaded": evaluations_loaded,
    }


def parse_classification_tool_result(result: Any) -> Optional[Mapping[str, Any]]:
    """Return a classification tool's structured result, never prose."""
    if isinstance(result, Mapping):
        parsed = result
    elif isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return None
    else:
        return None

    # Some MCP transports wrap a function return under ``result``.
    nested = parsed.get("result") if isinstance(parsed, Mapping) else None
    if isinstance(nested, Mapping) and "status" in nested:
        parsed = nested
    return parsed if isinstance(parsed, Mapping) else None


def _is_same_or_child(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def ingest_classification_result(
    tool_name: str,
    result: Any,
    *,
    db: Any,
    expected_output_dir: str | os.PathLike[str] | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Ingest artifacts from one successful classification tool invocation.

    The scan is deliberately bounded to the returned output directory, which
    must equal or be a descendant of the requested output directory.  This
    prevents an untrusted tool response from redirecting ETL to an unrelated
    filesystem tree.
    """
    if tool_name not in CLASSIFICATION_TOOLS:
        return {"status": "not_applicable", "reason": "unsupported_tool"}

    structured = parse_classification_tool_result(result)
    if structured is None:
        return {"status": "failed", "reason": "unstructured_tool_result"}
    if str(structured.get("status", "")).strip().lower() != "success":
        return {"status": "skipped", "reason": "tool_reported_failure"}

    raw_output_dir = structured.get("output_dir")
    if (
        not isinstance(raw_output_dir, (str, os.PathLike))
        or not os.fspath(raw_output_dir).strip()
    ):
        return {"status": "failed", "reason": "missing_output_dir"}

    actual_dir = Path(raw_output_dir).expanduser().resolve()
    requested_dir = Path(
        expected_output_dir
        or DEFAULT_CLASSIFICATION_OUTPUT_DIRS[tool_name]
    ).expanduser().resolve()
    if not _is_same_or_child(actual_dir, requested_dir):
        return {"status": "failed", "reason": "output_dir_out_of_scope"}
    if not actual_dir.is_dir():
        return {"status": "failed", "reason": "output_dir_not_found"}

    if tool_name == "koa_lasso_classification":
        candidates = find_manifests(str(actual_dir))
    else:
        candidates = [
            path
            for path in actual_dir.rglob("metrics_summary.txt")
            if path.is_file()
        ]
    if not candidates:
        return {"status": "failed", "reason": "required_evaluation_artifact_missing"}

    counts = scan_and_load(
        str(actual_dir),
        db=db,
        trace_id=trace_id,
        session_id=session_id,
        classification_tool=tool_name,
    )
    ingestion_status = "success" if counts["loaded"] > 0 else "failed"
    return {
        "status": ingestion_status,
        "reason": None if ingestion_status == "success" else "no_valid_records_loaded",
        "output_dir": str(actual_dir),
        "candidate_count": len(candidates),
        **counts,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan for classification manifests and load into experiment_results."
    )
    parser.add_argument(
        "--root", default=".", help="Root directory to scan (default: .)"
    )
    parser.add_argument(
        "--db-config",
        default=None,
        help="Path to db_config.json (default: src/db_utils/db_config.json)",
    )
    args = parser.parse_args()

    # Ensure src is on sys.path for direct invocation
    src_dir = os.path.join(os.path.dirname(__file__), "..")
    src_dir = os.path.abspath(src_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    result = scan_and_load(root_dir=args.root)
    print(f"ETL 完成: 导入 {result['loaded']} 条, 跳过 {result['skipped']} 条")


if __name__ == "__main__":
    _run_cli()
