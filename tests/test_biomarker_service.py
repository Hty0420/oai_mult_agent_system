from __future__ import annotations

import json
import hashlib
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from algorithm.biomarker_discovery.service import (
    BiomarkerDiscoveryError,
    audit_biomarker_discovery,
    run_biomarker_discovery,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_frame(prefix: str, count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grade = np.resize(np.arange(5, dtype=np.int64), count)
    rng.shuffle(grade)
    return pd.DataFrame(
        {
            "CaseID": [f"{prefix}-{index:04d}" for index in range(count)],
            "KLGrade": grade,
            "Age": rng.normal(62.0, 7.0, count),
            "Gender": rng.choice(["F", "M"], count),
            "BMI": rng.normal(28.0, 3.5, count),
            "KneeSide": rng.choice(["L", "R"], count),
            "MN3D_signal_mean": 0.8 * grade + rng.normal(0.0, 0.8, count),
            "MN3D_noise_mean": rng.normal(size=count),
        }
    )


def test_discovery_confirmation_audit_and_idempotent_retry() -> None:
    temporary_root = PROJECT_ROOT / "tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as raw_directory:
        directory = Path(raw_directory)
        train_path = directory / "train_features.csv"
        test_path = directory / "test_features.csv"
        _synthetic_frame("TRAIN", 200, 10).to_csv(train_path, index=False)
        _synthetic_frame("TEST", 100, 20).to_csv(test_path, index=False)
        output_dir = directory / "analysis"
        progress_events: list[tuple[float, float, str]] = []

        result = run_biomarker_discovery(
            str(train_path),
            str(test_path),
            str(output_dir),
            bootstrap_iterations=10,
            bootstrap_support_threshold=0.5,
            progress_callback=lambda progress, total, message: progress_events.append(
                (progress, total, message)
            ),
        )
        assert result["status"] == "success"
        assert result["candidate_count"] >= 1
        assert result["confirmed_count"] >= 1
        assert result["scientific_outcome"] == "completed_confirmed_candidates"
        assert progress_events[0][0] == 0.0
        assert progress_events[-1][0] == 100.0
        assert any("训练集特征扫描" in event[2] for event in progress_events)
        assert all(
            left[0] <= right[0]
            for left, right in zip(progress_events, progress_events[1:])
        )

        retry = run_biomarker_discovery(
            str(train_path),
            str(test_path),
            str(output_dir),
            bootstrap_iterations=10,
            bootstrap_support_threshold=0.5,
        )
        assert retry["idempotent_reuse"] is True
        assert retry["manifest_path"] == result["manifest_path"]

        audit = audit_biomarker_discovery(
            result["manifest_path"],
            str(directory / "audit"),
        )
        assert audit["status"] == "success"
        assert audit["valid"] is True
        assert audit["dataset_summary"]["train_labeled_rows"] == 200
        assert len(audit["confirmed_candidates"]) == audit["confirmed_count"]
        assert Path(audit["source_report_path"]).is_file()
        assert all(gate["status"] == "pass" for gate in audit["gates"].values())
        audit_retry = audit_biomarker_discovery(
            result["manifest_path"],
            str(directory / "audit"),
        )
        assert audit_retry["idempotent_reuse"] is True


def test_tampered_artifact_fails_independent_audit() -> None:
    temporary_root = PROJECT_ROOT / "tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as raw_directory:
        directory = Path(raw_directory)
        train_path = directory / "train_features.csv"
        test_path = directory / "test_features.csv"
        _synthetic_frame("TRAIN", 160, 30).to_csv(train_path, index=False)
        _synthetic_frame("TEST", 80, 40).to_csv(test_path, index=False)
        result = run_biomarker_discovery(
            str(train_path),
            str(test_path),
            str(directory / "analysis"),
            bootstrap_iterations=10,
            bootstrap_support_threshold=0.5,
        )
        confirmation = Path(result["confirmation_results_path"])
        confirmation.write_text(
            confirmation.read_text(encoding="utf-8-sig") + "\n",
            encoding="utf-8-sig",
        )

        audit = audit_biomarker_discovery(
            result["manifest_path"],
            str(directory / "audit"),
        )
        assert audit["status"] == "failed"
        assert audit["valid"] is False
        assert audit["gates"]["artifact_integrity"]["status"] == "fail"


def test_forged_confirmation_status_is_recomputed_not_trusted() -> None:
    temporary_root = PROJECT_ROOT / "tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as raw_directory:
        directory = Path(raw_directory)
        train_path = directory / "train_features.csv"
        test_path = directory / "test_features.csv"
        _synthetic_frame("TRAIN", 180, 31).to_csv(train_path, index=False)
        _synthetic_frame("TEST", 90, 41).to_csv(test_path, index=False)
        result = run_biomarker_discovery(
            str(train_path),
            str(test_path),
            str(directory / "analysis"),
            bootstrap_iterations=10,
            bootstrap_support_threshold=0.5,
        )
        assert result["confirmed_count"] >= 1

        confirmation_path = Path(result["confirmation_results_path"])
        confirmation = pd.read_csv(confirmation_path)
        confirmation["confirmation_status"] = "not_confirmed"
        confirmation.to_csv(confirmation_path, index=False, encoding="utf-8-sig")

        manifest_path = Path(result["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["confirmed_count"] = 0
        manifest["artifact_hashes"][confirmation_path.name] = hashlib.sha256(
            confirmation_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        audit = audit_biomarker_discovery(
            str(manifest_path),
            str(directory / "audit"),
        )
        assert audit["gates"]["artifact_integrity"]["status"] == "pass"
        assert audit["gates"]["reported_counts_reconcile"]["status"] == "pass"
        assert audit["gates"]["confirmation_decisions_recomputed"]["status"] == "fail"
        assert audit["valid"] is False


def test_train_test_case_overlap_is_blocked() -> None:
    temporary_root = PROJECT_ROOT / "tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as raw_directory:
        directory = Path(raw_directory)
        train = _synthetic_frame("TRAIN", 100, 50)
        test = _synthetic_frame("TEST", 60, 60)
        test.loc[0, "CaseID"] = train.loc[0, "CaseID"]
        train_path = directory / "train_features.csv"
        test_path = directory / "test_features.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        with pytest.raises(BiomarkerDiscoveryError, match="重叠"):
            run_biomarker_discovery(
                str(train_path),
                str(test_path),
                str(directory / "analysis"),
                bootstrap_iterations=10,
            )


def test_no_candidate_is_a_valid_audited_scientific_outcome() -> None:
    temporary_root = PROJECT_ROOT / "tmp"
    temporary_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as raw_directory:
        directory = Path(raw_directory)
        rng = np.random.default_rng(70)
        train = _synthetic_frame("TRAIN", 120, 71).drop(columns=["MN3D_signal_mean"])
        test = _synthetic_frame("TEST", 70, 72).drop(columns=["MN3D_signal_mean"])
        train["MN3D_noise_mean"] = rng.normal(size=len(train))
        test["MN3D_noise_mean"] = rng.normal(size=len(test))
        train_path = directory / "train_features.csv"
        test_path = directory / "test_features.csv"
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        result = run_biomarker_discovery(
            str(train_path),
            str(test_path),
            str(directory / "analysis"),
            fdr_alpha=1e-6,
            bootstrap_iterations=10,
        )
        assert result["candidate_count"] == 0
        assert result["confirmed_count"] == 0
        assert result["scientific_outcome"] == "completed_no_discovery_candidates"

        audit = audit_biomarker_discovery(
            result["manifest_path"],
            str(directory / "audit"),
        )
        assert audit["valid"] is True
