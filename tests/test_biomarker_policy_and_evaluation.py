from __future__ import annotations

import pytest

from agents.task_contracts import infer_task_spec
from agents.tool_policy import ToolPermissionError, ensure_tool_allowed
from db_utils.task_evaluator import evaluate_task_run


def test_agent_tool_permissions_are_fail_closed() -> None:
    ensure_tool_allowed("BIOMARKER_DISCOVERY_AGENT", "run_biomarker_discovery")
    ensure_tool_allowed("BIOMARKER_VERIFIER_AGENT", "audit_biomarker_discovery")
    with pytest.raises(ToolPermissionError):
        ensure_tool_allowed("BIOMARKER_DISCOVERY_AGENT", "audit_biomarker_discovery")
    with pytest.raises(ToolPermissionError):
        ensure_tool_allowed(None, "run_biomarker_discovery")


def test_valid_negative_scientific_outcome_passes_task_gates() -> None:
    spec = infer_task_spec("完成 MRI 生物标志物发现")
    run_id = "run-negative"
    steps = [
        {
            "step": 1,
            "agent": "FEATURE_EXTRACTION_AGENT",
            "status": "success",
            "tool_calls": [
                "extract_cartilage_thickness_features",
                "extract_cartilage_thickness_features",
            ],
        },
        {
            "step": 2,
            "agent": "BIOMARKER_DISCOVERY_AGENT",
            "status": "success",
            "tool_calls": ["run_biomarker_discovery"],
        },
        {
            "step": 3,
            "agent": "BIOMARKER_VERIFIER_AGENT",
            "status": "success",
            "tool_calls": ["audit_biomarker_discovery"],
        },
        {
            "step": 4,
            "agent": "BIOMARKER_REPORTER_AGENT",
            "status": "success",
            "tool_calls": [],
        },
    ]
    artifacts = [
        {
            "run_id": run_id,
            "feature_type": "CartilageThickness",
            "dataset_role": "train",
            "path": "train_features.csv",
        },
        {
            "run_id": run_id,
            "feature_type": "CartilageThickness",
            "dataset_role": "test",
            "path": "test_features.csv",
        },
    ]
    report = evaluate_task_run(
        trace_id=run_id,
        run_id=run_id,
        task_spec=spec,
        routed_agent="ORCHESTRATOR_AGENT",
        step_records=steps,
        current_artifacts=artifacts,
        overall_success=True,
        final_output="科学结局：completed_no_confirmed_biomarker",
    )

    assert report["task_type"] == "biomarker_discovery"
    assert report["task_success"] is True
    assert all(gate["status"] == "pass" for gate in report["gates"].values())
