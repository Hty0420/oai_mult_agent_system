from __future__ import annotations

from agents.dataset_profiler import profile_kl_dataset
from agents.task_contracts import (
    infer_goal_spec,
    infer_task_spec,
    materialize_biomarker_plan,
    required_agent_for_task,
)


QUERY = "自动完成 KL 0–4 横断面关联的 MRI 候选生物标志物发现"


def test_biomarker_goal_takes_precedence_over_kl_classification() -> None:
    goal = infer_goal_spec(QUERY)
    spec = infer_task_spec(QUERY)

    assert goal["goal_type"] == "BIOMARKER_DISCOVERY"
    assert goal["objective_variant"] == "cross_sectional_ordinal_kl_association"
    assert goal["capability_status"] == "supported"
    assert spec["feature_type"] == "CartilageThickness"
    assert spec["cartilage_operation"] == "batch_features"
    assert spec["requires_classification"] is False
    assert spec["requires_biomarker_discovery"] is True
    assert spec["requires_biomarker_verification"] is True
    assert spec["requires_biomarker_reporting"] is True
    assert required_agent_for_task(spec) == "ORCHESTRATOR_AGENT"


def test_default_dataset_and_fixed_four_role_plan_are_materialized() -> None:
    profile = profile_kl_dataset(QUERY)
    plan = materialize_biomarker_plan(
        dataset_profile=profile,
        current_input=QUERY,
    )

    assert profile["status"] == "ready"
    assert profile["source_files"]["train"]["row_count"] == 404
    assert profile["source_files"]["test"]["row_count"] == 103
    assert [step["agent"] for step in plan["steps"]] == [
        "FEATURE_EXTRACTION_AGENT",
        "BIOMARKER_DISCOVERY_AGENT",
        "BIOMARKER_VERIFIER_AGENT",
        "BIOMARKER_REPORTER_AGENT",
    ]
    assert "labels_train.csv" in plan["steps"][0]["input_override"]
    assert "labels_test.csv" in plan["steps"][0]["input_override"]
