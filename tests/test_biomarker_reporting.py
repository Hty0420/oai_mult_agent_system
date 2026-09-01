from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agents.biomarker_reporter_agent import BiomarkerReporterAgent
from agents.biomarker_reporting import (
    build_biomarker_reporting_packet,
    validate_biomarker_narrative,
)


def _valid_audit() -> dict:
    return {
        "valid": True,
        "scientific_outcome": "completed_confirmed_candidates",
        "candidate_count": 2,
        "confirmed_count": 1,
        "dataset_summary": {
            "train_input_rows": 404,
            "train_labeled_rows": 383,
            "train_unlabeled_excluded": 21,
            "test_input_rows": 103,
            "test_labeled_rows": 98,
            "test_unlabeled_excluded": 5,
            "eligible_feature_count": 56,
            "covariates_used": ["Age", "Gender=2", "BMI"],
            "covariates_unavailable": ["KneeSide:constant"],
        },
        "confirmed_candidates": [
            {
                "feature": "MN3D_cMT_mean",
                "direction": "higher_feature_lower_KL",
                "train_odds_ratio_per_sd": 0.72,
                "train_q_value": 0.01,
                "bootstrap_directional_support": 0.82,
                "test_odds_ratio_per_sd": 0.79,
                "test_holm_p_value": 0.03,
            }
        ],
        "source_report_path": "D:/workspace/biomarker_report.md",
        "report_path": "D:/workspace/biomarker_audit.md",
    }


def _valid_narrative() -> str:
    return (
        "本次结果最值得关注的是，训练阶段冻结的2个候选中，最终有1个在独立测试集中得到确认。"
        "训练集原始输入404例，其中383例有KL 0–4有效标签，21例未标注病例被排除；"
        "测试集输入103例，实际用于确认98例，另有5例因未标注排除，所以这些排除病例没有参与显著性计算。"
        "在56个符合质控条件的MRI特征中，MN3D_cMT_mean表现为特征越高与更低 KL 等级相关。"
        "它在训练集的OR/SD=0.720，这是每增加1个训练集标准差对应的效应大小；"
        "BH q=0.010表示经过FDR多重比较校正后仍达到发现标准。"
        "bootstrap方向稳定性=0.820，说明重采样时效应方向保持一致的程度；"
        "冻结后转入测试集，其OR/SD=0.790，Holm P=0.030，后者是测试确认阶段控制家族错误率后的显著性。"
        "这些数字共同支持KL横断面关联，但不代表因果，也不能据此预测未来进展、进行个体诊断或认定临床替代终点。"
    )


class _FakeLLM:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)

    def invoke(self, _messages):
        return SimpleNamespace(content=self.replies.pop(0))


def test_reporting_packet_contains_only_verified_display_facts() -> None:
    packet = build_biomarker_reporting_packet(_valid_audit())

    assert packet["train_labeled_rows"] == 383
    assert packet["confirmed_count"] == 1
    assert packet["confirmed_candidates"][0]["train_or_per_sd"] == "0.720"
    assert packet["confirmed_candidates"][0]["bootstrap_directional_support"] == "0.820"


def test_narrative_validator_requires_numbers_explanations_and_boundaries() -> None:
    packet = build_biomarker_reporting_packet(_valid_audit())

    assert validate_biomarker_narrative(_valid_narrative(), packet) == []

    invalid = _valid_narrative().replace("0.790", "0.990")
    errors = validate_biomarker_narrative(invalid, packet)
    assert any(error.startswith("missing_verified_literals:") for error in errors)
    assert any(error.startswith("unsupported_numeric_literals:") for error in errors)


def test_reporter_rewrites_failed_draft_and_returns_verified_natural_text() -> None:
    config = SimpleNamespace(
        biomarker_reporter_agent=SimpleNamespace(
            llm=_FakeLLM(["数字看起来不错。", _valid_narrative()])
        )
    )
    agent = BiomarkerReporterAgent(config)

    result = asyncio.run(
        agent.process(
            {
                "task_run_id": "reporting-run-001",
                "biomarker_audit": _valid_audit(),
            }
        )
    )

    assert result["success"] is True
    assert result["biomarker_reporting"]["mode"] == "verified_llm_narrative"
    assert result["biomarker_reporting"]["attempts"] == 2
    assert "MN3D_cMT_mean" in result["output"]
    assert "完整分析报告" in result["output"]


def test_run_dependent_style_instruction_is_not_single_fixed_wording() -> None:
    styles = {
        BiomarkerReporterAgent._style_for_run({"task_run_id": f"run-{index}"})
        for index in range(20)
    }
    assert len(styles) > 1
