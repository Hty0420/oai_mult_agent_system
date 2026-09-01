from __future__ import annotations

from agents.biomarker_summary import build_biomarker_user_summary


def test_verified_positive_result_is_rendered_as_natural_language() -> None:
    summary = build_biomarker_user_summary(
        {
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
                    "test_odds_ratio_per_sd": 0.79,
                    "test_holm_p_value": 0.03,
                }
            ],
            "source_report_path": "D:/workspace/biomarker_report.md",
            "report_path": "D:/workspace/biomarker_audit.md",
        }
    )

    assert "任务已完成，且独立审计通过" in summary
    assert "训练集：输入 404 例" in summary
    assert "MN3D_cMT_mean" in summary
    assert "与更低 KL 等级相关" in summary
    assert "Holm P=0.030" in summary
    assert "不能据此推断因果关系" in summary


def test_verified_negative_result_is_explained_without_overclaiming() -> None:
    summary = build_biomarker_user_summary(
        {
            "valid": True,
            "scientific_outcome": "completed_no_confirmed_biomarker",
            "candidate_count": 3,
            "confirmed_count": 0,
            "dataset_summary": {},
            "confirmed_candidates": [],
        }
    )

    assert "没有候选通过冻结测试集" in summary
    assert "本次没有可报告为“已确认”" in summary
