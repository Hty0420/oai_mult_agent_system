"""Typed facts and validation for natural biomarker-result narration."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from .biomarker_summary import format_biomarker_number


def build_biomarker_reporting_packet(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only independently verified facts to the writer role."""
    if audit.get("valid") is not True:
        raise ValueError("Only a valid independent audit can be narrated")
    dataset = dict(audit.get("dataset_summary", {}))
    candidates: list[dict[str, Any]] = []
    for item in audit.get("confirmed_candidates", []):
        if not isinstance(item, Mapping):
            continue
        candidates.append(
            {
                "feature": str(item.get("feature") or ""),
                "direction": str(item.get("direction") or ""),
                "train_or_per_sd": format_biomarker_number(
                    item.get("train_odds_ratio_per_sd")
                ),
                "train_bh_q": format_biomarker_number(item.get("train_q_value")),
                "bootstrap_directional_support": format_biomarker_number(
                    item.get("bootstrap_directional_support")
                ),
                "test_or_per_sd": format_biomarker_number(
                    item.get("test_odds_ratio_per_sd")
                ),
                "test_holm_p": format_biomarker_number(
                    item.get("test_holm_p_value")
                ),
            }
        )
    return {
        "audit_valid": True,
        "scientific_outcome": str(audit.get("scientific_outcome") or ""),
        "train_input_rows": int(dataset.get("train_input_rows", 0) or 0),
        "train_labeled_rows": int(dataset.get("train_labeled_rows", 0) or 0),
        "train_unlabeled_excluded": int(
            dataset.get("train_unlabeled_excluded", 0) or 0
        ),
        "test_input_rows": int(dataset.get("test_input_rows", 0) or 0),
        "test_labeled_rows": int(dataset.get("test_labeled_rows", 0) or 0),
        "test_unlabeled_excluded": int(
            dataset.get("test_unlabeled_excluded", 0) or 0
        ),
        "eligible_feature_count": int(
            dataset.get("eligible_feature_count", 0) or 0
        ),
        "candidate_count": int(audit.get("candidate_count", 0) or 0),
        "confirmed_count": int(audit.get("confirmed_count", 0) or 0),
        "covariates_used": [str(value) for value in dataset.get("covariates_used", [])],
        "covariates_unavailable": [
            str(value) for value in dataset.get("covariates_unavailable", [])
        ],
        "confirmed_candidates": candidates,
        "source_report_path": str(audit.get("source_report_path") or ""),
        "audit_report_path": str(audit.get("report_path") or ""),
        "interpretation_contract": {
            "or_per_sd": (
                "每增加一个训练集标准差，处于更高 KL 等级的比例优势比；"
                "OR>1 表示更高特征值与更高 KL 相关，OR<1 表示与更低 KL 相关"
            ),
            "bh_q": "训练集全特征扫描经过 Benjamini-Hochberg FDR 校正后的显著性",
            "holm_p": "冻结候选在测试集进行 Holm 家族错误率校正后的确认显著性",
            "bootstrap_support": "训练集重采样中效应方向保持一致的比例",
            "scope": "横断面统计关联；不是因果、进展预测、个体诊断或临床替代终点",
        },
    }


def required_reporting_literals(packet: Mapping[str, Any]) -> list[str]:
    values = [
        packet.get("train_input_rows"),
        packet.get("train_labeled_rows"),
        packet.get("train_unlabeled_excluded"),
        packet.get("test_input_rows"),
        packet.get("test_labeled_rows"),
        packet.get("test_unlabeled_excluded"),
        packet.get("eligible_feature_count"),
        packet.get("candidate_count"),
        packet.get("confirmed_count"),
    ]
    literals = [str(value) for value in values]
    for candidate in packet.get("confirmed_candidates", []):
        literals.extend(
            str(candidate.get(key) or "")
            for key in (
                "feature",
                "train_or_per_sd",
                "train_bh_q",
                "bootstrap_directional_support",
                "test_or_per_sd",
                "test_holm_p",
            )
        )
    return list(dict.fromkeys(value for value in literals if value))


def validate_biomarker_narrative(
    narrative: str,
    packet: Mapping[str, Any],
) -> list[str]:
    """Check factual coverage and reject common clinical/causal overclaims."""
    text = str(narrative or "").strip()
    errors: list[str] = []
    if len(text) < 180:
        errors.append("summary_too_short")
    missing = [value for value in required_reporting_literals(packet) if value not in text]
    if missing:
        errors.append("missing_verified_literals:" + ",".join(missing))
    allowed_numbers = {
        value
        for value in required_reporting_literals(packet)
        if re.fullmatch(r"\d+(?:\.\d+)?(?:e[+-]?\d+)?", value, re.IGNORECASE)
    }
    # KL 0-4 and the OR/SD definition legitimately introduce these constants.
    allowed_numbers.update({"0", "1", "4"})
    observed_numbers = set(
        re.findall(
            r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![A-Za-z0-9_.])",
            text,
            flags=re.IGNORECASE,
        )
    )
    unsupported_numbers = sorted(observed_numbers - allowed_numbers)
    if unsupported_numbers:
        errors.append("unsupported_numeric_literals:" + ",".join(unsupported_numbers))
    if not any(term in text for term in ("横断面", "不能推断因果", "不代表因果")):
        errors.append("cross_sectional_boundary_missing")
    if packet.get("confirmed_candidates"):
        required_explanations = {
            "or_per_sd_not_explained": ("OR/SD", "优势比", "效应大小"),
            "bh_q_not_explained": ("BH q", "BH校正", "BH 校正", "FDR"),
            "bootstrap_not_explained": ("bootstrap", "Bootstrap", "重采样", "方向稳定"),
            "holm_p_not_explained": ("Holm P", "Holm校正", "Holm 校正"),
        }
        for error, terms in required_explanations.items():
            if not any(term in text for term in terms):
                errors.append(error)
    elif int(packet.get("confirmed_count", 0) or 0) == 0 and not any(
        term in text for term in ("没有确认", "未确认", "未通过测试集", "0 个确认")
    ):
        errors.append("negative_confirmation_outcome_not_explained")
    prohibited_patterns = (
        r"证明了.{0,12}因果",
        r"已经过临床验证",
        r"可用于.{0,8}(?:临床)?诊断",
        r"可以预测.{0,8}(?:未来|进展)",
        r"可作为.{0,8}替代终点",
    )
    for pattern in prohibited_patterns:
        if re.search(pattern, text):
            errors.append(f"prohibited_overclaim:{pattern}")
    for candidate in packet.get("confirmed_candidates", []):
        feature = str(candidate.get("feature") or "")
        if not feature:
            continue
        direction = str(candidate.get("direction") or "")
        segment = text[text.find(feature) :]
        if direction == "higher_feature_higher_KL" and not any(
            term in segment[:240] for term in ("更高 KL", "KL 等级升高", "较高 KL")
        ):
            errors.append(f"direction_not_explained:{feature}")
        if direction == "higher_feature_lower_KL" and not any(
            term in segment[:240] for term in ("更低 KL", "KL 等级降低", "较低 KL")
        ):
            errors.append(f"direction_not_explained:{feature}")
    return errors


def append_verified_report_links(text: str, packet: Mapping[str, Any]) -> str:
    links: list[str] = []
    for label, key in (
        ("完整分析报告", "source_report_path"),
        ("独立审计报告", "audit_report_path"),
    ):
        raw_path = str(packet.get(key) or "").strip()
        if not raw_path:
            continue
        path = str(Path(raw_path).resolve()).replace("\\", "/")
        links.append(f"[{label}]({path})")
    if not links:
        return text.strip()
    return text.rstrip() + "\n\n结果文件：" + " · ".join(links)
