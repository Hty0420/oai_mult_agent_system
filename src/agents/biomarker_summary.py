"""Evidence-grounded Chinese summaries for verified biomarker runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping


def format_biomarker_number(value: Any, digits: int = 3) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "不可用"
    if not math.isfinite(parsed):
        return "不可用"
    if parsed != 0.0 and (abs(parsed) < 0.001 or abs(parsed) >= 1000):
        return f"{parsed:.2e}"
    return f"{parsed:.{digits}f}"


def _file_link(label: str, value: Any) -> str | None:
    path = str(value or "").strip()
    if not path:
        return None
    normalized = str(Path(path).resolve()).replace("\\", "/")
    return f"[{label}]({normalized})"


def build_biomarker_user_summary(audit: Mapping[str, Any]) -> str:
    """Describe only fields emitted by a successful independent audit."""
    if audit.get("valid") is not True:
        return "生物标志物发现结果未通过独立审计，因此不能生成科学结果摘要。"

    dataset = dict(audit.get("dataset_summary", {}))
    candidate_count = int(audit.get("candidate_count", 0) or 0)
    confirmed_count = int(audit.get("confirmed_count", 0) or 0)
    scientific_outcome = str(audit.get("scientific_outcome") or "")
    confirmed = [
        dict(item)
        for item in audit.get("confirmed_candidates", [])
        if isinstance(item, Mapping)
    ]

    lines = [
        "## KL 0–4 MRI 候选生物标志物发现结果",
        "",
        "任务已完成，且独立审计通过。训练集只用于候选发现，测试集只用于冻结候选的确认；候选集合、多重检验结果和输出文件哈希均通过复核。",
        "",
        "### 数据与分析概况",
        "",
        (
            f"- 训练集：输入 {int(dataset.get('train_input_rows', 0) or 0)} 例，"
            f"纳入 KL 0–4 分析 {int(dataset.get('train_labeled_rows', 0) or 0)} 例，"
            f"排除未标注 {int(dataset.get('train_unlabeled_excluded', 0) or 0)} 例。"
        ),
        (
            f"- 测试集：输入 {int(dataset.get('test_input_rows', 0) or 0)} 例，"
            f"纳入确认 {int(dataset.get('test_labeled_rows', 0) or 0)} 例，"
            f"排除未标注 {int(dataset.get('test_unlabeled_excluded', 0) or 0)} 例。"
        ),
        f"- 完成有序关联检验的 MRI 特征：{int(dataset.get('eligible_feature_count', 0) or 0)} 个。",
        f"- 训练集冻结候选：{candidate_count} 个；测试集统计学确认：{confirmed_count} 个。",
    ]

    covariates = [str(item) for item in dataset.get("covariates_used", [])]
    unavailable = [str(item) for item in dataset.get("covariates_unavailable", [])]
    if covariates:
        lines.append(f"- 模型调整变量：{', '.join(covariates)}。")
    if unavailable:
        lines.append(f"- 未进入模型的预设变量：{', '.join(unavailable)}。")

    lines.extend(["", "### 主要结果", ""])
    if scientific_outcome == "completed_no_discovery_candidates" or candidate_count == 0:
        lines.append(
            "训练集中没有特征通过预先规定的 BH-FDR、相关性去冗余和稳定性流程，因此没有候选进入测试集。这是有效的阴性结果，不应通过放宽阈值重新筛选。"
        )
    elif confirmed_count == 0:
        lines.append(
            f"训练集发现并冻结了 {candidate_count} 个候选，但没有候选通过冻结测试集的方向一致性、bootstrap 稳定性和 Holm 校正确认。因此本次没有可报告为“已确认”的 MRI 候选生物标志物。"
        )
    else:
        lines.append(
            f"共有 {confirmed_count} 个候选在冻结测试集中得到统计学确认："
        )
        lines.append("")
        for index, item in enumerate(confirmed[:10], start=1):
            direction = (
                "特征值越高，与更高 KL 等级相关"
                if item.get("direction") == "higher_feature_higher_KL"
                else "特征值越高，与更低 KL 等级相关"
            )
            lines.append(
                f"{index}. `{item.get('feature', 'unknown')}`：{direction}；"
                f"训练集 OR/SD={format_biomarker_number(item.get('train_odds_ratio_per_sd'))}，"
                f"BH q={format_biomarker_number(item.get('train_q_value'))}；"
                f"测试集 OR/SD={format_biomarker_number(item.get('test_odds_ratio_per_sd'))}，"
                f"Holm P={format_biomarker_number(item.get('test_holm_p_value'))}。"
            )
        if len(confirmed) > 10:
            lines.append(f"另有 {len(confirmed) - 10} 个确认候选见完整报告。")

    lines.extend(
        [
            "",
            "### 如何解释",
            "",
            "这些结果表示 MRI 定量特征与当前 KL 0–4 严重程度存在横断面统计关联。它们仍是研究用途的候选标志物，不能据此推断因果关系、预测未来进展、进行个体诊断或作为临床替代终点。",
            "",
        ]
    )
    report_link = _file_link("完整分析报告", audit.get("source_report_path"))
    audit_link = _file_link("独立审计报告", audit.get("report_path"))
    links = [item for item in (report_link, audit_link) if item]
    if links:
        lines.extend(["### 结果文件", "", " · ".join(links)])
    return "\n".join(lines).strip()
