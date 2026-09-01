"""Deterministic task and artifact contracts for multi-agent execution.

Conversation history may help interpret references, but it is never an
authoritative source of executable artifacts.  These helpers deliberately have
no LLM, database, or LangGraph dependency so the control rules are testable.
"""

from __future__ import annotations

import os
import re
from typing import Any, Mapping


_REUSE_TERMS = (
    "沿用",
    "复用",
    "刚才",
    "使用刚才",
    "用刚才",
    "使用上次",
    "使用之前",
    "已有特征",
    "现有特征",
    "reuse",
    "previous",
    "last run",
)

_KL_GOAL_TERMS = (
    "kl分级",
    "kl 分级",
    "koa分级",
    "koa 分级",
    "膝骨关节炎分级",
    "kl grading",
    "koa grading",
)

_BIOMARKER_GOAL_TERMS = (
    "生物标志物",
    "影像标志物",
    "影像生物标志物",
    "biomarker",
    "imaging marker",
    "imaging biomarker",
)

_GOAL_COMPLETION_TERMS = (
    "完成",
    "自动",
    "构建",
    "建立",
    "实现",
    "perform",
    "complete",
    "build",
)

_ORDINAL_TERMS = (
    "0-4",
    "0–4",
    "五分类",
    "5分类",
    "多分类",
    "有序分类",
    "ordinal",
    "multiclass",
)

_DOMAIN_DEFAULT_GOALS = {
    "完成任务",
    "完成当前任务",
    "自动完成任务",
    "complete the task",
}

_CARTILAGE_VALIDATION_TERMS = (
    "校验", "检查", "验证", "审核", "质控",
    "是否", "能否", "可否", "能不能", "可不可以",
    "适合", "符合", "满足", "能做", "能用", "可用于",
    "validate", "check", "verify", "eligible", "suitable", "compatible",
)

_CARTILAGE_GENERATION_TERMS = (
    "计算", "测量", "提取", "量化", "定量", "统计", "生成", "导出", "执行",
    "重建", "建模", "构建",
    "calculate", "measure", "extract", "quantify", "compute", "export", "run",
    "reconstruct", "build", "model",
)

_CARTILAGE_EXPLICIT_OUTPUT_TERMS = (
    "计算", "提取", "量化", "定量", "统计", "生成", "导出", "执行",
    "重建", "建模", "构建",
    "calculate", "extract", "quantify", "compute", "export", "run",
    "reconstruct", "build", "model",
)

_CARTILAGE_BATCH_TERMS = (
    "批量", "数据集", "病例表", "csv", "batch", "cohort",
)

_CARTILAGE_MESH_TERMS = (
    "三维网格", "3d网格", "3d 网格", "表面网格", "网格模型",
    "三维表面", "三维模型", "3d mesh", "surface mesh", "3d model",
)


def infer_feature_type(text: str) -> str | None:
    upper = str(text or "").upper()
    if (
        "CARTILAGETHICKNESS" in upper
        or "CARTILAGE THICKNESS" in upper
        or "CARTILAGE_THICKNESS" in upper
        or "3D-MN" in upper
        or "3D_MN" in upper
        or "16亚区" in str(text or "")
        or "16 亚区" in str(text or "")
        or "亚区厚度" in str(text or "")
        or any(term in str(text or "").lower() for term in _CARTILAGE_MESH_TERMS)
    ):
        return "CartilageThickness"
    if "LDSE" in upper:
        return "LDSE"
    if "MEAS" in upper:
        return "MEAS"
    if "RADIOMICS" in upper or "RADIOLOGY" in upper or "影像组学" in text:
        return "Radiomics"
    if "软骨厚度" in str(text or ""):
        return "CartilageThickness"
    return None


def infer_cartilage_operation(text: str) -> str | None:
    """Classify cartilage work by behavior, independent of exact prompt wording."""
    raw = str(text or "")
    lower = raw.lower()
    if infer_feature_type(raw) != "CartilageThickness":
        return None

    batch_requested = any(term in lower for term in _CARTILAGE_BATCH_TERMS)
    generation_requested = any(
        term in lower for term in _CARTILAGE_GENERATION_TERMS
    )
    explicit_output_requested = any(
        term in lower for term in _CARTILAGE_EXPLICIT_OUTPUT_TERMS
    )
    validation_requested = any(
        term in lower for term in _CARTILAGE_VALIDATION_TERMS
    )

    if any(term in lower for term in _CARTILAGE_MESH_TERMS):
        return "mesh"

    if batch_requested and generation_requested:
        return "batch_features"
    # Questions about eligibility/QC remain validation requests even when the
    # phrase names the downstream "analysis" they are checking suitability for.
    if validation_requested and not explicit_output_requested:
        return "validation"
    return "analysis"


def infer_goal_spec(text: str) -> dict[str, Any]:
    """Describe the user's scientific goal separately from requested steps.

    The current tool surface implements binary KOA classification derived from
    ``KLGrade`` (grades below 2 versus grades 2 and above).  We expose that
    limitation explicitly so an autonomous planner never represents the
    available tools as a five-class ordinal grader.
    """
    raw = str(text or "")
    lower = raw.lower()
    feature_type = infer_feature_type(raw)
    classification_method = (
        "LASSO" if "lasso" in lower
        else "Filter" if "filter" in lower
        else None
    )
    biomarker_goal = any(term in lower for term in _BIOMARKER_GOAL_TERMS)
    if biomarker_goal:
        biomarker_feature_type = feature_type or "CartilageThickness"
        return {
            "schema_version": "1.0",
            "goal_type": "BIOMARKER_DISCOVERY",
            "goal_source": "explicit",
            "autonomy_mode": "auto",
            "objective_variant": "cross_sectional_ordinal_kl_association",
            "target_column": "KLGrade",
            "label_policy": "KLGrade 0-4 ordinal; KLGrade -1/缺失排除",
            "primary_metric": "held_out_confirmed_candidate_count",
            "secondary_metrics": [
                "training_bh_fdr_candidate_count",
                "bootstrap_directional_support",
                "held_out_holm_adjusted_p_value",
            ],
            "constraints": {
                "feature_type": biomarker_feature_type,
                "classification_method": None,
                "discovery_split": "train_only",
                "confirmation_split": "frozen_test_only",
                "effect_model": "proportional_odds_cumulative_logit",
            },
            "complete_goal_requested": True,
            "capability_status": "supported",
            "capability_limitations": [
                "V1 发现的是与 KL 0-4 横断面相关的候选 MRI 影像标志物，"
                "不能证明因果、疾病进展预测或临床替代终点有效性。",
                "无通过预设确认门槛的候选也是有效的阴性科学结局。",
            ],
        }
    explicit_kl_goal = any(term in lower for term in _KL_GOAL_TERMS) or (
        "kl" in lower
        and any(
            term in lower
            for term in ("分级", "分类", "grading", "classification")
        )
    )
    domain_default_goal = lower.strip() in _DOMAIN_DEFAULT_GOALS
    kl_goal = explicit_kl_goal or domain_default_goal
    wants_complete_goal = kl_goal and any(
        term in lower for term in _GOAL_COMPLETION_TERMS
    )
    ordinal_requested = kl_goal and any(term in lower for term in _ORDINAL_TERMS)

    if not kl_goal:
        autonomy_mode = "manual"
    elif feature_type and classification_method:
        autonomy_mode = "manual"
    elif feature_type or classification_method:
        autonomy_mode = "partially_constrained"
    else:
        autonomy_mode = "auto"

    return {
        "schema_version": "1.0",
        "goal_type": "KL_GRADING" if kl_goal else None,
        "goal_source": (
            "explicit" if explicit_kl_goal
            else "domain_default" if domain_default_goal
            else None
        ),
        "autonomy_mode": autonomy_mode,
        "objective_variant": (
            "ordinal_0_4" if ordinal_requested
            else "binary_koa_from_kl_threshold" if kl_goal
            else None
        ),
        "target_column": "KLGrade" if kl_goal else None,
        "label_policy": (
            "KLGrade 0-4 ordinal" if ordinal_requested
            else "KLGrade < 2 vs KLGrade >= 2" if kl_goal
            else None
        ),
        "primary_metric": (
            "quadratic_weighted_kappa" if ordinal_requested
            else "AUC" if kl_goal
            else None
        ),
        "secondary_metrics": (
            ["macro_f1", "balanced_accuracy"] if ordinal_requested
            else ["F1", "Sensitivity", "Specificity"] if kl_goal
            else []
        ),
        "constraints": {
            "feature_type": feature_type,
            "classification_method": classification_method,
        },
        "complete_goal_requested": wants_complete_goal,
        "capability_status": (
            "unsupported" if ordinal_requested
            else "supported" if kl_goal
            else "not_applicable"
        ),
        "capability_limitations": (
            [
                "当前分类工具只支持由 KLGrade 阈值派生的二分类，"
                "不支持 KLGrade 0-4 五分类。"
            ]
            if ordinal_requested
            else [
                "当前自动流水线执行 KLGrade < 2 与 KLGrade >= 2 的二分类，"
                "不是 KLGrade 0-4 五分类。"
            ]
            if kl_goal
            else []
        ),
    }


def infer_task_spec(text: str) -> dict[str, Any]:
    """Build a bounded task contract from explicit terms in the current input."""
    raw = str(text or "")
    lower = raw.lower()
    goal_spec = infer_goal_spec(raw)
    feature_type = infer_feature_type(raw)
    cartilage_operation = infer_cartilage_operation(raw)
    allow_history_reuse = any(term in lower for term in _REUSE_TERMS)
    if goal_spec.get("goal_type") == "BIOMARKER_DISCOVERY":
        return {
            "schema_version": "2.0",
            "goal_type": "BIOMARKER_DISCOVERY",
            "autonomy_mode": "auto",
            "objective_variant": "cross_sectional_ordinal_kl_association",
            "feature_type": feature_type or "CartilageThickness",
            "cartilage_operation": "batch_features",
            "requires_feature_extraction": True,
            "requires_output_artifact": True,
            "requires_classification": False,
            "classification_method": None,
            "requires_report_read": False,
            "requires_biomarker_discovery": True,
            "requires_biomarker_verification": True,
            "requires_biomarker_reporting": True,
            "allow_history_reuse": False,
            "artifact_reuse_policy": "deny",
            "goal_spec": goal_spec,
        }
    extraction_terms = ("提取" in raw or "extract" in lower)
    cartilage_analysis_terms = bool(
        feature_type == "CartilageThickness"
        and cartilage_operation is not None
    )
    explicit_extraction = bool(
        feature_type
        and (extraction_terms or cartilage_analysis_terms)
        and not allow_history_reuse
    )
    autonomous_goal = bool(
        goal_spec.get("goal_type") == "KL_GRADING"
        and goal_spec.get("autonomy_mode") in {"auto", "partially_constrained"}
        and goal_spec.get("capability_status") == "supported"
    )
    full_goal_pipeline = bool(
        goal_spec.get("goal_type") == "KL_GRADING"
        and goal_spec.get("complete_goal_requested")
        and not allow_history_reuse
    )
    requires_extraction = explicit_extraction or autonomous_goal or full_goal_pipeline
    classification_signal = bool(
        "lasso" in lower
        or "filter" in lower
        or "分类" in raw
        or "分级" in raw
        or "classification" in lower
    )
    classification_method = (
        "LASSO" if "lasso" in lower
        else "Filter" if "filter" in lower
        else None
    )
    requires_report = bool(
        ("读取" in raw or "查看" in raw or "read" in lower)
        and ("报告" in raw or "结果" in raw or "report" in lower)
    )
    read_only_reference = bool(
        requires_report
        and allow_history_reuse
        and not any(
            term in lower
            for term in ("使用lasso", "使用filter", "执行", "进行", "重新", "run ")
        )
    )
    requires_classification = (
        classification_signal
        or goal_spec.get("goal_type") == "KL_GRADING"
    ) and not read_only_reference
    requires_report = bool(
        requires_report
        or (
            goal_spec.get("goal_type") == "KL_GRADING"
            and goal_spec.get("complete_goal_requested")
        )
    )
    requires_output_artifact = bool(
        requires_extraction
        and not (
            feature_type == "CartilageThickness"
            and cartilage_operation == "validation"
        )
    )
    return {
        "schema_version": "2.0",
        "goal_type": goal_spec.get("goal_type"),
        "autonomy_mode": goal_spec.get("autonomy_mode"),
        "objective_variant": goal_spec.get("objective_variant"),
        "feature_type": feature_type,
        "cartilage_operation": cartilage_operation,
        "requires_feature_extraction": requires_extraction,
        "requires_output_artifact": requires_output_artifact,
        "requires_classification": requires_classification,
        "classification_method": classification_method,
        "requires_report_read": requires_report,
        "allow_history_reuse": allow_history_reuse,
        "artifact_reuse_policy": "explicit" if allow_history_reuse else "deny",
        "goal_spec": goal_spec,
    }


def apply_selected_pipeline(
    spec: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind an autonomous goal contract to one validated pipeline candidate."""
    selected = dict(spec or {})
    selected["feature_type"] = candidate.get("feature_type")
    selected["classification_method"] = candidate.get("classification_method")
    selected["model_recipe"] = candidate.get("model_recipe")
    selected["model_recipe_params"] = dict(
        candidate.get("validation_recommended_params") or {}
    )
    selected["requires_feature_extraction"] = True
    selected["requires_output_artifact"] = True
    selected["requires_classification"] = True
    selected["requires_report_read"] = True
    selected["allow_history_reuse"] = False
    selected["artifact_reuse_policy"] = "deny"
    selected["selected_pipeline_id"] = candidate.get("candidate_id")
    return selected


def required_agent_for_task(spec: Mapping[str, Any] | None) -> str | None:
    spec = spec or {}
    if spec.get("goal_type") == "BIOMARKER_DISCOVERY":
        return "ORCHESTRATOR_AGENT"
    if (
        spec.get("goal_type") == "KL_GRADING"
        and spec.get("autonomy_mode") in {"auto", "partially_constrained"}
    ):
        return "ORCHESTRATOR_AGENT"
    if spec.get("requires_feature_extraction") and spec.get("requires_classification"):
        return "ORCHESTRATOR_AGENT"
    if spec.get("requires_feature_extraction"):
        return "FEATURE_EXTRACTION_AGENT"
    if spec.get("requires_classification") or spec.get("requires_report_read"):
        return "CLASSIFICATION_AGENT"
    return None


def normalize_orchestrator_plan(
    plan: dict[str, Any],
    *,
    task_spec: Mapping[str, Any],
    current_input: str,
) -> dict[str, Any]:
    """Repair required steps and ordering without trusting planner prose."""
    raw_steps = plan.get("steps")
    steps = [dict(step) for step in raw_steps if isinstance(step, dict)] if isinstance(raw_steps, list) else []
    feature_type = task_spec.get("feature_type") or "requested"

    if task_spec.get("requires_feature_extraction"):
        feature_indexes = [
            index for index, step in enumerate(steps)
            if step.get("agent") == "FEATURE_EXTRACTION_AGENT"
        ]
        if feature_indexes:
            feature_step = steps.pop(feature_indexes[0])
        else:
            feature_step = {"agent": "FEATURE_EXTRACTION_AGENT"}
        if not any(term in current_input.lower() for term in ("增强", "augment")):
            steps = [
                step
                for step in steps
                if step.get("agent") != "FEATURE_EXTRACTION_AGENT"
            ]
        feature_step["task_description"] = f"为当前任务提取 {feature_type} 特征"
        # The current request, not conversational history, is authoritative.
        feature_step["input_override"] = current_input
        classification_index = next(
            (
                index for index, step in enumerate(steps)
                if step.get("agent") == "CLASSIFICATION_AGENT"
            ),
            len(steps),
        )
        steps.insert(classification_index, feature_step)

    if task_spec.get("requires_classification"):
        classification_steps = [
            step for step in steps
            if step.get("agent") == "CLASSIFICATION_AGENT"
        ]
        classification_step = (
            classification_steps[0]
            if classification_steps
            else {"agent": "CLASSIFICATION_AGENT"}
        )
        # ClassificationAgent performs classification and report reading in one
        # atomic step. Duplicate planner steps could otherwise read a stale report.
        steps = [
            step for step in steps
            if step.get("agent") != "CLASSIFICATION_AGENT"
        ]
        classification_step["task_description"] = (
            "仅使用当前任务产生的特征执行分类并按需读取报告"
        )
        classification_step["input_override"] = current_input
        steps.append(classification_step)

    normalized = dict(plan)
    normalized["steps"] = steps
    return normalized


def materialize_biomarker_plan(
    *,
    dataset_profile: Mapping[str, Any],
    current_input: str,
) -> dict[str, Any]:
    """Build the fixed V1 discovery DAG from profiled current-run inputs."""
    source_files = dataset_profile.get("source_files", {})
    train = source_files.get("train", {}) if isinstance(source_files, Mapping) else {}
    test = source_files.get("test", {}) if isinstance(source_files, Mapping) else {}
    train_path = str(train.get("path") or "") if isinstance(train, Mapping) else ""
    test_path = str(test.get("path") or "") if isinstance(test, Mapping) else ""
    extraction_input = (
        "批量提取训练集和测试集 CartilageThickness 软骨厚度特征，"
        "包括 3D-MN 全局指标和 Wirth/Eckstein 16 亚区指标。\n"
        f"train_csv_path: {train_path}\n"
        f"test_csv_path: {test_path}\n"
        "仅使用当前任务列出的源 CSV，不复用历史特征产物。"
    )
    return {
        "schema_version": "1.0",
        "plan_type": "fixed_biomarker_discovery_v1",
        "selection_status": "pre_registered",
        "steps": [
            {
                "agent": "FEATURE_EXTRACTION_AGENT",
                "task_description": "提取训练集与测试集患者级 3D-MN/16 亚区软骨厚度特征",
                "input_override": extraction_input,
            },
            {
                "agent": "BIOMARKER_DISCOVERY_AGENT",
                "task_description": "仅在训练集发现候选，并在冻结测试集进行预设确认",
                "input_override": (
                    "执行 KL 0-4 横断面 MRI 候选生物标志物发现；"
                    "候选筛选不得使用测试集。"
                ),
            },
            {
                "agent": "BIOMARKER_VERIFIER_AGENT",
                "task_description": "独立审计候选冻结、数据隔离、多重检验与产物哈希",
                "input_override": "独立审计本次生物标志物发现运行，不修改候选集合。",
            },
            {
                "agent": "BIOMARKER_REPORTER_AGENT",
                "task_description": "基于独立审计通过的事实生成针对本次实验的自然语言解释",
                "input_override": "解释本次实验的实际数字、候选效应和研究边界，不使用固定话术。",
            },
        ],
        "final_message": (
            "生物标志物发现流程已结束；请以独立审计结果和本次运行报告为准。"
        ),
        "source_query": str(current_input or ""),
    }


def build_checkpoint_config(
    *, user_id: Any, session_id: Any, run_id: str
) -> dict[str, dict[str, str]]:
    """Use a session thread plus a per-request namespace."""
    session_scope = str(session_id or f"anonymous-{run_id}")
    user_scope = str(user_id if user_id is not None else "anonymous")
    return {
        "configurable": {
            "thread_id": f"user:{user_scope}:session:{session_scope}",
            "checkpoint_ns": f"run:{run_id}",
        }
    }


def normalized_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(path)))


def infer_feature_type_from_path(path: str) -> str | None:
    return infer_feature_type(path)


def feature_type_for_tool(tool_name: str, arguments: Mapping[str, Any] | None = None) -> str | None:
    """Return the actual feature family a tool invocation would produce."""
    arguments = arguments or {}
    if tool_name == "augment_online_extract_features":
        return infer_feature_type(str(arguments.get("feature_type", "")))
    if tool_name in {
        "validate_cartilage_segmentation",
        "analyze_cartilage_thickness",
        "extract_cartilage_thickness_features",
        "generate_cartilage_3d_mesh",
    }:
        return "CartilageThickness"
    return infer_feature_type(tool_name)


def validate_classification_artifacts(
    *,
    train_path: str,
    test_path: str,
    task_spec: Mapping[str, Any],
    current_artifacts: list[Mapping[str, Any]],
    run_id: str | None,
) -> None:
    """Reject cross-run or feature-type-mismatched classification inputs."""
    requested_type = task_spec.get("feature_type")
    requires_fresh = bool(
        task_spec.get("requires_feature_extraction")
        and not task_spec.get("allow_history_reuse")
    )
    artifacts_by_path = {
        normalized_path(str(item.get("path"))): item
        for item in current_artifacts
        if item.get("path")
    }

    for role, path in (("train", train_path), ("test", test_path)):
        if not str(path or "").strip():
            raise ValueError(f"Missing {role} feature artifact for the current task")
        artifact = artifacts_by_path.get(normalized_path(path))
        if requires_fresh and artifact is None:
            raise ValueError(
                f"Cross-task artifact reuse blocked: {role} input was not produced by the current run"
            )
        if artifact is not None:
            if run_id and artifact.get("run_id") != run_id:
                raise ValueError(f"Cross-run {role} artifact is not allowed")
            artifact_type = artifact.get("feature_type")
            if requested_type and artifact_type != requested_type:
                raise ValueError(
                    f"Feature type mismatch: requested {requested_type}, got {artifact_type}"
                )
            artifact_role = artifact.get("dataset_role")
            if artifact_role and artifact_role != role:
                raise ValueError(
                    f"Dataset role mismatch: expected {role}, got {artifact_role}"
                )
        elif requested_type:
            inferred = infer_feature_type_from_path(path)
            if inferred and inferred != requested_type:
                raise ValueError(
                    f"Feature type mismatch: requested {requested_type}, path appears to be {inferred}"
                )
