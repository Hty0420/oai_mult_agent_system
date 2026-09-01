"""Capability-driven candidate generation and history-aware KL pipeline ranking."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_CAPABILITY_REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "pipeline_capabilities.v1.json"
)


def load_capability_registry(
    path: str | Path = DEFAULT_CAPABILITY_REGISTRY,
) -> dict[str, Any]:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_capability_registry(registry)
    return registry


def validate_capability_registry(registry: Mapping[str, Any]) -> None:
    required_root = {
        "schema_version", "registry_id", "version", "status",
        "objective_variant", "selection_policy", "pipelines",
    }
    missing = required_root.difference(registry)
    if missing:
        raise ValueError(f"capability registry missing fields: {sorted(missing)}")
    if registry.get("schema_version") != "1.0":
        raise ValueError("unsupported capability registry schema_version")
    search_policy = registry.get("search_policy", {})
    if search_policy:
        cold_budget = int(search_policy.get("cold_start_max_candidates") or 0)
        experienced_budget = int(
            search_policy.get("experienced_max_candidates") or 0
        )
        if cold_budget < 1 or experienced_budget < 1:
            raise ValueError("pipeline search budgets must be positive integers")
        weights = search_policy.get("validation_metric_weights", {})
        if not weights or abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
            raise ValueError("pipeline validation metric weights must sum to 1")
    pipelines = registry.get("pipelines")
    if not isinstance(pipelines, list) or not pipelines:
        raise ValueError("capability registry pipelines must be a non-empty list")
    seen: set[str] = set()
    required_pipeline = {
        "candidate_id", "feature_type", "classification_method", "model_recipe",
        "recipe_config",
        "required_columns", "agents", "tools", "registry_prior", "cost_rank",
    }
    for candidate in pipelines:
        if not isinstance(candidate, Mapping):
            raise ValueError("pipeline candidate must be an object")
        missing_pipeline = required_pipeline.difference(candidate)
        if missing_pipeline:
            raise ValueError(
                f"pipeline candidate missing fields: {sorted(missing_pipeline)}"
            )
        candidate_id = str(candidate["candidate_id"])
        if not candidate_id or candidate_id in seen:
            raise ValueError(f"duplicate or empty candidate_id: {candidate_id!r}")
        seen.add(candidate_id)
        prior = float(candidate["registry_prior"])
        if not 0.0 <= prior <= 1.0:
            raise ValueError(f"invalid registry_prior for {candidate_id}")
        recipe = str(candidate.get("model_recipe") or "")
        expected_recipe = {
            "LASSO": "L1_LOGISTIC",
            "Filter": "ANOVA_TOPK_L2",
        }.get(candidate.get("classification_method"))
        if expected_recipe is None or recipe != expected_recipe:
            raise ValueError(
                f"candidate {candidate_id} has an incompatible model_recipe"
            )
        if not isinstance(candidate.get("recipe_config"), Mapping):
            raise ValueError(f"candidate {candidate_id} recipe_config must be an object")


def load_history_priors(
    *, task_fingerprint: str | None = None
) -> dict[str, Any]:
    """Load reliability from comparable tasks only when a fingerprint exists."""
    try:
        from db_utils.analysis_queries import (
            matched_task_policy_scorecard,
            matched_pipeline_trial_scorecard,
            task_policy_scorecard,
            tool_error_rates,
        )
        from db_utils.db_utils import DatabaseManager
    except ImportError as exc:
        return {
            "status": "unavailable",
            "reason": type(exc).__name__,
            "methods": {},
            "pipeline_trials": {},
            "tools": {},
        }
    db = None
    try:
        db = DatabaseManager()
        score_rows = (
            matched_task_policy_scorecard(db, task_fingerprint)
            if task_fingerprint
            else task_policy_scorecard(db)
        )
        tool_rows = tool_error_rates(db)
        trial_rows = (
            matched_pipeline_trial_scorecard(db, task_fingerprint)
            if task_fingerprint
            else []
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": type(exc).__name__,
            "methods": {},
            "pipeline_trials": {},
            "tools": {},
        }
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    method_accumulator: dict[str, dict[str, float]] = {}
    for row in score_rows:
        method = str(row.get("method_name") or "")
        runs = int(row.get("evaluated_runs") or 0)
        if not method or runs <= 0:
            continue
        bucket = method_accumulator.setdefault(
            method, {"runs": 0.0, "success": 0.0, "false_completion": 0.0}
        )
        bucket["runs"] += runs
        bucket["success"] += float(row.get("task_success_rate") or 0.0) * runs
        bucket["false_completion"] += float(
            row.get("false_completion_rate") or 0.0
        ) * runs

    methods: dict[str, dict[str, float]] = {}
    for method, bucket in method_accumulator.items():
        runs = bucket["runs"]
        methods[method] = {
            "evaluated_runs": int(runs),
            "task_success_rate": bucket["success"] / runs,
            "false_completion_rate": bucket["false_completion"] / runs,
        }

    tool_accumulator: dict[str, dict[str, float]] = {}
    for row in tool_rows:
        tool = str(row.get("tool_name") or "")
        count = int(row.get("cnt") or 0)
        if not tool or count <= 0:
            continue
        bucket = tool_accumulator.setdefault(tool, {"calls": 0.0, "failures": 0.0})
        bucket["calls"] += count
        if str(row.get("status") or "").lower() != "success":
            bucket["failures"] += count
    tools = {
        tool: {
            "calls": int(bucket["calls"]),
            "failure_rate": bucket["failures"] / bucket["calls"],
        }
        for tool, bucket in tool_accumulator.items()
        if bucket["calls"] > 0
    }
    pipeline_trials = {
        str(row.get("candidate_id")): {
            "evaluated_runs": int(row.get("evaluated_runs") or 0),
            "mean_validation_score": float(
                row.get("mean_validation_score") or 0.0
            ),
            "mean_auc": float(row.get("mean_auc") or 0.0),
            "mean_auc_ci_lower": float(row.get("mean_auc_ci_lower") or 0.0),
            "mean_selection_stability": float(
                row.get("mean_selection_stability") or 0.0
            ),
            "mean_duration_ms": float(row.get("mean_duration_ms") or 0.0),
            "model_recipe": row.get("model_recipe"),
        }
        for row in trial_rows
        if row.get("candidate_id") and int(row.get("evaluated_runs") or 0) > 0
    }
    return {
        "status": "available",
        "task_fingerprint": task_fingerprint,
        "comparison_scope": "matched_task" if task_fingerprint else "global_legacy",
        "methods": methods,
        "pipeline_trials": pipeline_trials,
        "tools": tools,
    }


def _required_columns_missing(
    candidate: Mapping[str, Any], dataset_profile: Mapping[str, Any]
) -> list[str]:
    required = set(str(column) for column in candidate.get("required_columns", []))
    missing: set[str] = set()
    source_files = dataset_profile.get("source_files", {})
    for role in ("train", "test"):
        profile = source_files.get(role)
        if not isinstance(profile, Mapping):
            missing.add(f"{role}:source_csv")
            continue
        columns = set(str(column) for column in profile.get("columns", []))
        missing.update(f"{role}:{column}" for column in required.difference(columns))
    return sorted(missing)


def generate_pipeline_candidates(
    *,
    goal_spec: Mapping[str, Any],
    dataset_profile: Mapping[str, Any],
    registry: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    registry = registry or load_capability_registry()
    constraints = goal_spec.get("constraints", {})
    constrained_feature = constraints.get("feature_type")
    constrained_method = constraints.get("classification_method")
    objective = goal_spec.get("objective_variant")
    candidates: list[dict[str, Any]] = []

    for raw in registry.get("pipelines", []):
        candidate = deepcopy(dict(raw))
        rejection_reasons: list[str] = []
        if objective != registry.get("objective_variant"):
            rejection_reasons.append("unsupported_objective_variant")
        if constrained_feature and candidate.get("feature_type") != constrained_feature:
            rejection_reasons.append("feature_constraint_mismatch")
        if constrained_method and candidate.get("classification_method") != constrained_method:
            rejection_reasons.append("classifier_constraint_mismatch")
        rejection_reasons.extend(_required_columns_missing(candidate, dataset_profile))
        if dataset_profile.get("status") != "ready":
            rejection_reasons.append("dataset_profile_not_ready")
        candidate.update(
            {
                "registry_version": registry.get("version"),
                "eligible": not rejection_reasons,
                "rejection_reasons": list(dict.fromkeys(rejection_reasons)),
            }
        )
        candidates.append(candidate)
    return candidates


def rank_pipeline_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    history: Mapping[str, Any] | None = None,
    selection_policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    history = history or {"status": "unavailable", "methods": {}, "tools": {}}
    selection_policy = selection_policy or {}
    history_weight = float(selection_policy.get("history_weight", 0.45))
    prior_weight = float(selection_policy.get("registry_prior_weight", 0.45))
    cost_weight = float(selection_policy.get("cost_weight", 0.10))
    methods = history.get("methods", {})
    pipeline_trials = history.get("pipeline_trials", {})
    tools = history.get("tools", {})
    ranked: list[dict[str, Any]] = []

    for raw in candidates:
        candidate = deepcopy(dict(raw))
        method_key = (
            f"{candidate.get('feature_type')}:{candidate.get('classification_method')}"
        )
        method_history = methods.get(method_key, {})
        task_runs = int(method_history.get("evaluated_runs") or 0)
        trial_history = pipeline_trials.get(
            str(candidate.get("candidate_id") or ""), {}
        )
        trial_runs = int(trial_history.get("evaluated_runs") or 0)
        runs = max(task_runs, trial_runs)
        observed_success = float(method_history.get("task_success_rate") or 0.0)
        observed_false_completion = float(
            method_history.get("false_completion_rate") or 0.0
        )
        # Beta(1, 1) smoothing prevents one historical run dominating selection.
        reliability = ((observed_success * task_runs) + 1.0) / (task_runs + 2.0)
        reliability *= 1.0 - min(1.0, observed_false_completion)
        tool_failure_rates = [
            float(tools.get(tool, {}).get("failure_rate") or 0.0)
            for tool in candidate.get("tools", [])
            if tool in tools
        ]
        tool_reliability = (
            1.0 - (sum(tool_failure_rates) / len(tool_failure_rates))
            if tool_failure_rates else 0.5
        )
        trial_reliability = 0.5
        if trial_runs:
            smoothed_validation = (
                float(trial_history.get("mean_validation_score") or 0.0)
                * trial_runs
                + 1.0
            ) / (trial_runs + 2.0)
            trial_reliability = (
                smoothed_validation
                + float(trial_history.get("mean_auc_ci_lower") or 0.0)
                + float(
                    trial_history.get("mean_selection_stability") or 0.0
                )
            ) / 3.0
        history_reliability = (
            reliability + tool_reliability + trial_reliability
        ) / 3.0
        cost_rank = max(1.0, float(candidate.get("cost_rank") or 1.0))
        normalized_cost = min(1.0, (cost_rank - 1.0) / 4.0)
        score = (
            prior_weight * float(candidate.get("registry_prior") or 0.0)
            + history_weight * history_reliability
            - cost_weight * normalized_cost
        )
        if not candidate.get("eligible"):
            score = -1.0
        candidate.update(
            {
                "score": round(score, 6),
                "score_components": {
                    "registry_prior": float(candidate.get("registry_prior") or 0.0),
                    "history_reliability": round(history_reliability, 6),
                    "history_runs": runs,
                    "task_history_runs": task_runs,
                    "paired_trial_runs": trial_runs,
                    "paired_trial_reliability": round(
                        trial_reliability, 6
                    ),
                    "normalized_cost": round(normalized_cost, 6),
                    "history_status": history.get("status", "unavailable"),
                },
            }
        )
        ranked.append(candidate)
    ranked.sort(
        key=lambda item: (
            bool(item.get("eligible")),
            float(item.get("score") or -1.0),
            str(item.get("candidate_id") or ""),
        ),
        reverse=True,
    )
    for index, candidate in enumerate(ranked, 1):
        candidate["rank"] = index if candidate.get("eligible") else None
    return ranked


def select_pipeline_candidate(
    candidates: Iterable[Mapping[str, Any]],
    *,
    llm_candidate_id: str | None = None,
    llm_tie_tolerance: float = 0.02,
) -> dict[str, Any] | None:
    eligible = [dict(item) for item in candidates if item.get("eligible")]
    if not eligible:
        return None
    eligible.sort(key=lambda item: float(item.get("score") or -1.0), reverse=True)
    best = eligible[0]
    if llm_candidate_id:
        proposed = next(
            (
                item for item in eligible
                if str(item.get("candidate_id")) == str(llm_candidate_id)
            ),
            None,
        )
        if proposed is not None and (
            float(best.get("score") or 0.0) - float(proposed.get("score") or 0.0)
            <= llm_tie_tolerance
        ):
            best = proposed
    return deepcopy(best)


def build_trial_portfolio(
    candidates: Iterable[Mapping[str, Any]],
    *,
    search_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose a small, diverse trial set from eligible ranked candidates.

    Cold-start search covers both feature-family and classifier alternatives.
    Once the incumbent has enough matched task history, the budget contracts to
    the incumbent plus one challenger.
    """
    policy = dict(search_policy or {})
    eligible = [deepcopy(dict(item)) for item in candidates if item.get("eligible")]
    eligible.sort(
        key=lambda item: (
            int(item.get("rank") or 10**9),
            -float(item.get("score") or -1.0),
            str(item.get("candidate_id") or ""),
        )
    )
    if not eligible:
        return {
            "schema_version": "1.0",
            "status": "blocked",
            "mode": policy.get("mode", "balanced"),
            "budget": 0,
            "candidate_ids": [],
            "feature_types": [],
            "reason": "no_eligible_candidates",
        }

    incumbent_runs = int(
        eligible[0].get("score_components", {}).get("history_runs") or 0
    )
    threshold = int(policy.get("min_history_runs_for_exploitation", 5))
    cold_start = incumbent_runs < threshold
    requested_budget = int(
        policy.get(
            "cold_start_max_candidates" if cold_start else "experienced_max_candidates",
            3 if cold_start else 2,
        )
    )
    budget = min(max(1, requested_budget), len(eligible))
    selected = [eligible[0]]

    def add_first(predicate) -> None:
        if len(selected) >= budget:
            return
        selected_ids = {str(item.get("candidate_id")) for item in selected}
        for item in eligible:
            if str(item.get("candidate_id")) in selected_ids:
                continue
            if predicate(item):
                selected.append(item)
                return

    incumbent_feature = eligible[0].get("feature_type")
    incumbent_method = eligible[0].get("classification_method")
    add_first(lambda item: item.get("feature_type") != incumbent_feature)
    add_first(lambda item: item.get("classification_method") != incumbent_method)
    while len(selected) < budget:
        before = len(selected)
        add_first(lambda item: True)
        if len(selected) == before:
            break

    return {
        "schema_version": "1.0",
        "status": "ready",
        "mode": policy.get("mode", "balanced"),
        "budget": len(selected),
        "candidate_ids": [item.get("candidate_id") for item in selected],
        "feature_types": list(
            dict.fromkeys(str(item.get("feature_type")) for item in selected)
        ),
        "cold_start": cold_start,
        "incumbent_history_runs": incumbent_runs,
        "reason": (
            "cold_start_diverse_training_validation"
            if cold_start
            else "incumbent_plus_challenger_training_validation"
        ),
    }


def attach_trial_results(
    candidates: Iterable[Mapping[str, Any]],
    trial_results: Iterable[Mapping[str, Any]],
    *,
    search_policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Attach matched training-only validation evidence to candidate records."""
    policy = dict(search_policy or {})
    weights = policy.get(
        "validation_metric_weights", {"AUC": 0.7, "PR_AUC": 0.2, "F1": 0.1}
    )
    results_by_id = {
        str(item.get("candidate_id")): dict(item) for item in trial_results
    }
    updated: list[dict[str, Any]] = []
    for raw in candidates:
        candidate = deepcopy(dict(raw))
        result = results_by_id.get(str(candidate.get("candidate_id")))
        if result is None:
            candidate.update({"trial_status": "not_run", "validation_score": None})
            updated.append(candidate)
            continue
        status = str(result.get("status") or "failed").lower()
        metrics = dict(result.get("metrics") or {})
        intervals = dict(result.get("metric_intervals") or {})
        stability = dict(result.get("selection_stability") or {})
        scope = result.get("validation_scope")
        test_used = bool(result.get("test_data_used"))
        expected_recipe = candidate.get("model_recipe")
        observed_recipe = result.get("model_recipe")
        recipe_matches = (
            not expected_recipe
            or (
                bool(observed_recipe)
                and str(expected_recipe) == str(observed_recipe)
            )
        )
        valid_scope = scope == policy.get(
            "validation_scope", "training_only_nested_grouped_cv"
        )
        try:
            metric_values = {name: float(metrics[name]) for name in weights}
            if not all(
                math.isfinite(value) and 0.0 <= value <= 1.0
                for value in metric_values.values()
            ):
                raise ValueError("validation metrics must be finite values in [0, 1]")
            validation_score = sum(
                float(weights[name]) * metric_values[name] for name in weights
            )
        except (KeyError, TypeError, ValueError):
            validation_score = None
        if status != "success" or not valid_scope or test_used or not recipe_matches:
            validation_score = None
        auc_lower_bound = None
        if validation_score is not None:
            try:
                auc_lower_bound = float(intervals.get("AUC", {}).get("lower"))
                if not math.isfinite(auc_lower_bound) or not 0.0 <= auc_lower_bound <= 1.0:
                    raise ValueError("invalid AUC interval lower bound")
            except (AttributeError, TypeError, ValueError):
                auc_lower_bound = float(metric_values.get("AUC", 0.0))
        selection_stability = None
        if validation_score is not None:
            try:
                selection_stability = float(
                    stability.get("mean_pairwise_jaccard")
                )
                if (
                    not math.isfinite(selection_stability)
                    or not 0.0 <= selection_stability <= 1.0
                ):
                    raise ValueError("invalid selection stability")
            except (TypeError, ValueError):
                selection_stability = 0.0
        candidate.update(
            {
                "trial_status": "success" if validation_score is not None else "failed",
                "validation_scope": scope,
                "test_data_used_for_selection": test_used,
                "validation_metrics": metrics if validation_score is not None else {},
                "validation_metric_intervals": (
                    intervals if validation_score is not None else {}
                ),
                "validation_auc_lower_bound": auc_lower_bound,
                "validation_selection_stability": selection_stability,
                "validation_recommended_params": (
                    dict(result.get("recommended_params") or {})
                    if validation_score is not None
                    else {}
                ),
                "validation_dataset_hash": result.get("train_sha256"),
                "validation_evaluator_version": result.get("evaluator_version"),
                "validation_patient_count": result.get("patient_count"),
                "validation_score": (
                    round(float(validation_score), 6)
                    if validation_score is not None
                    else None
                ),
                "trial_error": (
                    "model_recipe_mismatch"
                    if not recipe_matches
                    else result.get("error")
                ),
            }
        )
        updated.append(candidate)
    return updated


def select_pipeline_from_trials(
    candidates: Iterable[Mapping[str, Any]],
    *,
    search_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Select by matched training CV; use the prior rank only for near ties."""
    policy = dict(search_policy or {})
    successful = [
        deepcopy(dict(item))
        for item in candidates
        if item.get("eligible")
        and item.get("trial_status") == "success"
        and item.get("validation_score") is not None
        and not item.get("test_data_used_for_selection")
    ]
    minimum = int(policy.get("min_successful_trials", 1))
    if len(successful) < minimum:
        return None
    successful.sort(
        key=lambda item: (
            float(item.get("validation_score") or -1.0),
            float(item.get("score") or -1.0),
        ),
        reverse=True,
    )
    best_score = float(successful[0]["validation_score"])
    tolerance = float(policy.get("validation_tie_tolerance", 0.01))
    near_ties = [
        item
        for item in successful
        if best_score - float(item.get("validation_score") or -1.0) <= tolerance
    ]
    near_ties.sort(
        key=lambda item: (
            float(item.get("validation_auc_lower_bound") or -1.0),
            float(item.get("validation_selection_stability") or -1.0),
            float(item.get("score") or -1.0),
            -int(item.get("rank") or 10**9),
        ),
        reverse=True,
    )
    return near_ties[0]


def materialize_pipeline_search_plan(
    *,
    trial_portfolio: Mapping[str, Any],
    dataset_profile: Mapping[str, Any],
    user_query: str,
) -> dict[str, Any]:
    """Create a two-stage plan: training-only search, then one final test run."""
    if trial_portfolio.get("status") != "ready":
        return {
            "steps": [],
            "selection_status": "blocked",
            "selection_reason": trial_portfolio.get("reason"),
            "trial_portfolio": dict(trial_portfolio),
        }
    source_files = dataset_profile.get("source_files", {})
    train_path = str(source_files.get("train", {}).get("path") or "")
    test_path = str(source_files.get("test", {}).get("path") or "")
    candidate_ids = ", ".join(
        str(item) for item in trial_portfolio.get("candidate_ids", [])
    )
    return {
        "steps": [
            {
                "agent": "PIPELINE_OPTIMIZATION_AGENT",
                "task_description": "在训练集内部试跑候选方案并选择当前数据上的优先方案",
                "input_override": (
                    f"当前目标：{user_query}\n"
                    f"候选方案：{candidate_ids}\n"
                    "候选排序只能使用训练集患者级嵌套交叉验证，禁止使用测试集标签或测试指标。"
                    f"\ntrain_csv_path: {train_path}\ntest_csv_path: {test_path}"
                ),
            },
            {
                "agent": "CLASSIFICATION_AGENT",
                "task_description": "使用训练集试跑选出的方案执行一次最终分类并读取报告",
                "input_override": "使用上一步确定的唯一方案及其当前运行特征工件执行最终分类；不得比较测试集结果后改选方案。",
            },
        ],
        "selected_candidate_id": None,
        "selection_status": "trial_pending",
        "selection_reason": "training_only_candidate_search",
        "trial_portfolio": dict(trial_portfolio),
    }


def materialize_pipeline_plan(
    *,
    selected: Mapping[str, Any] | None,
    dataset_profile: Mapping[str, Any],
    user_query: str,
    llm_candidate_id: str | None = None,
) -> dict[str, Any]:
    if selected is None:
        return {
            "steps": [],
            "selected_candidate_id": None,
            "llm_selected_candidate_id": llm_candidate_id,
            "selection_status": "blocked",
            "selection_reason": "没有满足当前数据前置条件的候选流水线。",
        }
    source_files = dataset_profile.get("source_files", {})
    train_path = str(source_files.get("train", {}).get("path") or "")
    test_path = str(source_files.get("test", {}).get("path") or "")
    feature_type = str(selected.get("feature_type"))
    method = str(selected.get("classification_method"))
    return {
        "steps": [
            {
                "agent": "FEATURE_EXTRACTION_AGENT",
                "task_description": f"为当前 KL 分级目标提取 {feature_type} 训练集和测试集特征",
                "input_override": (
                    f"当前目标：{user_query}\n"
                    f"仅为本次任务提取 {feature_type} 特征。"
                    f"\ntrain_csv_path: {train_path}"
                    f"\ntest_csv_path: {test_path}"
                ),
            },
            {
                "agent": "CLASSIFICATION_AGENT",
                "task_description": f"使用当前运行的 {feature_type} 特征执行 {method} 分类并读取报告",
                "input_override": (
                    f"仅使用当前运行上一步生成的 {feature_type} 训练集和测试集特征，"
                    f"执行 {method} 分类，并读取本次分类产生的报告。"
                ),
            },
        ],
        "selected_candidate_id": selected.get("candidate_id"),
        "llm_selected_candidate_id": llm_candidate_id,
        "selection_status": "selected",
        "selection_reason": {
            "score": selected.get("score"),
            "rank": selected.get("rank"),
            "score_components": selected.get("score_components", {}),
        },
    }
