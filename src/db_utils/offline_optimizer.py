"""Read-only daily evidence aggregation and draft policy generation.

The optimizer never activates a policy.  It produces a privacy-minimizing
preview and, only when explicitly requested by the operator, registers the
exact preview as a ``draft`` policy for later replay and shadow validation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import uuid

from .trajectory import (
    TRAJECTORY_DATA_PARTITIONS,
    build_artifact_record,
    build_policy_record,
    build_trajectory_event,
    current_agent_version,
    personal_workspace_id,
    sanitize_for_audit,
    stable_json_hash,
)
from .self_optimization_control import (
    assess_policy_transition,
    constitution_reference,
    load_self_optimization_constitution,
    validate_candidate_policy_record,
    validate_self_optimization_constitution,
)


OFFLINE_OPTIMIZER_VERSION = "1.2.0"
DEFAULT_MIN_MATCHED_RUNS = 20


TASK_RUN_EVIDENCE_SQL = """
WITH task_metrics AS (
    SELECT
        trace_id,
        MAX(CASE WHEN metric_name = 'task_success' THEN metric_value END)
            AS task_success,
        MAX(CASE WHEN metric_name = 'false_completion' THEN metric_value END)
            AS false_completion,
        MAX(CASE WHEN metric_name = 'cross_run_artifact_violation'
                 THEN metric_value END) AS cross_run_violation,
        MAX(CASE WHEN metric_name = 'required_tool_coverage'
                 THEN metric_value END) AS tool_coverage
    FROM evaluation_results
    WHERE evaluator_name = 'task-contract-evaluator'
      AND record_type = 'metric'
    GROUP BY trace_id
), gate_failures AS (
    SELECT
        trace_id,
        SUM(gate_status = 'fail') AS failed_gate_count
    FROM evaluation_results
    WHERE evaluator_name = 'task-contract-evaluator'
      AND record_type = 'gate'
    GROUP BY trace_id
), scientific_quality AS (
    SELECT
        trace_id,
        COUNT(*) AS scientific_gate_count,
        SUM(gate_status = 'fail') AS scientific_failed_gate_count,
        SUM(gate_status = 'unknown') AS scientific_unknown_gate_count
    FROM evaluation_results
    WHERE evaluator_name = 'classification_manifest'
      AND record_type = 'gate'
      AND metric_name IN (
        'manifest_schema_complete',
        'metric_domain_valid',
        'confusion_metric_consistency',
        'patient_count_consistency',
        'artifact_files_complete',
        'input_hashes_verified',
        'train_test_inputs_distinct',
        'environment_provenance_complete'
      )
    GROUP BY trace_id
), latency AS (
    SELECT trace_id, SUM(total_latency_ms) AS total_latency_ms
    FROM agent_execution_log
    GROUP BY trace_id
)
SELECT
    t.trace_id,
    t.task_fingerprint,
    t.policy_id,
    COALESCE(e.method_name, 'unknown') AS method_name,
    m.task_success,
    m.false_completion,
    m.cross_run_violation,
    m.tool_coverage,
    COALESCE(g.failed_gate_count, 0) AS failed_gate_count,
    COALESCE(s.scientific_gate_count, 0) AS scientific_gate_count,
    COALESCE(s.scientific_failed_gate_count, 0)
        AS scientific_failed_gate_count,
    COALESCE(s.scientific_unknown_gate_count, 0)
        AS scientific_unknown_gate_count,
    l.total_latency_ms
FROM tasks t
JOIN task_metrics m ON m.trace_id = t.trace_id
LEFT JOIN evaluation_results e
  ON e.trace_id = t.trace_id
 AND e.evaluator_name = 'task-contract-evaluator'
 AND e.record_type = 'metric'
 AND e.metric_name = 'task_success'
LEFT JOIN gate_failures g ON g.trace_id = t.trace_id
LEFT JOIN scientific_quality s ON s.trace_id = t.trace_id
LEFT JOIN latency l ON l.trace_id = t.trace_id
WHERE t.task_fingerprint IS NOT NULL
  AND __SCOPE__
"""


FEEDBACK_COVERAGE_SQL = """
SELECT
    COUNT(DISTINCT t.trace_id) AS task_count,
    COUNT(DISTINCT tf.trace_id) AS task_feedback_count,
    COUNT(DISTINCT rf.trace_id) AS routing_feedback_count,
    COUNT(DISTINCT CASE WHEN tf.feedback_source = 'human' THEN tf.trace_id END)
        AS human_feedback_count
FROM tasks t
LEFT JOIN task_feedback tf ON tf.trace_id = t.trace_id
LEFT JOIN routing_feedback rf ON rf.trace_id = t.trace_id
WHERE __SCOPE__
"""


TOOL_EVIDENCE_SQL = """
SELECT
    tool_name,
    COUNT(*) AS calls,
    SUM(status <> 'success') AS failures,
    AVG(execution_time_ms) AS avg_latency_ms,
    MAX(execution_time_ms) AS max_latency_ms
FROM tool_calls_log tc
JOIN (
    SELECT DISTINCT trace_id
    FROM tasks t
    WHERE __SCOPE__
) scoped ON scoped.trace_id = tc.trace_id
WHERE tc.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY tool_name
ORDER BY calls DESC, tool_name
"""


PIPELINE_TRIAL_EVIDENCE_SQL = """
SELECT
    pt.trace_id,
    pt.task_fingerprint,
    pt.dataset_hash,
    pt.candidate_id,
    pt.feature_type,
    pt.classification_method,
    pt.model_recipe,
    pt.validation_score,
    pt.auc,
    pt.pr_auc,
    pt.f1,
    pt.auc_ci_lower,
    pt.selection_stability,
    pt.duration_ms
FROM pipeline_trials pt
JOIN tasks t ON t.trace_id = pt.trace_id
WHERE pt.status = 'success'
  AND pt.test_data_used = 0
  AND pt.validation_scope = 'training_only_nested_grouped_cv'
  AND __SCOPE__
ORDER BY pt.task_fingerprint, pt.trace_id, pt.candidate_id
"""


def _query(
    db: Any, sql: str, params: Sequence[Any] = ()
) -> list[dict[str, Any]]:
    with db.connection.cursor() as cursor:
        cursor.execute(sql, tuple(params))
        return list(cursor.fetchall())


def _scoped_query(
    sql: str,
    *,
    user_id: int,
    workspace_id: str,
    agent_version: str,
    data_partitions: Sequence[str],
) -> tuple[str, tuple[Any, ...]]:
    partitions = tuple(dict.fromkeys(str(item) for item in data_partitions))
    if not partitions or any(item not in TRAJECTORY_DATA_PARTITIONS for item in partitions):
        raise ValueError("data_partitions must contain only train/validation/test")
    placeholders = ", ".join(["%s"] * len(partitions))
    scope = (
        f"t.user_id = %s AND t.workspace_id = %s AND t.agent_version = %s "
        f"AND t.data_partition IN ({placeholders})"
    )
    params = (user_id, workspace_id, agent_version, *partitions)
    return sql.replace("__SCOPE__", scope), params


def collect_evidence_snapshot(
    db: Any,
    *,
    user_id: int,
    workspace_id: str | None = None,
    agent_version: str | None = None,
    data_partitions: Sequence[str] = ("train",),
) -> dict[str, Any]:
    """Collect isolated learning evidence for one logical tenant and version."""
    workspace_id = workspace_id or personal_workspace_id(user_id)
    agent_version = agent_version or current_agent_version()
    scope = {
        "user_id": user_id,
        "workspace_id": workspace_id,
        "agent_version": agent_version,
        "data_partitions": list(dict.fromkeys(data_partitions)),
    }
    task_sql, task_params = _scoped_query(
        TASK_RUN_EVIDENCE_SQL, **scope
    )
    feedback_sql, feedback_params = _scoped_query(
        FEEDBACK_COVERAGE_SQL, **scope
    )
    tool_sql, tool_params = _scoped_query(TOOL_EVIDENCE_SQL, **scope)
    trial_sql, trial_params = _scoped_query(
        PIPELINE_TRIAL_EVIDENCE_SQL, **scope
    )
    task_runs = _query(db, task_sql, task_params)
    feedback_rows = _query(db, feedback_sql, feedback_params)
    tool_rows = _query(db, tool_sql, tool_params)
    pipeline_trials = _query(db, trial_sql, trial_params)
    feedback = feedback_rows[0] if feedback_rows else {}
    return sanitize_for_audit(
        {
            "schema_version": "1.1",
            "optimizer_version": OFFLINE_OPTIMIZER_VERSION,
            "scope": scope,
            "task_runs": task_runs,
            "pipeline_trials": pipeline_trials,
            "feedback_coverage": feedback,
            "tool_evidence": tool_rows,
        }
    )


def _aggregate_methods(
    task_runs: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in task_runs:
        fingerprint = str(row.get("task_fingerprint") or "")
        method = str(row.get("method_name") or "unknown")
        if fingerprint:
            buckets[(fingerprint, method)].append(row)

    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (fingerprint, method), rows in buckets.items():
        count = len(rows)
        mean = lambda name: sum(float(row.get(name) or 0.0) for row in rows) / count
        contexts[fingerprint].append(
            {
                "method_name": method,
                "matched_runs": count,
                "task_success_rate": mean("task_success"),
                "false_completion_rate": mean("false_completion"),
                "cross_run_violation_rate": mean("cross_run_violation"),
                "mean_tool_coverage": mean("tool_coverage"),
                "failed_gate_runs": sum(
                    int(row.get("failed_gate_count") or 0) > 0 for row in rows
                ),
                "scientific_verified_runs": sum(
                    int(row.get("scientific_gate_count") or 0) == 8
                    and int(row.get("scientific_failed_gate_count") or 0) == 0
                    and int(row.get("scientific_unknown_gate_count") or 0) == 0
                    for row in rows
                ),
                "scientific_incomplete_runs": sum(
                    not (
                        int(row.get("scientific_gate_count") or 0) == 8
                        and int(row.get("scientific_failed_gate_count") or 0) == 0
                        and int(row.get("scientific_unknown_gate_count") or 0) == 0
                    )
                    for row in rows
                ),
                "mean_latency_ms": mean("total_latency_ms"),
            }
        )
    return dict(contexts)


def _paired_trial_recommendations(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize within-context candidate comparisons without promotion."""
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        fingerprint = str(row.get("task_fingerprint") or "")
        candidate_id = str(row.get("candidate_id") or "")
        if fingerprint and candidate_id:
            buckets[(fingerprint, candidate_id)].append(row)
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (fingerprint, candidate_id), trials in buckets.items():
        count = len(trials)
        mean = lambda name: sum(
            float(item.get(name) or 0.0) for item in trials
        ) / count
        contexts[fingerprint].append(
            {
                "candidate_id": candidate_id,
                "model_recipe": trials[0].get("model_recipe"),
                "paired_trials": count,
                "mean_validation_score": mean("validation_score"),
                "mean_auc_ci_lower": mean("auc_ci_lower"),
                "mean_selection_stability": mean("selection_stability"),
                "mean_duration_ms": mean("duration_ms"),
            }
        )
    recommendations: dict[str, Any] = {}
    for fingerprint, candidates in contexts.items():
        if len(candidates) < 2:
            continue
        candidates.sort(
            key=lambda item: (
                item["mean_validation_score"],
                item["mean_auc_ci_lower"],
                item["mean_selection_stability"],
                -item["mean_duration_ms"],
            ),
            reverse=True,
        )
        recommendations[fingerprint] = {
            "preferred_candidate_id": candidates[0]["candidate_id"],
            "evidence": candidates,
            "use_mode": "portfolio_prior_only",
            "production_activation_allowed": False,
        }
    return recommendations


def build_draft_policy_preview(
    snapshot: Mapping[str, Any],
    *,
    base_policy_id: str = "baseline-v1",
    min_matched_runs: int = DEFAULT_MIN_MATCHED_RUNS,
    constitution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic draft; never mark it active or promoted."""
    snapshot_scope = dict(snapshot.get("scope", {}))
    scoped_partitions = snapshot_scope.get("data_partitions")
    if list(scoped_partitions or ()) != ["train"]:
        raise ValueError(
            "policy candidates may only be learned from the train trajectory partition"
        )
    active_constitution = validate_self_optimization_constitution(
        constitution or load_self_optimization_constitution()
    )
    constitutional_floor = int(
        active_constitution["candidate_contract"][
            "minimum_matched_runs_per_method"
        ]
    )
    if min_matched_runs < constitutional_floor:
        raise ValueError(
            "min_matched_runs is below the self-optimization constitutional floor"
        )
    contexts = _aggregate_methods(snapshot.get("task_runs", []))
    paired_trial_recommendations = _paired_trial_recommendations(
        snapshot.get("pipeline_trials", [])
    )
    preferences: dict[str, Any] = {}
    insufficient: dict[str, Any] = {}

    for fingerprint, methods in contexts.items():
        eligible = [
            method
            for method in methods
            if method["matched_runs"] >= min_matched_runs
            and method["false_completion_rate"] == 0.0
            and method["cross_run_violation_rate"] == 0.0
            and method["failed_gate_runs"] == 0
            and method["scientific_verified_runs"] == method["matched_runs"]
            and method["scientific_incomplete_runs"] == 0
        ]
        if len(eligible) < 2:
            insufficient[fingerprint] = {
                "reason": "requires_two_safe_comparable_methods",
                "methods": methods,
            }
            continue
        eligible.sort(
            key=lambda item: (
                item["task_success_rate"],
                item["mean_tool_coverage"],
                -item["mean_latency_ms"],
            ),
            reverse=True,
        )
        preferences[fingerprint] = {
            "preferred_method": eligible[0]["method_name"],
            "evidence": eligible,
            "use_mode": "shadow_only",
        }

    feedback = dict(snapshot.get("feedback_coverage", {}))
    task_count = int(feedback.get("task_count") or len(snapshot.get("task_runs", [])))
    human_feedback_count = int(feedback.get("human_feedback_count") or 0)
    human_feedback_rate = human_feedback_count / task_count if task_count else 0.0
    snapshot_hash = stable_json_hash(snapshot)
    config = {
        "optimizer_version": OFFLINE_OPTIMIZER_VERSION,
        "base_policy_id": base_policy_id,
        "evidence_snapshot_sha256": snapshot_hash,
        "evidence_scope": snapshot.get("scope", {}),
        "min_matched_runs": min_matched_runs,
        "contextual_method_preferences": preferences,
        "shadow_eligible": bool(preferences),
        "production_activation_allowed": False,
        "required_next_gate": (
            "paired_shadow_replay_and_human_approval"
            if preferences
            else "collect_more_comparable_tasks_and_human_feedback"
        ),
        "self_optimization_constitution": constitution_reference(
            active_constitution
        ),
    }
    # Candidate identity covers the complete, sanitized configuration rather
    # than only its evidence snapshot. Constitutional or threshold changes
    # therefore create a new immutable policy version instead of overwriting an
    # existing ID.
    policy_id = f"daily-shadow-{stable_json_hash(config)[:16]}"
    policy = build_policy_record(
        policy_id=policy_id,
        policy_type="multi_agent_shadow_candidate",
        version=OFFLINE_OPTIMIZER_VERSION,
        config=config,
        status="draft",
        parent_policy_id=base_policy_id,
        created_by="offline_optimizer",
    )
    policy = validate_candidate_policy_record(policy, active_constitution)
    shadow_readiness = assess_policy_transition(
        candidate=policy,
        current_status="draft",
        target_status="shadow",
        evidence={"candidate_config_valid": True},
        approval=None,
        constitution=active_constitution,
    )
    return sanitize_for_audit(
        {
            "schema_version": "1.1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "optimizer_version": OFFLINE_OPTIMIZER_VERSION,
            "evidence_snapshot_sha256": snapshot_hash,
            "evidence": {
                "task_count": task_count,
                "fingerprinted_run_count": len(snapshot.get("task_runs", [])),
                "human_feedback_count": human_feedback_count,
                "human_feedback_rate": human_feedback_rate,
                "tool_evidence": snapshot.get("tool_evidence", []),
                "pipeline_trial_count": len(
                    snapshot.get("pipeline_trials", [])
                ),
            },
            "candidate_policy": policy,
            "self_optimization_constitution": constitution_reference(
                active_constitution
            ),
            "shadow_transition_readiness": shadow_readiness,
            "contextual_preferences": preferences,
            "paired_trial_recommendations": paired_trial_recommendations,
            "insufficient_contexts": insufficient,
            "release_decision": "draft_only",
            "activation_blocked": True,
            "activation_block_reason": (
                "Human approval and paired shadow validation are mandatory."
            ),
        }
    )


def write_preview(preview: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a non-active daily multi-agent policy preview."
    )
    parser.add_argument(
        "--output",
        default="task_results_history/offline_optimizer/latest_policy_preview.json",
    )
    parser.add_argument("--base-policy-id", default="baseline-v1")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--user-id", type=int)
    identity.add_argument("--username")
    parser.add_argument("--workspace-id")
    parser.add_argument("--agent-version", default=current_agent_version())
    parser.add_argument(
        "--min-matched-runs", type=int, default=DEFAULT_MIN_MATCHED_RUNS
    )
    parser.add_argument(
        "--register-draft",
        action="store_true",
        help="Register the exact candidate as draft; never activates it.",
    )
    args = parser.parse_args(argv)

    from .db_utils import DatabaseManager

    db = DatabaseManager()
    try:
        user_id = args.user_id
        if user_id is None:
            user_id = db.find_user_id(args.username)
            if not user_id:
                parser.error("username does not exist")
        snapshot = collect_evidence_snapshot(
            db,
            user_id=user_id,
            workspace_id=args.workspace_id,
            agent_version=args.agent_version,
            data_partitions=("train",),
        )
        preview = build_draft_policy_preview(
            snapshot,
            base_policy_id=args.base_policy_id,
            min_matched_runs=args.min_matched_runs,
        )
        path = write_preview(preview, args.output)
        if args.register_draft:
            db.register_policy_version(dict(preview["candidate_policy"]))
            trace_id = str(uuid.uuid4())
            artifact = build_artifact_record(
                path=path,
                artifact_type="offline_policy_preview",
                created_by="offline_optimizer",
                trace_id=trace_id,
                metadata={
                    "candidate_policy_id": preview["candidate_policy"][
                        "policy_id"
                    ],
                    "release_decision": preview["release_decision"],
                    "activation_blocked": True,
                },
            )
            artifact["validation_status"] = "draft"
            db.register_artifact(artifact)
            db.save_trajectory_event(
                build_trajectory_event(
                    event_type="policy_candidate_generated",
                    trace_id=trace_id,
                    actor_type="optimizer",
                    actor_name="offline_optimizer",
                    status="success",
                    user_id=user_id,
                    workspace_id=snapshot["scope"]["workspace_id"],
                    agent_version=snapshot["scope"]["agent_version"],
                    data_partition="train",
                    payload={
                        "candidate_policy_id": preview["candidate_policy"][
                            "policy_id"
                        ],
                        "config_sha256": preview["candidate_policy"][
                            "config_sha256"
                        ],
                        "artifact_id": artifact["artifact_id"],
                        "activation_blocked": True,
                        "self_optimization_constitution": preview[
                            "self_optimization_constitution"
                        ],
                        "shadow_transition_readiness": preview[
                            "shadow_transition_readiness"
                        ],
                    },
                )
            )
        print(
            json.dumps(
                {
                    "output": str(path),
                    "candidate_policy_id": preview["candidate_policy"]["policy_id"],
                    "registered_as_draft": bool(args.register_draft),
                    "activation_blocked": True,
                },
                ensure_ascii=False,
            )
        )
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
