"""Read-only analysis queries for the multi-agent trace data.

These functions return SQL strings or execute queries against a
``DatabaseManager`` instance.  Use them for dashboards, ad-hoc analysis,
or scheduled reporting.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Routing quality
# ---------------------------------------------------------------------------

ROUTING_CONFUSION_MATRIX = """
SELECT
    r.routed_agent,
    f.expected_agent,
    COUNT(*) AS cnt
FROM agent_routing_log r
JOIN routing_feedback f ON f.trace_id = r.trace_id
WHERE r.trace_id IS NOT NULL
GROUP BY r.routed_agent, f.expected_agent
ORDER BY cnt DESC
"""

ROUTING_ACCURACY = """
SELECT
    SUM(CASE WHEN r.routed_agent = f.expected_agent THEN 1 ELSE 0 END) AS correct,
    COUNT(*) AS total,
    ROUND(100.0 * SUM(CASE WHEN r.routed_agent = f.expected_agent THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 1) AS pct_correct
FROM agent_routing_log r
JOIN routing_feedback f ON f.trace_id = r.trace_id
WHERE r.trace_id IS NOT NULL
"""

# This measures logging/orchestration consistency only; it is not a reference-
# standard routing accuracy metric.
ROUTING_EXECUTION_CONSISTENCY = """
SELECT
    SUM(CASE WHEN r.routed_agent = a.agent_name THEN 1 ELSE 0 END) AS consistent,
    COUNT(*) AS total,
    ROUND(100.0 * SUM(CASE WHEN r.routed_agent = a.agent_name THEN 1 ELSE 0 END)
          / NULLIF(COUNT(*), 0), 1) AS pct_consistent
FROM agent_routing_log r
JOIN agent_execution_log a ON a.trace_id = r.trace_id
WHERE r.trace_id IS NOT NULL
"""

ROUTING_MISSES = """
SELECT
    r.trace_id,
    r.user_query,
    r.routed_agent,
    r.confidence,
    f.expected_agent,
    f.label_source,
    r.created_at
FROM agent_routing_log r
JOIN routing_feedback f ON f.trace_id = r.trace_id
WHERE r.routed_agent <> f.expected_agent AND r.trace_id IS NOT NULL
ORDER BY r.created_at DESC
"""

TASK_FEEDBACK_COVERAGE = """
SELECT
    COUNT(DISTINCT t.trace_id) AS task_count,
    COUNT(DISTINCT tf.trace_id) AS tasks_with_any_feedback,
    COUNT(DISTINCT CASE WHEN tf.feedback_source = 'human'
                        THEN tf.trace_id END) AS tasks_with_human_feedback,
    COUNT(DISTINCT rf.trace_id) AS tasks_with_routing_feedback,
    ROUND(
        COUNT(DISTINCT CASE WHEN tf.feedback_source = 'human'
                            THEN tf.trace_id END)
        / NULLIF(COUNT(DISTINCT t.trace_id), 0),
        6
    ) AS human_feedback_rate
FROM tasks t
LEFT JOIN task_feedback tf ON tf.trace_id = t.trace_id
LEFT JOIN routing_feedback rf ON rf.trace_id = t.trace_id
"""

TASK_OUTCOME_SUMMARY = """
SELECT
    outcome_status,
    completion_gate_status,
    COUNT(*) AS tasks
FROM tasks
WHERE outcome_status IS NOT NULL
GROUP BY outcome_status, completion_gate_status
ORDER BY tasks DESC, outcome_status
"""

# ---------------------------------------------------------------------------
# Tool performance
# ---------------------------------------------------------------------------

TOOL_ERROR_RATES = """
SELECT
    tool_name,
    status,
    error_category,
    COUNT(*) AS cnt,
    ROUND(AVG(execution_time_ms), 0) AS avg_ms,
    ROUND(MAX(execution_time_ms), 0) AS max_ms
FROM tool_calls_log
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY tool_name, status, error_category
ORDER BY cnt DESC
"""

TOOL_SLOWEST = """
SELECT
    tool_name,
    ROUND(AVG(execution_time_ms), 0) AS avg_ms,
    COUNT(*) AS calls
FROM tool_calls_log
WHERE status = 'success'
GROUP BY tool_name
ORDER BY avg_ms DESC
"""

# ---------------------------------------------------------------------------
# Agent performance
# ---------------------------------------------------------------------------

AGENT_PERFORMANCE = """
SELECT
    agent_name,
    COUNT(*) AS runs,
    ROUND(AVG(total_latency_ms), 0) AS avg_total_ms,
    ROUND(AVG(llm_latency_ms), 0) AS avg_llm_ms,
    ROUND(AVG(tool_count), 1) AS avg_tools,
    SUM(success) AS success_runs,
    SUM(1 - success) AS failed_runs
FROM agent_execution_log
WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY agent_name
ORDER BY runs DESC
"""

AGENT_ERROR_CATEGORIES = """
SELECT
    agent_name,
    error_category,
    COUNT(*) AS cnt
FROM agent_execution_log
WHERE success = 0 AND error_category IS NOT NULL
GROUP BY agent_name, error_category
ORDER BY cnt DESC
"""

# ---------------------------------------------------------------------------
# Orchestrator plan quality
# ---------------------------------------------------------------------------

ORCHESTRATOR_COMPLETION = """
SELECT
    CASE
        WHEN completed_steps = total_steps THEN 'fully_completed'
        WHEN completed_steps > 0 THEN 'partial'
        ELSE 'zero_progress'
    END AS completion_level,
    COUNT(*) AS cnt,
    ROUND(AVG(total_steps), 1) AS avg_planned_steps
FROM tasks
WHERE status IN ('completed', 'failed')
  AND created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY completion_level
"""

ORCHESTRATOR_STEP_FAILURES = """
SELECT
    ts.agent_name,
    ts.error_category,
    COUNT(*) AS cnt
FROM task_steps ts
WHERE ts.status = 'failed'
  AND ts.created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY ts.agent_name, ts.error_category
ORDER BY cnt DESC
"""

# ---------------------------------------------------------------------------
# Experiment leaderboard
# ---------------------------------------------------------------------------

EXPERIMENT_LEADERBOARD = """
SELECT
    feature_type,
    classification_method,
    COUNT(*) AS runs,
    ROUND(AVG(AUC), 4) AS avg_auc,
    ROUND(MAX(AUC), 4) AS best_auc,
    ROUND(AVG(F1), 4) AS avg_f1,
    ROUND(AVG(Accuracy), 4) AS avg_accuracy
FROM experiment_results
WHERE AUC IS NOT NULL
GROUP BY feature_type, classification_method
ORDER BY best_auc DESC
"""

EXPERIMENT_BEST_BY_TYPE = """
SELECT
    feature_type,
    classification_method,
    AUC,
    F1,
    Accuracy,
    Sensitivity,
    Specificity,
    NumPatients,
    selected_feature_count,
    output_dir
FROM experiment_results e1
WHERE AUC = (
    SELECT MAX(AUC)
    FROM experiment_results e2
    WHERE e2.feature_type = e1.feature_type
      AND e2.classification_method = e1.classification_method
)
ORDER BY AUC DESC
"""

# ---------------------------------------------------------------------------
# Versioned offline policy evaluation
# ---------------------------------------------------------------------------

POLICY_EVALUATION_LEADERBOARD = """
SELECT
    policy_id,
    task_type,
    method_name,
    split_type,
    metric_name,
    COUNT(*) AS evaluated_runs,
    ROUND(AVG(metric_value), 6) AS mean_value,
    ROUND(STDDEV_SAMP(metric_value), 6) AS sd_value,
    MIN(metric_value) AS min_value,
    MAX(metric_value) AS max_value
FROM evaluation_results
WHERE record_type = 'metric'
GROUP BY policy_id, task_type, method_name, split_type, metric_name
ORDER BY task_type, split_type, metric_name, mean_value DESC
"""

POLICY_HARD_GATE_SUMMARY = """
SELECT
    policy_id,
    task_type,
    method_name,
    metric_name AS gate_name,
    gate_status,
    COUNT(*) AS cnt
FROM evaluation_results
WHERE record_type = 'gate'
GROUP BY policy_id, task_type, method_name, metric_name, gate_status
ORDER BY policy_id, task_type, method_name, metric_name, gate_status
"""

STRICT_PROMOTION_CANDIDATES = """
SELECT
    e.policy_id,
    e.task_type,
    e.method_name,
    e.split_type,
    e.metric_name,
    COUNT(*) AS evaluated_runs,
    ROUND(AVG(e.metric_value), 6) AS mean_value
FROM evaluation_results e
WHERE e.record_type = 'metric'
  AND NOT EXISTS (
      SELECT 1
      FROM evaluation_results g
      WHERE g.policy_id = e.policy_id
        AND g.task_type = e.task_type
        AND g.method_name = e.method_name
        AND g.record_type = 'gate'
        AND g.gate_status <> 'pass'
  )
GROUP BY e.policy_id, e.task_type, e.method_name, e.split_type, e.metric_name
ORDER BY e.task_type, e.metric_name, mean_value DESC
"""

TASK_POLICY_SCORECARD = """
WITH run_metrics AS (
    SELECT
        trace_id,
        policy_id,
        task_type,
        method_name,
        MAX(CASE WHEN metric_name = 'task_success' THEN metric_value END) AS task_success,
        MAX(CASE WHEN metric_name = 'false_completion' THEN metric_value END) AS false_completion,
        MAX(CASE WHEN metric_name = 'cross_run_artifact_violation' THEN metric_value END) AS cross_run_violation,
        MAX(CASE WHEN metric_name = 'required_tool_coverage' THEN metric_value END) AS tool_coverage
    FROM evaluation_results
    WHERE evaluator_name = 'task-contract-evaluator'
      AND record_type = 'metric'
    GROUP BY trace_id, policy_id, task_type, method_name
)
SELECT
    policy_id,
    task_type,
    method_name,
    COUNT(*) AS evaluated_runs,
    ROUND(AVG(task_success), 6) AS task_success_rate,
    ROUND(AVG(false_completion), 6) AS false_completion_rate,
    ROUND(AVG(cross_run_violation), 6) AS cross_run_violation_rate,
    ROUND(AVG(tool_coverage), 6) AS mean_required_tool_coverage
FROM run_metrics
GROUP BY policy_id, task_type, method_name
ORDER BY task_type, method_name, task_success_rate DESC
"""

# Same scorecard, restricted to a deterministic comparable-task fingerprint.
# This is the query used by online planning; the global scorecard remains a
# dashboard view and must not be used to transfer performance across datasets.
MATCHED_TASK_POLICY_SCORECARD = """
WITH run_metrics AS (
    SELECT
        e.trace_id,
        e.policy_id,
        e.task_type,
        e.method_name,
        MAX(CASE WHEN e.metric_name = 'task_success'
                 THEN e.metric_value END) AS task_success,
        MAX(CASE WHEN e.metric_name = 'false_completion'
                 THEN e.metric_value END) AS false_completion,
        MAX(CASE WHEN e.metric_name = 'cross_run_artifact_violation'
                 THEN e.metric_value END) AS cross_run_violation,
        MAX(CASE WHEN e.metric_name = 'required_tool_coverage'
                 THEN e.metric_value END) AS tool_coverage
    FROM evaluation_results e
    JOIN tasks t ON t.trace_id = e.trace_id
    WHERE e.evaluator_name = 'task-contract-evaluator'
      AND e.record_type = 'metric'
      AND t.task_fingerprint = %s
    GROUP BY e.trace_id, e.policy_id, e.task_type, e.method_name
)
SELECT
    policy_id,
    task_type,
    method_name,
    COUNT(*) AS evaluated_runs,
    ROUND(AVG(task_success), 6) AS task_success_rate,
    ROUND(AVG(false_completion), 6) AS false_completion_rate,
    ROUND(AVG(cross_run_violation), 6) AS cross_run_violation_rate,
    ROUND(AVG(tool_coverage), 6) AS mean_required_tool_coverage
FROM run_metrics
GROUP BY policy_id, task_type, method_name
ORDER BY task_type, method_name, task_success_rate DESC
"""

MATCHED_PIPELINE_TRIAL_SCORECARD = """
SELECT
    task_fingerprint,
    candidate_id,
    feature_type,
    classification_method,
    model_recipe,
    COUNT(*) AS evaluated_runs,
    ROUND(AVG(validation_score), 6) AS mean_validation_score,
    ROUND(AVG(auc), 6) AS mean_auc,
    ROUND(AVG(auc_ci_lower), 6) AS mean_auc_ci_lower,
    ROUND(AVG(selection_stability), 6) AS mean_selection_stability,
    ROUND(AVG(duration_ms), 0) AS mean_duration_ms
FROM pipeline_trials
WHERE task_fingerprint = %s
  AND status = 'success'
  AND test_data_used = 0
  AND validation_scope = 'training_only_nested_grouped_cv'
GROUP BY task_fingerprint, candidate_id, feature_type,
         classification_method, model_recipe
ORDER BY mean_validation_score DESC, candidate_id
"""

TASK_HARD_GATE_FAILURES = """
SELECT
    policy_id,
    task_type,
    method_name,
    metric_name AS gate_name,
    COUNT(*) AS evaluated_runs,
    SUM(gate_status = 'fail') AS failed_runs,
    ROUND(AVG(gate_status = 'fail'), 6) AS failure_rate
FROM evaluation_results
WHERE evaluator_name = 'task-contract-evaluator'
  AND record_type = 'gate'
GROUP BY policy_id, task_type, method_name, metric_name
ORDER BY failure_rate DESC, failed_runs DESC, gate_name
"""

AUTONOMOUS_PIPELINE_SELECTIONS = """
SELECT
    JSON_UNQUOTE(JSON_EXTRACT(payload_json, '$.selected_candidate_id'))
        AS selected_candidate_id,
    COUNT(*) AS decisions,
    SUM(status = 'success') AS selected_runs,
    SUM(status = 'blocked') AS blocked_runs,
    MIN(created_at) AS first_seen,
    MAX(created_at) AS last_seen
FROM trajectory_events
WHERE event_type = 'pipeline_selected'
GROUP BY selected_candidate_id
ORDER BY decisions DESC, selected_candidate_id
"""

# ---------------------------------------------------------------------------
# Trace timeline (reconstruct one request)
# ---------------------------------------------------------------------------

TRACE_TIMELINE = """
SELECT 'routing' AS stage,
       created_at,
       CONCAT('Router → ', routed_agent, ' (confidence: ', ROUND(confidence, 2), ')') AS detail
FROM agent_routing_log WHERE trace_id = %s
UNION ALL
SELECT 'agent' AS stage,
       created_at,
       CONCAT(agent_name, ' ', IF(success, '✓', '✗'),
              ' (', total_latency_ms, 'ms, ', tool_count, ' tools)') AS detail
FROM agent_execution_log WHERE trace_id = %s
UNION ALL
SELECT 'tool' AS stage,
       created_at,
       CONCAT(tool_name, ' (', execution_time_ms, 'ms, ', status, ')') AS detail
FROM tool_calls_log WHERE trace_id = %s
ORDER BY created_at
"""

# ---------------------------------------------------------------------------
# Convenience runners
# ---------------------------------------------------------------------------


def run_query(db: Any, sql: str, params: tuple = ()) -> list[dict]:
    """Execute a read-only query and return rows as dicts."""
    with db.connection.cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchall()


def routing_confusion_matrix(db: Any) -> list[dict]:
    return run_query(db, ROUTING_CONFUSION_MATRIX)


def routing_accuracy(db: Any) -> dict | None:
    rows = run_query(db, ROUTING_ACCURACY)
    return rows[0] if rows else None


def routing_execution_consistency(db: Any) -> dict | None:
    rows = run_query(db, ROUTING_EXECUTION_CONSISTENCY)
    return rows[0] if rows else None


def routing_misses(db: Any) -> list[dict]:
    return run_query(db, ROUTING_MISSES)


def task_feedback_coverage(db: Any) -> dict | None:
    rows = run_query(db, TASK_FEEDBACK_COVERAGE)
    return rows[0] if rows else None


def task_outcome_summary(db: Any) -> list[dict]:
    return run_query(db, TASK_OUTCOME_SUMMARY)


def tool_error_rates(db: Any) -> list[dict]:
    return run_query(db, TOOL_ERROR_RATES)


def agent_performance(db: Any) -> list[dict]:
    return run_query(db, AGENT_PERFORMANCE)


def experiment_leaderboard(db: Any) -> list[dict]:
    return run_query(db, EXPERIMENT_LEADERBOARD)


def policy_evaluation_leaderboard(db: Any) -> list[dict]:
    return run_query(db, POLICY_EVALUATION_LEADERBOARD)


def policy_hard_gate_summary(db: Any) -> list[dict]:
    return run_query(db, POLICY_HARD_GATE_SUMMARY)


def task_policy_scorecard(db: Any) -> list[dict]:
    return run_query(db, TASK_POLICY_SCORECARD)


def matched_task_policy_scorecard(
    db: Any, task_fingerprint: str
) -> list[dict]:
    if not str(task_fingerprint or "").strip():
        return []
    return run_query(db, MATCHED_TASK_POLICY_SCORECARD, (task_fingerprint,))


def matched_pipeline_trial_scorecard(
    db: Any, task_fingerprint: str
) -> list[dict]:
    if not str(task_fingerprint or "").strip():
        return []
    return run_query(
        db, MATCHED_PIPELINE_TRIAL_SCORECARD, (task_fingerprint,)
    )


def task_hard_gate_failures(db: Any) -> list[dict]:
    return run_query(db, TASK_HARD_GATE_FAILURES)


def autonomous_pipeline_selections(db: Any) -> list[dict]:
    return run_query(db, AUTONOMOUS_PIPELINE_SELECTIONS)


def trace_timeline(db: Any, trace_id: str) -> list[dict]:
    return run_query(db, TRACE_TIMELINE, (trace_id, trace_id, trace_id))
