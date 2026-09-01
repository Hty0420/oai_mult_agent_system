"""Deterministic control plane for self-optimization policy candidates.

The optimizer may propose bounded, shadow-only method preferences.  This
module owns the immutable constitution reference, validates candidate records,
and decides whether a lifecycle transition is ready.  It deliberately performs
no database mutation and cannot activate a policy by itself.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .trajectory import sanitize_for_audit, stable_json_hash


DEFAULT_CONSTITUTION_PATH = (
    Path(__file__).resolve().parents[1]
    / "evaluation"
    / "self_optimization_constitution.v1.json"
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREFERENCE_KEYS = frozenset(
    {"preferred_method", "evidence", "use_mode"}
)
_EVIDENCE_KEYS = frozenset(
    {
        "method_name",
        "matched_runs",
        "task_success_rate",
        "false_completion_rate",
        "cross_run_violation_rate",
        "mean_tool_coverage",
        "failed_gate_runs",
        "scientific_verified_runs",
        "scientific_incomplete_runs",
        "mean_latency_ms",
    }
)
_CANDIDATE_CONFIG_KEYS = frozenset(
    {
        "optimizer_version",
        "base_policy_id",
        "evidence_snapshot_sha256",
        "evidence_scope",
        "min_matched_runs",
        "contextual_method_preferences",
        "shadow_eligible",
        "production_activation_allowed",
        "required_next_gate",
        "self_optimization_constitution",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_utc_timestamp(value: Any, field: str) -> datetime:
    _require(isinstance(value, str) and value.strip(), f"{field} is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    _require(parsed.tzinfo is not None, f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_self_optimization_constitution(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a detached constitution object."""
    constitution = deepcopy(dict(raw))
    required = {
        "schema_version",
        "constitution_id",
        "version",
        "status",
        "immutable_controls",
        "candidate_contract",
        "lifecycle",
        "evaluation",
    }
    _require(required.issubset(constitution), "constitution is missing required fields")
    _require(constitution["schema_version"] == "1.0", "unsupported constitution schema")
    _require(constitution["status"] == "active", "constitution must be active")
    _require(
        isinstance(constitution["constitution_id"], str)
        and bool(constitution["constitution_id"].strip()),
        "constitution_id is required",
    )

    controls = constitution["immutable_controls"]
    required_true_controls = {
        "research_only",
        "autonomous_diagnosis_prohibited",
        "treatment_recommendation_prohibited",
        "patient_specific_decision_prohibited",
        "external_write_requires_scoped_approval",
        "tool_permissions_are_not_optimizer_mutable",
        "hard_gates_are_not_optimizer_mutable",
        "evaluation_sets_are_not_optimizer_mutable",
        "self_activation_prohibited",
        "final_test_tuning_prohibited",
    }
    _require(isinstance(controls, Mapping), "immutable_controls must be an object")
    _require(
        all(controls.get(name) is True for name in required_true_controls),
        "all immutable safety controls must be enabled",
    )

    contract = constitution["candidate_contract"]
    _require(isinstance(contract, Mapping), "candidate_contract must be an object")
    methods = contract.get("allowed_methods")
    _require(
        isinstance(methods, list)
        and bool(methods)
        and len(methods) == len(set(methods))
        and all(isinstance(method, str) and method for method in methods),
        "candidate_contract.allowed_methods must be unique non-empty strings",
    )
    _require(
        contract.get("production_activation_allowed") is False,
        "optimizer-generated candidates must not permit production activation",
    )
    _require(
        int(contract.get("minimum_comparable_methods") or 0) >= 2,
        "minimum_comparable_methods must be at least 2",
    )
    _require(
        int(contract.get("minimum_matched_runs_per_method") or 0) >= 2,
        "minimum_matched_runs_per_method must be at least 2",
    )
    _require(
        contract.get("required_learning_partition") == "train",
        "self-optimization must be restricted to the train trajectory partition",
    )
    _require(
        set(contract.get("required_scope_fields") or ())
        == {"user_id", "workspace_id", "agent_version", "data_partitions"},
        "candidate_contract must require the complete learning scope",
    )

    lifecycle = constitution["lifecycle"]
    transitions = lifecycle.get("transitions") if isinstance(lifecycle, Mapping) else None
    _require(isinstance(transitions, Mapping), "lifecycle.transitions must be an object")
    known_states = set(transitions)
    _require(
        known_states == {"draft", "shadow", "canary", "active", "rolled_back", "retired"},
        "lifecycle must define every supported policy state",
    )
    for source, targets in transitions.items():
        _require(
            isinstance(targets, list)
            and all(target in known_states for target in targets),
            f"invalid lifecycle targets for {source}",
        )
    _require("active" not in transitions["draft"], "draft policies cannot skip to active")
    _require("active" not in transitions["shadow"], "shadow policies cannot skip canary")

    required_gates = lifecycle.get("required_gates")
    _require(isinstance(required_gates, Mapping), "lifecycle.required_gates must be an object")
    for state in ("shadow", "canary", "active", "retired"):
        gates = required_gates.get(state)
        _require(
            isinstance(gates, list)
            and len(gates) == len(set(gates))
            and all(isinstance(gate, str) and gate for gate in gates),
            f"required gates for {state} must be unique strings",
        )
    return constitution


def load_self_optimization_constitution(
    path: str | Path = DEFAULT_CONSTITUTION_PATH,
) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_self_optimization_constitution(raw)


def constitution_reference(
    constitution: Mapping[str, Any],
) -> dict[str, str]:
    validated = validate_self_optimization_constitution(constitution)
    return {
        "constitution_id": str(validated["constitution_id"]),
        "version": str(validated["version"]),
        "sha256": stable_json_hash(validated),
    }


def validate_candidate_policy_config(
    config: Mapping[str, Any],
    constitution: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject any optimizer output outside the constitution's narrow schema."""
    validated_constitution = validate_self_optimization_constitution(constitution)
    candidate = deepcopy(dict(config))
    _require(
        set(candidate) == _CANDIDATE_CONFIG_KEYS,
        "candidate config contains missing or optimizer-forbidden fields",
    )
    expected_reference = constitution_reference(validated_constitution)
    _require(
        candidate["self_optimization_constitution"] == expected_reference,
        "candidate constitution reference does not match the active constitution",
    )
    _require(
        candidate["production_activation_allowed"] is False,
        "optimizer-generated candidates cannot enable production activation",
    )
    _require(
        isinstance(candidate["base_policy_id"], str)
        and bool(candidate["base_policy_id"].strip()),
        "base_policy_id is required",
    )
    _require(
        isinstance(candidate["evidence_snapshot_sha256"], str)
        and bool(_SHA256.fullmatch(candidate["evidence_snapshot_sha256"])),
        "evidence snapshot must have a SHA-256 digest",
    )

    contract = validated_constitution["candidate_contract"]
    scope = candidate["evidence_scope"]
    _require(isinstance(scope, Mapping), "evidence_scope must be an object")
    _require(
        set(scope) == set(contract["required_scope_fields"]),
        "evidence_scope must contain the complete tenant/version/partition scope",
    )
    _require(
        isinstance(scope["user_id"], int)
        and not isinstance(scope["user_id"], bool)
        and scope["user_id"] > 0,
        "evidence_scope.user_id must be a positive integer",
    )
    for field in ("workspace_id", "agent_version"):
        _require(
            isinstance(scope[field], str) and bool(scope[field].strip()),
            f"evidence_scope.{field} is required",
        )
    _require(
        scope["data_partitions"] == [contract["required_learning_partition"]],
        "evidence_scope may only use the train trajectory partition",
    )
    min_runs = candidate["min_matched_runs"]
    _require(
        isinstance(min_runs, int)
        and not isinstance(min_runs, bool)
        and min_runs >= int(contract["minimum_matched_runs_per_method"]),
        "candidate min_matched_runs is below the constitutional floor",
    )
    preferences = candidate["contextual_method_preferences"]
    _require(isinstance(preferences, Mapping), "contextual preferences must be an object")
    allowed_methods = set(contract["allowed_methods"])
    for fingerprint, preference in preferences.items():
        _require(
            isinstance(fingerprint, str) and bool(_SHA256.fullmatch(fingerprint)),
            "contextual preference keys must be task SHA-256 fingerprints",
        )
        _require(isinstance(preference, Mapping), "contextual preference must be an object")
        _require(
            set(preference) == _PREFERENCE_KEYS,
            "contextual preference contains unknown fields",
        )
        _require(
            preference["preferred_method"] in allowed_methods,
            "preferred method is not constitutionally allowed",
        )
        _require(
            preference["use_mode"] == contract["allowed_use_mode"],
            "optimizer preferences must remain shadow-only",
        )
        evidence = preference["evidence"]
        _require(
            isinstance(evidence, list)
            and len(evidence) >= int(contract["minimum_comparable_methods"]),
            "preference needs enough comparable methods",
        )
        seen_methods: set[str] = set()
        for method in evidence:
            _require(isinstance(method, Mapping), "method evidence must be an object")
            _require(set(method) == _EVIDENCE_KEYS, "method evidence has an invalid schema")
            method_name = method["method_name"]
            _require(method_name in allowed_methods, "evidence method is not allowed")
            _require(method_name not in seen_methods, "method evidence must be unique")
            seen_methods.add(method_name)
            _require(
                int(method["matched_runs"]) >= min_runs,
                "method evidence is below min_matched_runs",
            )
            _require(float(method["false_completion_rate"]) == 0.0, "unsafe false completion evidence")
            _require(float(method["cross_run_violation_rate"]) == 0.0, "unsafe cross-run evidence")
            _require(int(method["failed_gate_runs"]) == 0, "hard-gate failures cannot support a preference")
            _require(
                int(method["scientific_verified_runs"])
                == int(method["matched_runs"]),
                "every preference run requires verified scientific evidence",
            )
            _require(
                int(method["scientific_incomplete_runs"]) == 0,
                "incomplete scientific evidence cannot support a preference",
            )
        _require(
            preference["preferred_method"] in seen_methods,
            "preferred method must be present in the evidence",
        )
    _require(
        candidate["shadow_eligible"] is bool(preferences),
        "shadow_eligible must reflect whether safe preferences exist",
    )
    return sanitize_for_audit(candidate)


def validate_candidate_policy_record(
    record: Mapping[str, Any],
    constitution: Mapping[str, Any],
    *,
    require_draft: bool = True,
) -> dict[str, Any]:
    candidate = deepcopy(dict(record))
    required = {
        "policy_id",
        "policy_type",
        "version",
        "status",
        "config",
        "config_sha256",
        "created_by",
    }
    _require(required.issubset(candidate), "candidate policy record is incomplete")
    contract = validate_self_optimization_constitution(constitution)["candidate_contract"]
    _require(candidate["policy_type"] == contract["policy_type"], "candidate policy_type is not allowed")
    if require_draft:
        _require(candidate["status"] == "draft", "optimizer may register draft policies only")
    else:
        _require(
            candidate["status"]
            in validate_self_optimization_constitution(constitution)["lifecycle"][
                "transitions"
            ],
            "candidate has an unknown lifecycle status",
        )
    validated_config = validate_candidate_policy_config(candidate["config"], constitution)
    _require(
        candidate["config_sha256"] == stable_json_hash(validated_config),
        "candidate config hash does not match its sanitized config",
    )
    candidate["config"] = validated_config
    return sanitize_for_audit(candidate)


def validate_policy_approval(
    approval: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    target_status: str,
    constitution: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate an authenticated, exact-hash, single-use approval packet."""
    packet = deepcopy(dict(approval))
    required = {
        "approval_id",
        "authenticated_control_event",
        "approver_id",
        "approver_role",
        "scope",
        "policy_id",
        "config_sha256",
        "constitution_sha256",
        "target_status",
        "action_nonce",
        "issued_at",
        "expires_at",
        "single_use",
        "consumed_at",
    }
    _require(set(packet) == required, "approval packet has a missing or unknown field")
    validated_constitution = validate_self_optimization_constitution(constitution)
    lifecycle = validated_constitution["lifecycle"]
    _require(packet["authenticated_control_event"] is True, "approval must be an authenticated control event")
    _require(packet["scope"] == lifecycle["approval_scope"], "approval scope mismatch")
    _require(packet["approver_role"] in lifecycle["authorized_approver_roles"], "approver role is not authorized")
    _require(packet["policy_id"] == candidate["policy_id"], "approval policy mismatch")
    _require(packet["config_sha256"] == candidate["config_sha256"], "approval config hash mismatch")
    _require(packet["constitution_sha256"] == constitution_reference(validated_constitution)["sha256"], "approval constitution mismatch")
    _require(packet["target_status"] == target_status, "approval target status mismatch")
    _require(isinstance(packet["action_nonce"], str) and len(packet["action_nonce"]) >= 32, "approval action_nonce is too short")
    _require(packet["single_use"] is True and packet["consumed_at"] is None, "approval is not unused single-use authorization")
    issued = _parse_utc_timestamp(packet["issued_at"], "issued_at")
    expires = _parse_utc_timestamp(packet["expires_at"], "expires_at")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    _require(issued <= current < expires, "approval is not currently valid")
    _require(expires > issued, "approval expiry must follow issue time")
    return sanitize_for_audit(packet)


def assess_policy_transition(
    *,
    candidate: Mapping[str, Any],
    current_status: str,
    target_status: str,
    evidence: Mapping[str, Any] | None,
    approval: Mapping[str, Any] | None,
    constitution: Mapping[str, Any],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return an auditable readiness decision without mutating policy state."""
    validated_constitution = validate_self_optimization_constitution(constitution)
    validated_candidate = validate_candidate_policy_record(
        candidate,
        validated_constitution,
        require_draft=False,
    )
    lifecycle = validated_constitution["lifecycle"]
    allowed_targets = lifecycle["transitions"].get(current_status)
    _require(allowed_targets is not None, "unknown current policy status")
    transition_allowed = target_status in allowed_targets
    observed = dict(evidence or {})
    required_gates = list(lifecycle["required_gates"].get(target_status, []))
    missing_gates = sorted(gate for gate in required_gates if observed.get(gate) is not True)
    approval_required = target_status in lifecycle["approval_required_for"]
    approval_error: str | None = None
    approval_valid = not approval_required
    if approval_required:
        if approval is None:
            approval_error = "authenticated_scoped_approval_required"
        else:
            try:
                validate_policy_approval(
                    approval,
                    candidate=validated_candidate,
                    target_status=target_status,
                    constitution=validated_constitution,
                    now=now,
                )
                approval_valid = True
            except ValueError as exc:
                approval_error = str(exc)
    if target_status == "rolled_back" and not str(observed.get("rollback_reason") or "").strip():
        missing_gates.append("rollback_reason")
    authorized = transition_allowed and not missing_gates and approval_valid
    return {
        "schema_version": "1.0",
        "policy_id": validated_candidate["policy_id"],
        "config_sha256": validated_candidate["config_sha256"],
        "constitution": constitution_reference(validated_constitution),
        "current_status": current_status,
        "target_status": target_status,
        "transition_allowed": transition_allowed,
        "required_gates": required_gates,
        "missing_gates": sorted(set(missing_gates)),
        "approval_required": approval_required,
        "approval_valid": approval_valid,
        "approval_error": approval_error,
        "authorized": authorized,
        "decision": "authorized" if authorized else "blocked",
    }
