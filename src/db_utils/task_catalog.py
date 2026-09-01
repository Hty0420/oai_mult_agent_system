"""Versioned development task catalog for routing and workflow evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.task_contracts import infer_task_spec, required_agent_for_task
from db_utils.trajectory import stable_json_hash


DEFAULT_TASK_CATALOG = (
    Path(__file__).resolve().parents[1] / "evaluation" / "task_catalog.v1.json"
)


def load_task_catalog(path: str | Path = DEFAULT_TASK_CATALOG) -> dict[str, Any]:
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    validate_task_catalog(payload)
    return payload


def validate_task_catalog(catalog: dict[str, Any]) -> None:
    required_root = {
        "schema_version", "catalog_id", "version", "status",
        "intended_split", "reference_standard", "tasks",
    }
    missing_root = required_root.difference(catalog)
    if missing_root:
        raise ValueError(f"task catalog missing fields: {sorted(missing_root)}")
    if catalog["schema_version"] != "1.0":
        raise ValueError("unsupported task catalog schema_version")
    if catalog["status"] not in {"draft", "frozen", "retired"}:
        raise ValueError("task catalog status must be draft, frozen, or retired")
    if catalog["intended_split"] not in {"development", "validation", "test"}:
        raise ValueError("invalid intended_split")
    if catalog["status"] == "frozen" and not catalog.get("frozen_sha256"):
        raise ValueError("a frozen catalog requires frozen_sha256")

    tasks = catalog.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task catalog tasks must be a non-empty list")
    seen_ids: set[str] = set()
    required_task = {
        "task_id", "query", "expected_entry_agent", "expected_agents",
        "required_tool_groups", "forbidden_tools", "expected_terminal_status",
        "expected_task_success", "scenario", "tags",
    }
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"task {index} must be an object")
        missing = required_task.difference(task)
        if missing:
            raise ValueError(f"task {index} missing fields: {sorted(missing)}")
        task_id = str(task["task_id"]).strip()
        if not task_id or task_id in seen_ids:
            raise ValueError(f"duplicate or empty task_id: {task_id!r}")
        seen_ids.add(task_id)
        if task["expected_terminal_status"] not in {"success", "failed"}:
            raise ValueError(f"task {task_id} has invalid expected_terminal_status")
        if not isinstance(task["expected_task_success"], bool):
            raise ValueError(f"task {task_id} expected_task_success must be boolean")
        if not isinstance(task["expected_agents"], list):
            raise ValueError(f"task {task_id} expected_agents must be a list")
        if not all(
            isinstance(group, list) and group and all(isinstance(name, str) for name in group)
            for group in task["required_tool_groups"]
        ):
            raise ValueError(f"task {task_id} has invalid required_tool_groups")
        inferred = infer_task_spec(str(task["query"]))
        expected_entry = required_agent_for_task(inferred) or "CONVERSATION_AGENT"
        if task["expected_entry_agent"] != expected_entry:
            raise ValueError(
                f"task {task_id} expected_entry_agent={task['expected_entry_agent']} "
                f"does not match inferred {expected_entry}"
            )


def catalog_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    tasks = catalog["tasks"]
    return {
        "catalog_id": catalog["catalog_id"],
        "version": catalog["version"],
        "status": catalog["status"],
        "intended_split": catalog["intended_split"],
        "task_count": len(tasks),
        "expected_success_count": sum(task["expected_task_success"] for task in tasks),
        "expected_failure_count": sum(not task["expected_task_success"] for task in tasks),
        "content_sha256": stable_json_hash(catalog),
        "tag_counts": {
            tag: sum(tag in task["tags"] for task in tasks)
            for tag in sorted({tag for task in tasks for tag in task["tags"]})
        },
    }


if __name__ == "__main__":
    print(json.dumps(catalog_summary(load_task_catalog()), ensure_ascii=False, indent=2))
