"""Deterministic, budgeted pipeline search for autonomous KL tasks."""

from __future__ import annotations

import json
import os
from time import perf_counter
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .path_utils import parse_tool_result_paths
from .pipeline_planner import (
    attach_trial_results,
    build_trial_portfolio,
    select_pipeline_from_trials,
)
from .tool_result_utils import (
    parse_structured_tool_result,
    tool_error_message,
    tool_result_succeeded,
)
from db_utils.pipeline_trials import build_pipeline_trial_record

if TYPE_CHECKING:
    from .state import AgentState


async def get_tool_manager():
    """Import the effectful MCP gateway lazily for dependency-light planning tests."""
    from .tool_manager import get_tool_manager as factory

    return await factory()


class PipelineOptimizationAgent:
    """Extract each trial feature family once and rank on training CV only."""

    def __init__(self, config=None, *, output_root: str | Path | None = None):
        self.config = config
        self.llm = None
        self.system_prompt = (
            "Deterministic training-only pipeline search; test metrics are forbidden."
        )
        self.output_root = Path(output_root) if output_root else Path(__file__).resolve().parents[1]

    @staticmethod
    def _source_paths(dataset_profile: Mapping[str, Any]) -> tuple[str, str]:
        files = dataset_profile.get("source_files", {})
        train = str(files.get("train", {}).get("path") or "")
        test = str(files.get("test", {}).get("path") or "")
        if not train or not test:
            raise ValueError("pipeline search requires current train and test source CSVs")
        return train, test

    @staticmethod
    def _first_csv(result: Any) -> str | None:
        for _, path in parse_tool_result_paths(result):
            if path.lower().endswith(".csv"):
                return path
        return None

    @staticmethod
    def _paired_csvs(result: Any) -> tuple[str | None, str | None]:
        train = None
        test = None
        for key, path in parse_tool_result_paths(result):
            lowered = key.lower()
            if "train" in lowered and path.lower().endswith(".csv"):
                train = path
            elif "test" in lowered and path.lower().endswith(".csv"):
                test = path
        return train, test

    @staticmethod
    def _atomic_manifest(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    async def _call(
        self,
        manager,
        tool_name: str,
        arguments: dict[str, Any],
        tool_calls: list[str],
        tool_results: list[Any],
        stream_callback=None,
    ) -> Any:
        tool_calls.append(tool_name)
        result = await manager.call_tool(
            tool_name,
            arguments,
            stream_callback=stream_callback,
        )
        tool_results.append(result)
        if not tool_result_succeeded(result):
            raise RuntimeError(tool_error_message(result))
        return result

    async def _extract_feature_family(
        self,
        *,
        manager,
        feature_type: str,
        train_source: str,
        test_source: str,
        run_root: Path,
        tool_calls: list[str],
        tool_results: list[Any],
        stream_callback=None,
    ) -> dict[str, str]:
        normalized = str(feature_type).strip().upper()
        if normalized == "MEAS":
            paths: dict[str, str] = {}
            for role, source in (("train", train_source), ("test", test_source)):
                result = await self._call(
                    manager,
                    "meas_features_extraction",
                    {
                        "output_dir": str(run_root / "MEAS" / role),
                        "csv_path": source,
                    },
                    tool_calls,
                    tool_results,
                    stream_callback,
                )
                path = self._first_csv(result)
                if not path:
                    raise RuntimeError(f"MEAS {role} extraction returned no feature CSV")
                paths[role] = path
            return paths
        if normalized == "LDSE":
            result = await self._call(
                manager,
                "ldse_train_test_features_extraction",
                {
                    "output_dir": str(run_root / "LDSE"),
                    "train_csv_path": train_source,
                    "test_csv_path": test_source,
                    "n_components": 50,
                    "random_seed": 42,
                },
                tool_calls,
                tool_results,
                stream_callback,
            )
            train_path, test_path = self._paired_csvs(result)
            if not train_path or not test_path:
                raise RuntimeError("LDSE extraction returned incomplete train/test feature paths")
            return {"train": train_path, "test": test_path}
        if normalized == "RADIOMICS":
            paths = {}
            for role, source in (("train", train_source), ("test", test_source)):
                result = await self._call(
                    manager,
                    "extract_radiomics_features",
                    {
                        "csv_path": source,
                        "output_dir": str(run_root / "Radiomics" / role),
                        "extract_all_cases": True,
                    },
                    tool_calls,
                    tool_results,
                    stream_callback,
                )
                path = self._first_csv(result)
                if not path:
                    raise RuntimeError(
                        f"Radiomics {role} extraction returned no feature CSV"
                    )
                paths[role] = path
            return paths
        raise ValueError(f"unsupported feature_type: {feature_type}")

    async def process(self, state: "AgentState") -> "AgentState":
        tool_calls: list[str] = []
        tool_results: list[Any] = []
        trial_results: list[dict[str, Any]] = []
        trial_durations_ms: dict[str, int] = {}
        candidates = [dict(item) for item in state.get("pipeline_candidates", [])]
        decision = dict(state.get("planning_decision", {}) or {})
        policy = dict(decision.get("search_policy", {}) or {})
        portfolio = dict(decision.get("trial_portfolio", {}) or {})
        if not portfolio:
            portfolio = build_trial_portfolio(candidates, search_policy=policy)
        candidate_ids = set(str(item) for item in portfolio.get("candidate_ids", []))
        trial_candidates = [
            item for item in candidates if str(item.get("candidate_id")) in candidate_ids
        ]
        run_id = str(state.get("task_run_id") or state.get("trace_id") or uuid.uuid4())
        run_root = self.output_root / f"Pipeline_Search_{run_id}"
        stream_callback = state.get("stream_callback")

        try:
            if portfolio.get("status") != "ready" or not trial_candidates:
                raise ValueError("pipeline trial portfolio is empty or blocked")
            train_source, test_source = self._source_paths(
                state.get("dataset_profile", {})
            )
            manager = await get_tool_manager()
            feature_paths: dict[str, dict[str, str]] = {}
            feature_errors: dict[str, str] = {}

            for feature_type in portfolio.get("feature_types", []):
                try:
                    feature_paths[str(feature_type)] = await self._extract_feature_family(
                        manager=manager,
                        feature_type=str(feature_type),
                        train_source=train_source,
                        test_source=test_source,
                        run_root=run_root,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                        stream_callback=stream_callback,
                    )
                except Exception as exc:
                    feature_errors[str(feature_type)] = f"{type(exc).__name__}: {exc}"

            for candidate in trial_candidates:
                candidate_id = str(candidate.get("candidate_id"))
                feature_type = str(candidate.get("feature_type"))
                paths = feature_paths.get(feature_type)
                if not paths:
                    trial_results.append(
                        {
                            "candidate_id": candidate_id,
                            "status": "failed",
                            "validation_scope": "training_only_nested_grouped_cv",
                            "test_data_used": False,
                            "error": feature_errors.get(
                                feature_type, "feature extraction unavailable"
                            ),
                        }
                    )
                    continue
                try:
                    trial_started = perf_counter()
                    raw = await self._call(
                        manager,
                        "validate_kl_pipeline_candidate",
                        {
                            "train_path": paths["train"],
                            "classification_method": candidate.get(
                                "classification_method"
                            ),
                            "model_recipe": candidate.get("model_recipe"),
                            "outer_splits": 3,
                            "inner_splits": 3,
                            "bootstrap_samples": 500,
                        },
                        tool_calls,
                        tool_results,
                        stream_callback,
                    )
                    parsed = parse_structured_tool_result(raw)
                    if not isinstance(parsed, dict):
                        raise RuntimeError("validation tool returned an unstructured result")
                    trial_results.append({"candidate_id": candidate_id, **parsed})
                except Exception as exc:
                    trial_results.append(
                        {
                            "candidate_id": candidate_id,
                            "status": "failed",
                            "validation_scope": "training_only_nested_grouped_cv",
                            "test_data_used": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                finally:
                    if "trial_started" in locals():
                        trial_durations_ms[candidate_id] = int(
                            (perf_counter() - trial_started) * 1000
                        )
                        del trial_started

            evaluated_candidates = attach_trial_results(
                candidates, trial_results, search_policy=policy
            )
            evaluated_by_id = {
                str(item.get("candidate_id")): item
                for item in evaluated_candidates
            }
            pipeline_trial_records: list[dict[str, Any]] = []
            trial_record_errors: list[dict[str, str]] = []
            for raw_result in trial_results:
                candidate_id = str(raw_result.get("candidate_id"))
                evaluated = evaluated_by_id.get(candidate_id)
                if not evaluated:
                    continue
                try:
                    pipeline_trial_records.append(
                        build_pipeline_trial_record(
                            trace_id=run_id,
                            task_id=state.get("task_id"),
                            task_fingerprint=(
                                state.get("task_fingerprint")
                                or state.get("task_spec", {}).get(
                                    "task_fingerprint"
                                )
                            ),
                            dataset_hash=state.get("dataset_profile", {}).get(
                                "dataset_hash"
                            ),
                            policy_id=state.get("policy_id"),
                            candidate=evaluated,
                            result=raw_result,
                            duration_ms=trial_durations_ms.get(candidate_id),
                        )
                    )
                except Exception as exc:
                    trial_record_errors.append(
                        {
                            "candidate_id": candidate_id,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            selected = select_pipeline_from_trials(
                evaluated_candidates, search_policy=policy
            )
            if selected is None:
                raise RuntimeError("no candidate passed training-only validation")
            selected_paths = feature_paths.get(str(selected.get("feature_type")))
            if not selected_paths:
                raise RuntimeError("selected candidate is missing feature artifacts")

            manifest_path = run_root / "pipeline_search_manifest.json"
            manifest = {
                "schema_version": "1.0",
                "run_id": run_id,
                "selection_policy_version": policy.get("version"),
                "portfolio": portfolio,
                "test_labels_or_metrics_used_for_selection": False,
                "trial_results": trial_results,
                "pipeline_trial_records": pipeline_trial_records,
                "trial_record_errors": trial_record_errors,
                "selected_candidate_id": selected.get("candidate_id"),
                "selected_feature_type": selected.get("feature_type"),
                "selected_classification_method": selected.get(
                    "classification_method"
                ),
                "selected_model_recipe": selected.get("model_recipe"),
                "selected_recommended_params": selected.get(
                    "validation_recommended_params", {}
                ),
                "selected_feature_paths": selected_paths,
            }
            self._atomic_manifest(manifest_path, manifest)

            artifacts = []
            for feature_type, paths in feature_paths.items():
                for role, path in paths.items():
                    artifacts.append(
                        {
                            "schema_version": "1.0",
                            "run_id": run_id,
                            "feature_type": feature_type,
                            "dataset_role": role,
                            "path": path,
                            "producer_agent": "PIPELINE_OPTIMIZATION_AGENT",
                            "producer_tool": "budgeted_feature_extraction",
                        }
                    )
            selected_artifacts = [
                item
                for item in artifacts
                if item.get("feature_type") == selected.get("feature_type")
            ]
            successful_trials = [
                item for item in evaluated_candidates if item.get("trial_status") == "success"
            ]
            output = (
                f"训练集候选试跑完成：{len(successful_trials)}/{len(trial_candidates)} 个候选有效；"
                f"选择 {selected.get('candidate_id')}，训练集验证得分 "
                f"{selected.get('validation_score')}。候选排序未使用测试集标签或测试指标。"
            )
            return {
                **state,
                "output": output,
                "agent_name": "PIPELINE_OPTIMIZATION_AGENT",
                "success": True,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "pipeline_candidates": evaluated_candidates,
                "selected_pipeline": selected,
                "trial_results": trial_results,
                "pipeline_trial_records": pipeline_trial_records,
                "trial_record_errors": trial_record_errors,
                "selected_feature_paths": selected_paths,
                "pipeline_search_manifest_path": str(manifest_path),
                "current_artifacts": [
                    *(state.get("current_artifacts", []) or []), *selected_artifacts
                ],
            }
        except Exception as exc:
            return {
                **state,
                "output": f"候选方案训练集试跑失败：{type(exc).__name__}: {exc}",
                "agent_name": "PIPELINE_OPTIMIZATION_AGENT",
                "success": False,
                "error_message": str(exc),
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "trial_results": trial_results,
            }
