"""Fail-closed per-agent MCP tool permissions.

The orchestrator may schedule a role, but that role can only invoke the tools
required by its bounded responsibility.  This policy is deterministic and can
be tested without starting MCP servers.
"""

from __future__ import annotations

from collections.abc import Mapping


class ToolPermissionError(PermissionError):
    """Raised when an agent attempts a tool outside its allowlist."""


AGENT_TOOL_ALLOWLIST: Mapping[str, frozenset[str]] = {
    "SEGMENTATION_AGENT": frozenset(
        {
            "umamba_segment_mri",
            "lkm_unet_segment_mri",
            "lightm_unet_segment_mri",
        }
    ),
    "FEATURE_EXTRACTION_AGENT": frozenset(
        {
            "extract_radiomics_features",
            "meas_features_extraction",
            "ldse_features_extraction",
            "ldse_train_test_features_extraction",
            "augment_dataset_offline",
            "augment_online_extract_features",
            "assess_feature_quality",
            "validate_cartilage_segmentation",
            "analyze_cartilage_thickness",
            "extract_cartilage_thickness_features",
            "generate_cartilage_3d_mesh",
        }
    ),
    "FEATURE_COMBINATION_AGENT": frozenset(
        {
            "combine_radiomics_features_by_region",
            "combine_radiomics_features_all",
            "lasso_comb_radiomics_features_by_region",
            "lasso_comb_radiomics_features_all",
        }
    ),
    "CLASSIFICATION_AGENT": frozenset(
        {
            "koa_lasso_classification",
            "koa_filter_classification",
            "read_classification_results",
            "assess_feature_quality",
        }
    ),
    "PIPELINE_OPTIMIZATION_AGENT": frozenset(
        {
            "meas_features_extraction",
            "ldse_train_test_features_extraction",
            "extract_radiomics_features",
            "validate_kl_pipeline_candidate",
        }
    ),
    "WEB_SEARCH_AGENT": frozenset({"web_search"}),
    "BIOMARKER_DISCOVERY_AGENT": frozenset({"run_biomarker_discovery"}),
    "BIOMARKER_VERIFIER_AGENT": frozenset({"audit_biomarker_discovery"}),
    "BIOMARKER_REPORTER_AGENT": frozenset(),
    "ORCHESTRATOR_AGENT": frozenset(),
    "CONVERSATION_AGENT": frozenset(),
    "ROUTER_AGENT": frozenset(),
}


def ensure_tool_allowed(agent_name: str | None, tool_name: str) -> None:
    """Reject unknown roles and cross-role tool calls."""
    normalized_agent = str(agent_name or "").strip().upper()
    normalized_tool = str(tool_name or "").strip()
    if normalized_agent not in AGENT_TOOL_ALLOWLIST:
        raise ToolPermissionError(
            f"Tool call denied: unknown or missing agent context {normalized_agent!r}"
        )
    if normalized_tool not in AGENT_TOOL_ALLOWLIST[normalized_agent]:
        raise ToolPermissionError(
            f"Tool call denied: {normalized_agent} cannot invoke {normalized_tool}"
        )
