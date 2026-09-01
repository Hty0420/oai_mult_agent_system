"""MCP tools for deterministic KL-ordinal MRI biomarker discovery."""

import asyncio
import os
import sys
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm.biomarker_discovery.service import (  # noqa: E402
    audit_biomarker_discovery as audit_service,
    run_biomarker_discovery as discovery_service,
)


mcp = FastMCP("biomarker_discovery")


def _failed(exc: Exception) -> dict:
    return {"status": "failed", "message": str(exc), "files": []}


@mcp.tool()
async def run_biomarker_discovery(
    ctx: Context,
    train_feature_path: Annotated[
        str,
        Field(description="训练集患者/膝关节级 MRI 特征 CSV；候选筛选仅在此表进行。"),
    ],
    test_feature_path: Annotated[
        str,
        Field(description="冻结测试集特征 CSV；仅用于确认训练阶段冻结的候选。"),
    ],
    output_dir: Annotated[
        str,
        Field(description="必须不存在或为空的工作区内输出目录。"),
    ],
    endpoint: Annotated[
        str,
        Field(description="有序结局列，V1 固定为 KLGrade。"),
    ] = "KLGrade",
    id_column: Annotated[
        str,
        Field(description="患者/膝关节级唯一标识列。"),
    ] = "CaseID",
    covariates: Annotated[
        str,
        Field(description="逗号分隔的预设调整变量。缺失变量会记录但不会伪造。"),
    ] = "Age,Gender,BMI,KneeSide",
    fdr_alpha: Annotated[
        float,
        Field(description="训练集 Benjamini-Hochberg FDR 阈值。", gt=0.0, lt=1.0),
    ] = 0.05,
    confirmation_alpha: Annotated[
        float,
        Field(description="冻结测试集 Holm 校正确认阈值。", gt=0.0, lt=1.0),
    ] = 0.05,
    correlation_threshold: Annotated[
        float,
        Field(description="训练集候选去冗余的 Spearman 相关阈值。", gt=0.0, le=1.0),
    ] = 0.90,
    maximum_candidates: Annotated[
        int,
        Field(description="冻结到测试集的最大候选数量。", ge=1, le=100),
    ] = 20,
    bootstrap_iterations: Annotated[
        int,
        Field(description="训练集候选方向稳定性 bootstrap 次数。", ge=10, le=2000),
    ] = 100,
    bootstrap_support_threshold: Annotated[
        float,
        Field(description="候选方向稳定性门槛。", ge=0.0, le=1.0),
    ] = 0.70,
    random_seed: Annotated[
        int,
        Field(description="可复现随机种子。"),
    ] = 42,
) -> dict:
    """发现与 KL 0-4 相关的候选 MRI 标志物并在冻结测试集确认。"""
    try:
        loop = asyncio.get_running_loop()

        def forward_progress(progress, total, message):
            future = asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress, total, message),
                loop,
            )

            def consume(completed):
                try:
                    completed.result()
                except Exception:
                    pass

            future.add_done_callback(consume)

        return await asyncio.to_thread(
            discovery_service,
            train_feature_path,
            test_feature_path,
            output_dir,
            endpoint=endpoint,
            id_column=id_column,
            covariates=covariates,
            fdr_alpha=fdr_alpha,
            confirmation_alpha=confirmation_alpha,
            correlation_threshold=correlation_threshold,
            maximum_candidates=maximum_candidates,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_support_threshold=bootstrap_support_threshold,
            random_seed=random_seed,
            progress_callback=forward_progress,
        )
    except Exception as exc:
        return _failed(exc)


@mcp.tool()
def audit_biomarker_discovery(
    manifest_path: Annotated[
        str,
        Field(description="run_biomarker_discovery 输出的 biomarker_manifest.json。"),
    ],
    output_dir: Annotated[
        str,
        Field(description="必须不存在或为空的独立审计输出目录。"),
    ],
) -> dict:
    """独立核验哈希、候选冻结、多重检验和报告计数。"""
    try:
        return audit_service(manifest_path, output_dir)
    except Exception as exc:
        return _failed(exc)


if __name__ == "__main__":
    mcp.run(transport="stdio")
