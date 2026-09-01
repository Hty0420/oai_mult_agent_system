"""MCP server for validated OAI-ZIB cartilage thickness analysis."""

import asyncio
import os
import sys
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithm.cartilage_thickness.service import (  # noqa: E402
    analyze_cartilage_thickness_file,
    extract_cartilage_thickness_features_file,
    generate_cartilage_3d_mesh_file,
    validate_cartilage_segmentation_file,
)


mcp = FastMCP("cartilage_analysis")


def _failed(exc: Exception) -> dict:
    return {
        "status": "failed",
        "message": str(exc),
        "files": [],
    }


@mcp.tool()
def validate_cartilage_segmentation(
    mask_path: Annotated[
        str,
        Field(description="待校验的 OAI-ZIB 六分类膝关节分割 NIfTI 路径（.nii/.nii.gz）。"),
    ],
) -> dict:
    """校验分割标签、几何方向、spacing 和软骨/骨结构是否适合厚度分析。"""
    try:
        return validate_cartilage_segmentation_file(mask_path)
    except Exception as exc:
        return _failed(exc)


@mcp.tool()
def analyze_cartilage_thickness(
    mask_path: Annotated[
        str,
        Field(description="OAI-ZIB 六分类膝关节分割 NIfTI 路径。"),
    ],
    output_dir: Annotated[
        str,
        Field(description="必须不存在或为空的结果目录，用于保存 CSV、JSON 和 NIfTI 厚度图。"),
    ],
    export_thickness_maps: Annotated[
        bool,
        Field(description="是否导出股骨、内侧胫骨和外侧胫骨软骨厚度 NIfTI 图。"),
    ] = True,
    export_subregion_mask: Annotated[
        bool,
        Field(description="是否额外导出编码为 1-16 的亚区标签 NIfTI。"),
    ] = False,
) -> dict:
    """执行 3D-MN 面积加权软骨厚度和 Wirth/Eckstein 16 亚区联合分析。"""
    try:
        return analyze_cartilage_thickness_file(
            mask_path,
            output_dir,
            export_thickness_maps=export_thickness_maps,
            export_subregion_mask=export_subregion_mask,
        )
    except Exception as exc:
        return _failed(exc)


@mcp.tool()
def generate_cartilage_3d_mesh(
    mask_path: Annotated[
        str,
        Field(description="OAI-ZIB 六分类膝关节分割 NIfTI 路径，不是原始灰度 MRI。"),
    ],
    output_dir: Annotated[
        str,
        Field(description="必须不存在或为空的三维网格输出目录。"),
    ],
    label_ids: Annotated[
        str,
        Field(
            description=(
                "要建模的标签，以逗号分隔：1股骨、2股骨软骨、3胫骨、"
                "4内侧胫骨软骨、5外侧胫骨软骨。"
            )
        ),
    ] = "1,2,3,4,5",
    smooth: Annotated[
        bool,
        Field(description="是否应用高斯与拉普拉斯平滑；默认生成平滑表面。"),
    ] = True,
    output_formats: Annotated[
        str,
        Field(description="导出格式，支持 ply、stl、npz，可用逗号组合。"),
    ] = "ply,npz",
) -> dict:
    """从六标签膝关节分割生成物理空间中的分标签三维三角网格。"""
    try:
        return generate_cartilage_3d_mesh_file(
            mask_path,
            output_dir,
            label_ids=label_ids,
            smooth=smooth,
            output_formats=output_formats,
        )
    except Exception as exc:
        return _failed(exc)


@mcp.tool()
async def extract_cartilage_thickness_features(
    csv_path: Annotated[
        str,
        Field(description="包含 CaseID、MaskPath，可选 SourceCaseID、KLGrade 列的病例 CSV。"),
    ],
    output_dir: Annotated[
        str,
        Field(description="必须不存在或为空的患者级厚度特征输出目录。"),
    ],
    ctx: Context,
) -> dict:
    """批量提取全局 3D-MN 与 16 亚区患者级厚度特征，输出分类可用 CSV。"""
    try:
        loop = asyncio.get_running_loop()

        def forward_progress(event: dict) -> None:
            processed = int(event.get("processed_cases", 0))
            total = int(event.get("total_cases", 0))
            successful = int(event.get("successful_cases", 0))
            failed = int(event.get("failed_cases", 0))
            case_id = event.get("case_id")
            detail = f"已处理 {processed}/{total}；成功 {successful}；失败 {failed}"
            if case_id:
                detail += f"；当前 CaseID={case_id}"
            future = asyncio.run_coroutine_threadsafe(
                ctx.report_progress(
                    float(processed),
                    float(total) if total else None,
                    detail,
                ),
                loop,
            )

            def consume_report_result(completed) -> None:
                try:
                    completed.result()
                except Exception:
                    pass

            future.add_done_callback(consume_report_result)

        await ctx.report_progress(0.0, None, "正在读取批量病例 CSV")
        return await asyncio.to_thread(
            extract_cartilage_thickness_features_file,
            csv_path,
            output_dir,
            progress_callback=forward_progress,
        )
    except Exception as exc:
        return _failed(exc)


if __name__ == "__main__":
    mcp.run(transport="stdio")
