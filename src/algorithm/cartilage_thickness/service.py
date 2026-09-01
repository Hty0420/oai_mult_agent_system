"""File-oriented service boundary for cartilage thickness MCP tools.

The numerical modules operate on NumPy arrays in OAI-ZIB index order
``[D, H, W]``.  This boundary deliberately owns NIfTI axis conversion,
validation, JSON/CSV serialization, and artifact export so raw arrays never
cross the MCP protocol.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk

from .subregion import ALL_SUBREGIONS, compute_subregion_thickness
from .thickness import compute_cartilage_thickness
from .mesh import build_mesh


SUPPORTED_LABELS = {0, 1, 2, 3, 4, 5}
LABEL_NAMES = {
    0: "Background",
    1: "Femur",
    2: "Femoral Cartilage",
    3: "Tibia",
    4: "Medial Tibial Cartilage",
    5: "Lateral Tibial Cartilage",
}
CARTILAGE_BONE_PAIRS = {2: 1, 4: 3, 5: 3}
COMPARTMENT_CODES = {2: "FC", 4: "MTC", 5: "LTC"}
MESH_FILE_CODES = {
    1: "femur",
    2: "femoral_cartilage",
    3: "tibia",
    4: "medial_tibial_cartilage",
    5: "lateral_tibial_cartilage",
}
MESH_COLORS_RGB = {
    1: (255, 15, 5),
    2: (5, 242, 46),
    3: (10, 71, 255),
    4: (255, 10, 230),
    5: (255, 224, 5),
}
GLOBAL_FEATURES = (
    "mean", "std", "median", "p25", "p75", "p90", "p95", "max",
    "total_volume_mm3", "n_vertices", "n_outliers",
)
SUBREGION_FEATURES = (
    "mean", "std", "median", "p25", "p75", "p90", "p95",
)


class CartilageAnalysisError(ValueError):
    """Expected validation or input error suitable for a tool response."""


def _resolved_file(path: str | os.PathLike[str]) -> Path:
    candidate = Path(str(path)).expanduser().resolve()
    if not candidate.is_file():
        raise CartilageAnalysisError(f"文件不存在或不是普通文件: {candidate}")
    if not str(candidate).lower().endswith((".nii", ".nii.gz")):
        raise CartilageAnalysisError("仅支持 .nii 或 .nii.gz 分割文件。")
    return candidate


def _prepare_output_dir(path: str | os.PathLike[str]) -> Path:
    output_dir = Path(str(path)).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CartilageAnalysisError(
            f"输出目录必须不存在或为空，避免混入旧结果: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_mask(path: str | os.PathLike[str]) -> tuple[Path, sitk.Image, np.ndarray, tuple[float, float, float]]:
    mask_path = _resolved_file(path)
    try:
        image = sitk.ReadImage(str(mask_path))
    except Exception as exc:
        raise CartilageAnalysisError(f"无法读取 NIfTI 分割文件: {exc}") from exc
    if image.GetDimension() != 3:
        raise CartilageAnalysisError(f"分割必须是三维图像，当前维度为 {image.GetDimension()}。")
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise CartilageAnalysisError("分割必须是单通道标签图。")

    # SimpleITK exposes arrays as [z, y, x].  The validated implementation and
    # OAI-ZIB NIfTI files use index order [x, y, z] == [D, H, W].
    array_zyx = sitk.GetArrayFromImage(image)
    pred = np.transpose(np.asarray(array_zyx), (2, 1, 0))
    spacing = tuple(float(value) for value in image.GetSpacing())
    return mask_path, image, pred, spacing


def _orientation_summary(image: sitk.Image) -> dict[str, Any]:
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    dominant_axes = np.argmax(np.abs(direction), axis=0)
    return {
        "origin": [float(value) for value in image.GetOrigin()],
        "direction": [float(value) for value in image.GetDirection()],
        "dominant_physical_axes": [int(value) for value in dominant_axes],
        "axis_aligned": bool(
            len(set(int(value) for value in dominant_axes)) == 3
            and np.all(np.max(np.abs(direction), axis=0) >= 0.9)
        ),
    }


def _inspect_loaded_mask(
    mask_path: Path,
    image: sitk.Image,
    raw_pred: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[dict[str, Any], np.ndarray | None]:
    errors: list[str] = []
    warnings: list[str] = []

    if not np.all(np.isfinite(raw_pred)):
        errors.append("标签图包含 NaN 或无穷值。")
        pred = None
        unique_values = np.asarray([], dtype=np.float64)
    else:
        rounded = np.rint(raw_pred)
        if not np.allclose(raw_pred, rounded, atol=1e-4, rtol=0.0):
            errors.append("标签图包含非整数标签值。")
            pred = None
        else:
            pred = rounded.astype(np.uint8, copy=False)
        unique_values = np.unique(rounded)

    if len(spacing) != 3 or not all(math.isfinite(v) and v > 0 for v in spacing):
        errors.append(f"体素间距无效: {spacing}")

    unique_labels = {int(value) for value in unique_values.tolist()}
    sorted_labels = sorted(unique_labels)
    unexpected = sorted(unique_labels - SUPPORTED_LABELS)
    if unexpected:
        if len(unique_labels) > 64:
            errors.append(
                f"检测到 {len(unique_labels)} 个不同整数值，范围 "
                f"{sorted_labels[0]}-{sorted_labels[-1]}；该文件更像灰度 MRI，"
                "不是六标签分割。厚度分析需要仅含 0、1、2、3、4、5 的标签 mask。"
            )
        else:
            errors.append(f"存在不支持的标签: {unexpected}；期望标签为 0-5。")

    label_counts: dict[str, int] = {}
    label_volumes: dict[str, float] = {}
    voxel_volume = float(np.prod(spacing)) if len(spacing) == 3 else 0.0
    if pred is not None:
        for label in sorted(SUPPORTED_LABELS):
            count = int(np.count_nonzero(pred == label))
            label_counts[str(label)] = count
            label_volumes[str(label)] = float(count * voxel_volume)

        present_cartilage = [label for label in CARTILAGE_BONE_PAIRS if label_counts[str(label)] > 0]
        if not present_cartilage:
            errors.append("未检测到标签 2、4、5 中的任何软骨结构。")
        for cartilage_label in present_cartilage:
            bone_label = CARTILAGE_BONE_PAIRS[cartilage_label]
            if label_counts[str(bone_label)] == 0:
                errors.append(
                    f"{LABEL_NAMES[cartilage_label]} 存在，但配对骨标签 {bone_label} 缺失。"
                )
            if label_counts[str(cartilage_label)] < 100:
                warnings.append(
                    f"标签 {cartilage_label} 仅有 {label_counts[str(cartilage_label)]} 个体素，"
                    "厚度统计可能不稳定。"
                )
        missing_cartilage = [
            label for label in CARTILAGE_BONE_PAIRS if label_counts[str(label)] == 0
        ]
        if missing_cartilage:
            warnings.append(f"缺少软骨标签: {missing_cartilage}，对应结果将为空。")

    orientation = _orientation_summary(image)
    if not orientation["axis_aligned"]:
        warnings.append(
            "图像方向不是接近轴对齐的 OAI-ZIB 方向；16 亚区解剖命名可能不可靠。"
        )
    elif orientation["dominant_physical_axes"] != [0, 1, 2]:
        errors.append(
            "图像索引轴发生置换，不满足算法所需的 [D,H,W]=[左右,前后,上下] 顺序。"
        )

    report = {
        "status": "success",
        "valid_for_analysis": not errors,
        "mask_path": str(mask_path),
        "shape_dhw": [int(value) for value in raw_pred.shape],
        "spacing_mm": [float(value) for value in spacing],
        "voxel_volume_mm3": voxel_volume,
        "labels_present": (
            sorted_labels
            if len(sorted_labels) <= 64
            else sorted_labels[:32] + sorted_labels[-8:]
        ),
        "label_value_count": len(sorted_labels),
        "labels_present_truncated": len(sorted_labels) > 64,
        "label_names": {str(key): value for key, value in LABEL_NAMES.items()},
        "label_voxel_counts": label_counts,
        "label_volumes_mm3": label_volumes,
        "orientation": orientation,
        "errors": errors,
        "warnings": warnings,
        "method_requirements": {
            "label_schema": "0=background, 1=femur, 2=femoral cartilage, "
                            "3=tibia, 4=medial tibial cartilage, "
                            "5=lateral tibial cartilage",
            "index_order": "pred[D,H,W] = [left-right, anterior-posterior, superior-inferior]",
        },
    }
    return report, pred


def validate_cartilage_segmentation_file(mask_path: str) -> dict[str, Any]:
    """Validate one OAI-ZIB six-label NIfTI segmentation."""
    path, image, raw_pred, spacing = _load_mask(mask_path)
    report, _ = _inspect_loaded_mask(path, image, raw_pred, spacing)
    return report


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _run_numerical_analysis(
    pred: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # The research implementation prints progress summaries.  stdio is the MCP
    # transport, so tool execution must keep those diagnostics off stdout.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        thickness_result = compute_cartilage_thickness(pred, spacing)
        subregion_result = compute_subregion_thickness(
            pred,
            spacing,
            vertex_data=thickness_result.get("vertex_data"),
        )
    return thickness_result, subregion_result


def _case_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_reference_nifti(
    array_dhw: np.ndarray,
    reference: sitk.Image,
    output_path: Path,
) -> None:
    image = sitk.GetImageFromArray(np.transpose(array_dhw, (2, 1, 0)))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(output_path), True)


def _global_rows(statistics: dict[Any, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in (2, 4, 5):
        stats = statistics.get(label, {})
        rows.append(
            {
                "label": label,
                "compartment": COMPARTMENT_CODES[label],
                "name": LABEL_NAMES[label],
                "mean_mm": stats.get("mean"),
                "std_mm": stats.get("std"),
                "median_mm": stats.get("median"),
                "p25_mm": stats.get("p25"),
                "p75_mm": stats.get("p75"),
                "p90_mm": stats.get("p90"),
                "p95_mm": stats.get("p95"),
                "max_mm": stats.get("max"),
                "volume_mm3": stats.get("total_volume_mm3"),
                "n_vertices": stats.get("n_vertices", 0),
                "n_outliers": stats.get("n_outliers", 0),
            }
        )
    return rows


def _subregion_rows(subregion_result: dict[str, Any]) -> list[dict[str, Any]]:
    stats_by_name = subregion_result.get("subregions", {})
    counts = subregion_result.get("vertex_counts", {})
    rows: list[dict[str, Any]] = []
    for name in ALL_SUBREGIONS:
        stats = stats_by_name.get(name, {})
        rows.append(
            {
                "subregion": name,
                "mean_mm": stats.get("mean"),
                "std_mm": stats.get("std"),
                "median_mm": stats.get("median"),
                "p25_mm": stats.get("p25"),
                "p75_mm": stats.get("p75"),
                "p90_mm": stats.get("p90"),
                "p95_mm": stats.get("p95"),
                "n_vertices": int(counts.get(name, 0)),
            }
        )
    return rows


def analyze_cartilage_thickness_file(
    mask_path: str,
    output_dir: str,
    *,
    export_thickness_maps: bool = True,
    export_subregion_mask: bool = False,
) -> dict[str, Any]:
    """Run global 3D-MN and Wirth/Eckstein 16-subregion analysis."""
    path, image, raw_pred, spacing = _load_mask(mask_path)
    qc, pred = _inspect_loaded_mask(path, image, raw_pred, spacing)
    if pred is None or not qc["valid_for_analysis"]:
        return {
            "status": "failed",
            "message": "分割未通过软骨厚度分析前置校验。",
            "qc": qc,
            "files": [],
        }

    destination = _prepare_output_dir(output_dir)
    thickness_result, subregion_result = _run_numerical_analysis(pred, spacing)
    global_statistics = _json_safe(thickness_result.get("statistics", {}))
    subregion_statistics = _json_safe(subregion_result.get("subregions", {}))
    vertex_counts = _json_safe(subregion_result.get("vertex_counts", {}))
    plate_means = _json_safe(subregion_result.get("plate_means", {}))

    global_csv = destination / "cartilage_thickness_summary.csv"
    subregion_csv = destination / "cartilage_subregions16_summary.csv"
    analysis_json = destination / "cartilage_thickness_analysis.json"
    _write_csv(
        global_csv,
        _global_rows(thickness_result.get("statistics", {})),
        [
            "label", "compartment", "name", "mean_mm", "std_mm",
            "median_mm", "p25_mm", "p75_mm", "p90_mm", "p95_mm",
            "max_mm", "volume_mm3", "n_vertices", "n_outliers",
        ],
    )
    _write_csv(
        subregion_csv,
        _subregion_rows(subregion_result),
        [
            "subregion", "mean_mm", "std_mm", "median_mm", "p25_mm",
            "p75_mm", "p90_mm", "p95_mm", "n_vertices",
        ],
    )

    files = [str(global_csv), str(subregion_csv), str(analysis_json)]
    map_files: dict[str, str] = {}
    if export_thickness_maps:
        stem = _case_stem(path)
        for label, code in COMPARTMENT_CODES.items():
            thickness_map = thickness_result.get("thickness_maps", {}).get(label)
            if thickness_map is None or not np.any(thickness_map > 0):
                continue
            map_path = destination / f"{stem}_{code}_thickness_mm.nii.gz"
            _write_reference_nifti(
                np.asarray(thickness_map, dtype=np.float32), image, map_path
            )
            map_files[code] = str(map_path)
            files.append(str(map_path))

    subregion_mask_path: str | None = None
    if export_subregion_mask:
        encoded = np.zeros(pred.shape, dtype=np.uint8)
        for index, name in enumerate(ALL_SUBREGIONS, start=1):
            mask = subregion_result.get("masks", {}).get(name)
            if mask is not None:
                encoded[np.asarray(mask, dtype=bool)] = index
        encoded_path = destination / f"{_case_stem(path)}_subregions16.nii.gz"
        _write_reference_nifti(encoded, image, encoded_path)
        subregion_mask_path = str(encoded_path)
        files.append(str(encoded_path))

    payload = {
        "status": "success",
        "message": "已完成 3D-MN 全局软骨厚度和 16 亚区分析。",
        "method": "3D-MN bidirectional surface normals with area weighting",
        "mask_path": str(path),
        "output_dir": str(destination),
        "qc": qc,
        "global_statistics": global_statistics,
        "subregions16": subregion_statistics,
        "plate_means_mm": plate_means,
        "vertex_counts": vertex_counts,
        "files": files,
        "global_csv": str(global_csv),
        "subregion_csv": str(subregion_csv),
        "analysis_json": str(analysis_json),
        "thickness_maps": map_files,
        "subregion_mask": subregion_mask_path,
        "provenance": {
            "algorithm_source": "Cartilage thickness estimation/utils",
            "algorithm_snapshot_date": "2026-08-25",
            "spacing_mm": [float(value) for value in spacing],
            "label_schema": "OAI-ZIB six-class",
        },
    }
    with analysis_json.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2)
    return payload


def _parse_mesh_label_ids(label_ids: Any) -> list[int]:
    if isinstance(label_ids, str):
        normalized = label_ids
        for separator in ("，", ";", "；", "|", "/"):
            normalized = normalized.replace(separator, ",")
        raw_values = [value.strip() for value in normalized.split(",") if value.strip()]
    elif isinstance(label_ids, Iterable):
        raw_values = list(label_ids)
    else:
        raw_values = [label_ids]
    try:
        labels = list(dict.fromkeys(int(value) for value in raw_values))
    except (TypeError, ValueError) as exc:
        raise CartilageAnalysisError(
            "label_ids 必须是以逗号分隔的整数，例如 1,2,3,4,5。"
        ) from exc
    if not labels:
        raise CartilageAnalysisError("至少需要指定一个网格标签。")
    unsupported = [label for label in labels if label not in MESH_FILE_CODES]
    if unsupported:
        raise CartilageAnalysisError(
            f"网格标签仅支持 1-5（不包含背景），当前不支持: {unsupported}"
        )
    return labels


def _parse_mesh_formats(output_formats: Any) -> list[str]:
    if isinstance(output_formats, str):
        normalized = output_formats.lower()
        for separator in ("，", ";", "；", "|", "/"):
            normalized = normalized.replace(separator, ",")
        values = [value.strip().lstrip(".") for value in normalized.split(",")]
    elif isinstance(output_formats, Iterable):
        values = [str(value).strip().lower().lstrip(".") for value in output_formats]
    else:
        values = [str(output_formats).strip().lower().lstrip(".")]
    formats = list(dict.fromkeys(value for value in values if value))
    supported = {"ply", "stl", "npz"}
    unsupported = [value for value in formats if value not in supported]
    if not formats or unsupported:
        raise CartilageAnalysisError(
            "output_formats 仅支持 ply、stl、npz，可使用逗号组合，例如 ply,npz。"
        )
    return formats


def _vertices_to_lps(vertices_local_mm: np.ndarray, image: sitk.Image) -> np.ndarray:
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    return (
        vertices_local_mm.astype(np.float64) @ direction.T + origin
    ).astype(np.float32)


def _mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces].astype(np.float64)
    cross_products = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return float(0.5 * np.linalg.norm(cross_products, axis=1).sum())


def _write_binary_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    label_id: int,
) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment generated_by cartilage_analysis MCP",
            "comment coordinate_system ITK_LPS_mm",
            f"comment label_id {label_id}",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            f"element face {len(faces)}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    ).encode("ascii")
    vertex_records = np.empty(
        len(vertices),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex_records["x"] = vertices[:, 0]
    vertex_records["y"] = vertices[:, 1]
    vertex_records["z"] = vertices[:, 2]
    vertex_records["red"], vertex_records["green"], vertex_records["blue"] = color
    face_records = np.empty(
        len(faces),
        dtype=[("vertex_count", "u1"), ("vertex_indices", "<i4", (3,))],
    )
    face_records["vertex_count"] = 3
    face_records["vertex_indices"] = faces
    with path.open("wb") as handle:
        handle.write(header)
        vertex_records.tofile(handle)
        face_records.tofile(handle)


def _write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    triangles = vertices[faces].astype(np.float32)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid, None]
    normals[~valid] = 0
    records = np.empty(
        len(faces),
        dtype=[
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ],
    )
    records["normal"] = normals
    records["vertices"] = triangles
    records["attribute"] = 0
    header = b"cartilage_analysis ITK_LPS_mm".ljust(80, b"\0")
    with path.open("wb") as handle:
        handle.write(header)
        np.asarray([len(faces)], dtype="<u4").tofile(handle)
        records.tofile(handle)


def generate_cartilage_3d_mesh_file(
    mask_path: str,
    output_dir: str,
    *,
    label_ids: Any = "1,2,3,4,5",
    smooth: bool = True,
    output_formats: Any = "ply,npz",
) -> dict[str, Any]:
    """Generate per-label 3D meshes in physical ITK LPS millimetres."""
    labels = _parse_mesh_label_ids(label_ids)
    formats = _parse_mesh_formats(output_formats)
    path, image, raw_pred, spacing = _load_mask(mask_path)
    qc, pred = _inspect_loaded_mask(path, image, raw_pred, spacing)
    if pred is None or not qc["valid_for_analysis"]:
        return {
            "status": "failed",
            "message": "分割未通过三维网格生成前置校验。",
            "qc": qc,
            "files": [],
        }

    prepared_meshes: list[dict[str, Any]] = []
    skipped_labels: dict[str, str] = {}
    for label_id in labels:
        voxel_count = int(np.count_nonzero(pred == label_id))
        if voxel_count < 100:
            skipped_labels[str(label_id)] = (
                f"{LABEL_NAMES[label_id]} 仅有 {voxel_count} 个体素，无法稳定生成表面。"
            )
            continue
        try:
            vertices_local, faces = build_mesh(
                pred == label_id,
                label_id,
                pred,
                spacing,
                smooth=bool(smooth),
            )
        except Exception as exc:
            skipped_labels[str(label_id)] = f"网格算法执行失败: {exc}"
            continue
        if vertices_local is None or faces is None:
            skipped_labels[str(label_id)] = "未提取到有效三角表面。"
            continue
        vertices_lps = _vertices_to_lps(vertices_local, image)
        prepared_meshes.append(
            {
                "label_id": label_id,
                "label_name": LABEL_NAMES[label_id],
                "file_code": MESH_FILE_CODES[label_id],
                "color_rgb": MESH_COLORS_RGB[label_id],
                "voxel_count": voxel_count,
                "vertices_local_mm": vertices_local,
                "vertices_lps_mm": vertices_lps,
                "faces": faces,
            }
        )

    if not prepared_meshes:
        return {
            "status": "failed",
            "message": "请求的标签均未能生成有效三维网格。",
            "qc": qc,
            "requested_labels": labels,
            "skipped_labels": skipped_labels,
            "files": [],
        }

    destination = _prepare_output_dir(output_dir)
    stem = _case_stem(path)
    exported_files: list[str] = []
    mesh_records: dict[str, dict[str, Any]] = {}
    for mesh in prepared_meshes:
        label_id = int(mesh["label_id"])
        base_name = f"{stem}_label{label_id}_{mesh['file_code']}"
        format_paths: dict[str, str] = {}
        for output_format in formats:
            output_path = destination / f"{base_name}.{output_format}"
            if output_format == "ply":
                _write_binary_ply(
                    output_path,
                    mesh["vertices_lps_mm"],
                    mesh["faces"],
                    mesh["color_rgb"],
                    label_id,
                )
            elif output_format == "stl":
                _write_binary_stl(
                    output_path,
                    mesh["vertices_lps_mm"],
                    mesh["faces"],
                )
            else:
                np.savez_compressed(
                    output_path,
                    vertices=mesh["vertices_lps_mm"],
                    vertices_lps_mm=mesh["vertices_lps_mm"],
                    vertices_local_mm=mesh["vertices_local_mm"],
                    faces=mesh["faces"],
                    label_id=np.asarray(label_id, dtype=np.int16),
                    label_name=np.asarray(mesh["label_name"]),
                    color_rgb=np.asarray(mesh["color_rgb"], dtype=np.uint8),
                    spacing_mm=np.asarray(image.GetSpacing(), dtype=np.float64),
                    origin_lps_mm=np.asarray(image.GetOrigin(), dtype=np.float64),
                    direction=np.asarray(image.GetDirection(), dtype=np.float64),
                    coordinate_system=np.asarray("ITK_LPS_mm"),
                    source_mask=np.asarray(str(path)),
                )
            format_paths[output_format] = str(output_path)
            exported_files.append(str(output_path))

        vertices_lps = mesh["vertices_lps_mm"]
        mesh_records[str(label_id)] = {
            "label_id": label_id,
            "label_name": mesh["label_name"],
            "color_rgb": list(mesh["color_rgb"]),
            "voxel_count": int(mesh["voxel_count"]),
            "vertex_count": int(len(vertices_lps)),
            "triangle_count": int(len(mesh["faces"])),
            "surface_area_mm2": _mesh_surface_area(vertices_lps, mesh["faces"]),
            "bounds_lps_mm": {
                "min": [float(value) for value in vertices_lps.min(axis=0)],
                "max": [float(value) for value in vertices_lps.max(axis=0)],
            },
            "files": format_paths,
        }

    manifest_path = destination / "cartilage_3d_mesh_manifest.json"
    all_files = [*exported_files, str(manifest_path)]
    status = "partial_success" if skipped_labels else "success"
    payload = {
        "status": status,
        "message": (
            f"已为 {len(mesh_records)} 个解剖标签生成三维表面网格。"
            + (f"另有 {len(skipped_labels)} 个标签被跳过。" if skipped_labels else "")
        ),
        "mask_path": str(path),
        "output_dir": str(destination),
        "requested_labels": labels,
        "generated_labels": [int(value) for value in mesh_records],
        "skipped_labels": skipped_labels,
        "smooth": bool(smooth),
        "output_formats": formats,
        "coordinate_system": "ITK_LPS_mm",
        "geometry": {
            "spacing_mm": [float(value) for value in image.GetSpacing()],
            "origin_lps_mm": [float(value) for value in image.GetOrigin()],
            "direction": [float(value) for value in image.GetDirection()],
            "size_xyz": [int(value) for value in image.GetSize()],
        },
        "meshes": mesh_records,
        "manifest_path": str(manifest_path),
        "files": all_files,
        "qc": qc,
        "provenance": {
            "algorithm_source": "Cartilage thickness estimation/utils/mesh.py",
            "algorithm_snapshot_date": "2026-08-29",
            "surface_method": "marching cubes with optional Gaussian/Laplacian smoothing",
            "label_schema": "OAI-ZIB six-class",
        },
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2)
    return payload


def _normalize_source_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "caseid": "CaseID",
        "case_id": "CaseID",
        "maskpath": "MaskPath",
        "mask_path": "MaskPath",
        "klgrade": "KLGrade",
        "kl_grade": "KLGrade",
        "koa_grade": "KLGrade",
        "sourcecaseid": "SourceCaseID",
        "source_case_id": "SourceCaseID",
    }
    renamed: dict[str, str] = {}
    for column in frame.columns:
        clean = str(column).strip().lstrip("\ufeff")
        renamed[column] = aliases.get(clean.lower(), clean)
    return frame.rename(columns=renamed)


def _feature_row(
    source_row: pd.Series,
    thickness_result: dict[str, Any],
    subregion_result: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"CaseID": str(source_row["CaseID"])}
    for metadata in (
        "SourceCaseID", "KLGrade", "Age", "Gender", "BMI", "KneeSide"
    ):
        if metadata in source_row.index:
            row[metadata] = source_row[metadata]

    statistics = thickness_result.get("statistics", {})
    for label, code in COMPARTMENT_CODES.items():
        stats = statistics.get(label, {})
        for feature in GLOBAL_FEATURES:
            row[f"MN3D_{code}_{feature}"] = stats.get(feature, 0.0)

    sub_stats = subregion_result.get("subregions", {})
    counts = subregion_result.get("vertex_counts", {})
    for name in ALL_SUBREGIONS:
        stats = sub_stats.get(name, {})
        for feature in SUBREGION_FEATURES:
            row[f"MN3D_{name}_{feature}"] = stats.get(feature, 0.0)
        row[f"MN3D_{name}_n_vertices"] = int(counts.get(name, 0))

    for plate in ("MT", "LT", "MF", "LF"):
        row[f"MN3D_{plate}_area_weighted_mean"] = (
            subregion_result.get("plate_means", {}).get(plate, 0.0)
        )
    return _json_safe(row)


def extract_cartilage_thickness_features_file(
    csv_path: str,
    output_dir: str,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Extract patient-level global and 16-subregion thickness features."""
    source = Path(str(csv_path)).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() != ".csv":
        raise CartilageAnalysisError(f"CSV 文件不存在或格式不正确: {source}")
    destination = _prepare_output_dir(output_dir)

    try:
        # Read source metadata as text so identifiers such as ``001`` are not
        # irreversibly coerced to integer ``1`` before feature generation.
        frame = _normalize_source_columns(pd.read_csv(source, dtype=str))
    except Exception as exc:
        raise CartilageAnalysisError(f"无法读取 CSV: {exc}") from exc
    missing = [name for name in ("CaseID", "MaskPath") if name not in frame.columns]
    if missing:
        raise CartilageAnalysisError(f"CSV 缺少必需列: {missing}")

    total_cases = int(len(frame))

    def report_progress(
        processed_cases: int,
        case_id: str | None = None,
        case_status: str | None = None,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "processed_cases": int(processed_cases),
                    "total_cases": total_cases,
                    "successful_cases": int(len(rows)),
                    "failed_cases": int(len(failures)),
                    "case_id": case_id,
                    "case_status": case_status,
                }
            )
        except Exception:
            # Progress reporting is observational and must never invalidate a
            # numerically successful extraction run.
            pass

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    report_progress(0)
    for processed_cases, (_, source_row) in enumerate(frame.iterrows(), start=1):
        case_id = str(source_row.get("CaseID", ""))
        mask_value = source_row.get("MaskPath")
        if pd.isna(mask_value) or not str(mask_value).strip():
            failures.append({"CaseID": case_id, "error": "MaskPath 为空"})
            report_progress(processed_cases, case_id, "failed")
            continue
        try:
            mask_path, image, raw_pred, spacing = _load_mask(str(mask_value))
            qc, pred = _inspect_loaded_mask(mask_path, image, raw_pred, spacing)
            if pred is None or not qc["valid_for_analysis"]:
                raise CartilageAnalysisError("; ".join(qc["errors"]))
            thickness_result, subregion_result = _run_numerical_analysis(pred, spacing)
            rows.append(_feature_row(source_row, thickness_result, subregion_result))
            case_status = "success"
        except Exception as exc:
            failures.append({"CaseID": case_id, "error": str(exc)})
            case_status = "failed"
        report_progress(processed_cases, case_id, case_status)

    if not rows:
        return {
            "status": "failed",
            "message": "没有病例成功生成软骨厚度特征。",
            "source_csv": str(source),
            "processed_cases": 0,
            "failed_cases": len(failures),
            "failures": failures,
            "files": [],
        }

    features = pd.DataFrame(rows)
    output_csv = destination / "cartilage_thickness_features_summary.csv"
    features.to_csv(output_csv, index=False, encoding="utf-8-sig")
    feature_columns = [
        column for column in features.columns
        if column not in {
            "CaseID", "SourceCaseID", "KLGrade", "Age", "Gender", "BMI", "KneeSide"
        }
    ]
    manifest_path = destination / "cartilage_thickness_features_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "method": "3D-MN plus Wirth/Eckstein 16 subregions",
        "source_csv": str(source),
        "output_csv": str(output_csv),
        "input_rows": int(len(frame)),
        "processed_cases": int(len(rows)),
        "failed_cases": int(len(failures)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "failures": failures,
        "label_schema": "OAI-ZIB six-class",
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    status = "success" if not failures else "partial_success"
    return {
        "status": status,
        "message": (
            f"已为 {len(rows)} 个病例生成 {len(feature_columns)} 个 3D-MN/16亚区厚度特征。"
        ),
        "source_csv": str(source),
        "output_dir": str(destination),
        "output_csv": str(output_csv),
        "manifest_path": str(manifest_path),
        "processed_cases": int(len(rows)),
        "failed_cases": int(len(failures)),
        "feature_count": int(len(feature_columns)),
        "failures": failures,
        "files": [str(output_csv), str(manifest_path)],
    }
