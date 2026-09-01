"""Dependency-light safety normalization for feature-extraction plans."""

from __future__ import annotations

import re
from typing import Any
from datetime import datetime

from .path_utils import (
    infer_dataset_role,
    parse_tool_result_paths,
    select_train_test_source_paths,
)
from .task_contracts import infer_cartilage_operation


_LDSE_SINGLE = "ldse_features_extraction"
_LDSE_PAIRED = "ldse_train_test_features_extraction"
_CARTILAGE_TOOLS = {
    "validate_cartilage_segmentation",
    "analyze_cartilage_thickness",
    "extract_cartilage_thickness_features",
    "generate_cartilage_3d_mesh",
}


def _mesh_label_ids_from_text(text: str) -> str:
    raw = str(text or "")
    lower = raw.lower()
    if any(
        term in lower
        for term in (
            "全部标签", "所有标签", "全标签", "完整膝关节", "骨和软骨",
            "骨与软骨", "all labels", "whole knee",
        )
    ):
        return "1,2,3,4,5"

    labels: list[int] = []
    if any(term in lower for term in ("股骨软骨", "femoral cartilage")):
        labels.append(2)
    if any(
        term in lower
        for term in ("内侧胫骨软骨", "medial tibial cartilage")
    ):
        labels.append(4)
    if any(
        term in lower
        for term in ("外侧胫骨软骨", "lateral tibial cartilage")
    ):
        labels.append(5)
    if "股骨" in lower and not any(
        name in lower for name in ("股骨软骨", "femoral cartilage")
    ):
        labels.append(1)
    if "胫骨" in lower and not any(
        name in lower
        for name in (
            "内侧胫骨软骨", "外侧胫骨软骨",
            "medial tibial cartilage", "lateral tibial cartilage",
        )
    ):
        labels.append(3)
    if any(
        term in lower
        for term in ("三类软骨", "所有软骨", "全部软骨", "all cartilage")
    ):
        labels.extend((2, 4, 5))
    if labels:
        return ",".join(str(label) for label in dict.fromkeys(labels))
    if any(
        term in lower
        for term in (
            "仅软骨", "只生成软骨", "三类软骨", "软骨三维网格",
            "软骨的三维网格", "cartilage mesh",
        )
    ):
        return "2,4,5"
    return "1,2,3,4,5"


def _mesh_output_formats_from_text(text: str) -> str:
    lower = str(text or "").lower()
    requested = [name for name in ("ply", "stl") if name in lower]
    if not requested and "npz" in lower:
        return "npz"
    if not requested:
        requested = ["ply"]
    if "npz" not in requested:
        requested.append("npz")
    return ",".join(requested)


def enforce_cartilage_analysis_plan(
    decision: dict,
    current_input: str,
    available_csv_paths: list[str],
    available_nifti_paths: list[str],
    timestamp: str | None = None,
) -> None:
    """Build a deterministic P0 cartilage-tool plan from an explicit request."""
    raw = str(current_input or "")
    lower = raw.lower()
    operation = infer_cartilage_operation(raw)
    cartilage_request = operation is not None
    if not cartilage_request:
        return

    tools = decision.get("tools")
    if not isinstance(tools, list):
        tools = []
        decision["tools"] = tools
    feature_tools = {
        "meas_features_extraction",
        "ldse_features_extraction",
        "ldse_train_test_features_extraction",
        "extract_radiomics_features",
        *_CARTILAGE_TOOLS,
    }
    tools[:] = [
        call for call in tools
        if not (isinstance(call, dict) and call.get("tool") in feature_tools)
    ]

    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_request = bool(available_csv_paths and operation == "batch_features")
    validation_request = operation == "validation"

    if batch_request:
        for index, csv_path in enumerate(dict.fromkeys(available_csv_paths), start=1):
            role = infer_dataset_role(csv_path)
            suffix = role.capitalize() if role else f"Batch{index}"
            tools.append(
                {
                    "tool": "extract_cartilage_thickness_features",
                    "arguments": {
                        "csv_path": csv_path,
                        "output_dir": f"CartilageThickness_Features_{suffix}_{stamp}",
                    },
                    "message": f"正在为{suffix}数据提取 3D-MN 和 16 亚区软骨厚度特征。",
                }
            )
        return

    if not available_nifti_paths:
        _append_final_message(
            decision,
            "请提供真实的 OAI-ZIB 六分类分割 .nii 或 .nii.gz 文件路径。",
        )
        return

    for index, mask_path in enumerate(dict.fromkeys(available_nifti_paths), start=1):
        if validation_request:
            tools.append(
                {
                    "tool": "validate_cartilage_segmentation",
                    "arguments": {"mask_path": mask_path},
                    "message": "正在校验分割标签和空间几何是否适合软骨厚度分析。",
                }
            )
            continue
        if operation == "mesh":
            tools.append(
                {
                    "tool": "generate_cartilage_3d_mesh",
                    "arguments": {
                        "mask_path": mask_path,
                        "output_dir": f"Cartilage3D_Mesh_{stamp}_{index}",
                        "label_ids": _mesh_label_ids_from_text(raw),
                        "smooth": not any(
                            term in lower
                            for term in ("不平滑", "不要平滑", "原始体素", "voxel surface")
                        ),
                        "output_formats": _mesh_output_formats_from_text(raw),
                    },
                    "message": "正在从六标签分割重建物理空间三维三角网格。",
                }
            )
            continue
        tools.append(
            {
                "tool": "analyze_cartilage_thickness",
                "arguments": {
                    "mask_path": mask_path,
                    "output_dir": f"CartilageThickness_Analysis_{stamp}_{index}",
                    "export_thickness_maps": "不导出厚度图" not in raw,
                    "export_subregion_mask": any(
                        term in lower
                        for term in ("亚区掩膜", "亚区mask", "subregion mask")
                    ),
                },
                "message": "正在执行 3D-MN 全局厚度和 Wirth/Eckstein 16 亚区联合分析。",
            }
        )


def enforce_requested_feature_extraction_plan(
    decision: dict,
    available_csv_paths: list[str],
    requested_feature_type: str | None,
    timestamp: str | None = None,
) -> None:
    """Deny mismatched feature tools and create a missing explicit plan.

    This is a control-plane repair for explicit MEAS/LDSE/Radiomics requests;
    it does not depend on conversational history choosing the right tool.
    """
    if requested_feature_type not in {"MEAS", "LDSE", "Radiomics"}:
        return
    tools = decision.get("tools")
    if not isinstance(tools, list):
        tools = []
        decision["tools"] = tools

    feature_tool_types = {
        "meas_features_extraction": "MEAS",
        "ldse_features_extraction": "LDSE",
        "ldse_train_test_features_extraction": "LDSE",
        "extract_radiomics_features": "Radiomics",
    }
    tools[:] = [
        call
        for call in tools
        if not (
            isinstance(call, dict)
            and call.get("tool") in feature_tool_types
            and feature_tool_types[call.get("tool")] != requested_feature_type
        )
    ]
    if any(
        isinstance(call, dict)
        and feature_tool_types.get(call.get("tool")) == requested_feature_type
        for call in tools
    ):
        return

    train_source, test_source = select_train_test_source_paths(available_csv_paths)
    if not train_source or not test_source:
        _append_final_message(
            decision,
            f"当前任务缺少可识别的 train/test 源 CSV，未执行 {requested_feature_type} 特征提取。",
        )
        return
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if requested_feature_type == "LDSE":
        tools.append(
            {
                "tool": _LDSE_PAIRED,
                "arguments": {
                    "output_dir": f"LDSE_Features_TrainTest_{stamp}",
                    "train_csv_path": train_source,
                    "test_csv_path": test_source,
                    "n_components": 50,
                    "random_seed": 42,
                },
                "message": "为当前任务联合提取训练集和测试集 LDSE 特征。",
            }
        )
        return

    tool_name = (
        "meas_features_extraction"
        if requested_feature_type == "MEAS"
        else "extract_radiomics_features"
    )
    prefix = "MEAS" if requested_feature_type == "MEAS" else "Radiomics"
    for role, source in (("Train", train_source), ("Test", test_source)):
        tools.append(
            {
                "tool": tool_name,
                "arguments": {
                    "output_dir": f"{prefix}_Features_{role}_{stamp}",
                    "csv_path": source,
                },
                "message": f"为当前任务提取 {role} {requested_feature_type} 特征。",
            }
        )


def ensure_safe_feature_extraction_plan(
    decision: dict,
    available_csv_paths: list[str],
) -> None:
    """Normalize LDSE plans so train/test extraction cannot fit test data."""
    tools = decision.get("tools")
    if not isinstance(tools, list):
        return

    train_source, test_source = select_train_test_source_paths(available_csv_paths)
    _ensure_online_ldse_sequence(tools, train_source, test_source)
    _ensure_meas_pair(tools, train_source, test_source)

    ldse_calls = [
        call for call in tools
        if isinstance(call, dict) and call.get("tool") in {_LDSE_SINGLE, _LDSE_PAIRED}
    ]
    if not ldse_calls:
        return

    paired_calls = [call for call in ldse_calls if call.get("tool") == _LDSE_PAIRED]
    single_calls = [call for call in ldse_calls if call.get("tool") == _LDSE_SINGLE]

    if train_source and test_source:
        _replace_ldse_calls_with_paired(tools, ldse_calls, train_source, test_source)
        return

    if paired_calls:
        tools[:] = [call for call in tools if call not in paired_calls]
        _append_final_message(decision, "请提供可识别的训练集和测试集 CSV 路径。")

    if not single_calls:
        return

    if train_source:
        _retain_train_single(tools, single_calls, train_source)
    elif test_source:
        _retain_safe_test_single(tools, single_calls, test_source, decision)
    else:
        _remove_calls(tools, single_calls)
        _append_final_message(decision, "请提供可识别训练/测试角色的 CSV 数据或模型输入。")


def _ensure_online_ldse_sequence(
    tools: list[Any],
    train_source: str | None,
    test_source: str | None,
) -> None:
    """Normalize existing online-LDSE calls without inventing augmentation."""
    online_calls = [
        call
        for call in tools
        if isinstance(call, dict)
        and call.get("tool") == "augment_online_extract_features"
        and str(_arguments(call).get("feature_type", "")).upper() == "LDSE"
    ]
    if not online_calls:
        return

    by_role: dict[str, list[dict]] = {"train": [], "test": [], "unknown": []}
    for call in online_calls:
        arguments = _arguments(call)
        csv_path = str(arguments.get("csv_path", ""))
        explicit_role = str(arguments.get("dataset_role", "")).strip().lower()
        if train_source and csv_path == train_source:
            role = "train"
        elif test_source and csv_path == test_source:
            role = "test"
        elif explicit_role in {"train", "test"}:
            role = explicit_role
        else:
            role = infer_dataset_role(csv_path, str(arguments.get("output_dir", "")))
        if role in {"train", "test"}:
            arguments["dataset_role"] = role
            if role == "train":
                arguments.pop("ldse_model_path", None)
            call["arguments"] = arguments
            by_role[role].append(call)
        else:
            by_role["unknown"].append(call)

    ordered_iterator = iter(by_role["train"] + by_role["test"] + by_role["unknown"])
    for index, call in enumerate(tools):
        if call in online_calls:
            tools[index] = next(ordered_iterator)


def inject_online_ldse_model_path(test_call: dict, train_result: Any) -> bool:
    """Inject the exact returned train model into one pending online test call."""
    if test_call.get("tool") != "augment_online_extract_features":
        return False
    arguments = _arguments(test_call)
    if (
        str(arguments.get("feature_type", "")).upper() != "LDSE"
        or arguments.get("dataset_role") != "test"
    ):
        return False
    if str(arguments.get("ldse_model_path", "")).strip():
        return False

    model_path = next(
        (
            path
            for key, path in parse_tool_result_paths(train_result)
            if key == "model_path"
        ),
        None,
    )
    if not model_path:
        return False
    arguments["ldse_model_path"] = model_path
    test_call["arguments"] = arguments
    return True


def _replace_ldse_calls_with_paired(
    tools: list[Any],
    ldse_calls: list[dict],
    train_source: str,
    test_source: str,
) -> None:
    first_call = ldse_calls[0]
    first_arguments = _arguments(first_call)
    paired_call = {
        "tool": _LDSE_PAIRED,
        "arguments": {
            "output_dir": _paired_output_dir(str(first_arguments.get("output_dir", ""))),
            "train_csv_path": train_source,
            "test_csv_path": test_source,
            "n_components": first_arguments.get("n_components", 50),
            "random_seed": first_arguments.get("random_seed", 42),
        },
        "message": first_call.get("message", "正在联合提取训练集和测试集 LDSE 特征。"),
    }

    replacement: list[Any] = []
    inserted = False
    for call in tools:
        if call in ldse_calls:
            if not inserted:
                replacement.append(paired_call)
                inserted = True
            continue
        replacement.append(call)
    tools[:] = replacement


def _retain_train_single(tools: list[Any], single_calls: list[dict], train_source: str) -> None:
    call = single_calls[0]
    arguments = _arguments(call)
    arguments["csv_path"] = train_source
    arguments["mode"] = "fit_transform"
    arguments["dataset_role"] = "train"
    arguments.setdefault("n_components", 50)
    arguments.setdefault("random_seed", 42)
    call["arguments"] = arguments
    _remove_calls(tools, single_calls[1:])


def _retain_safe_test_single(
    tools: list[Any],
    single_calls: list[dict],
    test_source: str,
    decision: dict,
) -> None:
    call = next(
        (
            candidate for candidate in single_calls
            if _arguments(candidate).get("mode") == "transform"
            and bool(str(_arguments(candidate).get("model_path", "")).strip())
        ),
        None,
    )
    if call is None:
        _remove_calls(tools, single_calls)
        _append_final_message(decision, "请提供已验证训练得到的 model_path。")
        return

    arguments = _arguments(call)
    arguments["csv_path"] = test_source
    arguments["dataset_role"] = "test"
    arguments.setdefault("n_components", 50)
    arguments.setdefault("random_seed", 42)
    call["arguments"] = arguments
    _remove_calls(tools, [candidate for candidate in single_calls if candidate is not call])


def _ensure_meas_pair(tools: list[Any], train_source: str | None, test_source: str | None) -> None:
    """Guarantee one train and one test MEAS call with isolated output dirs."""
    if not train_source or not test_source:
        return

    meas_calls = [
        call for call in tools
        if isinstance(call, dict) and call.get("tool") == "meas_features_extraction"
    ]
    if not meas_calls:
        return

    roles = [
        (call, infer_dataset_role(
            str(_arguments(call).get("csv_path", "")),
            str(_arguments(call).get("output_dir", "")),
        ))
        for call in meas_calls
    ]
    # Repair arguments even when the LLM already emitted two calls. In
    # particular, never let train/test write the same meas_features_summary.csv.
    for call, role in roles:
        if role not in {"train", "test"}:
            continue
        arguments = _arguments(call)
        arguments["csv_path"] = train_source if role == "train" else test_source
        arguments["output_dir"] = _role_output_dir(
            str(arguments.get("output_dir", "MEAS_Features")), role
        )
        call["arguments"] = arguments

    present_roles = {role for _, role in roles if role in {"train", "test"}}
    if present_roles == {"train", "test"}:
        return

    missing_role = "test" if "train" in present_roles else "train"
    source_role = "train" if missing_role == "test" else "test"
    source_call = next((call for call, role in roles if role == source_role), None)
    if source_call is None:
        return

    source_arguments = _arguments(source_call)
    cloned_call = {
        **source_call,
        "arguments": dict(source_arguments),
        "message": f"正在提取{('测试集' if missing_role == 'test' else '训练集')} MEAS 特征。",
    }
    cloned_call["arguments"]["csv_path"] = test_source if missing_role == "test" else train_source
    cloned_call["arguments"]["output_dir"] = _role_output_dir(
        str(source_arguments.get("output_dir", "")), missing_role
    )
    tools.insert(tools.index(source_call) + 1, cloned_call)


def _arguments(call: dict) -> dict:
    arguments = call.get("arguments")
    return dict(arguments) if isinstance(arguments, dict) else {}


def _remove_calls(tools: list[Any], calls: list[dict]) -> None:
    tools[:] = [call for call in tools if call not in calls]


def _append_final_message(decision: dict, request: str) -> None:
    existing = decision.get("final_message", "")
    existing = existing if isinstance(existing, str) else str(existing)
    if request in existing:
        return
    decision["final_message"] = f"{existing}\n{request}" if existing else request


def _paired_output_dir(output_dir: str) -> str:
    if not output_dir:
        return "LDSE_Features_TrainTest"
    parent, directory_name = _split_output_dir(output_dir)
    for pattern, replacement in (
        (r"(?<![A-Za-z0-9])Train(?![A-Za-z0-9])", "TrainTest"),
        (r"(?<![A-Za-z0-9])Test(?![A-Za-z0-9])", "TrainTest"),
        (r"(?<![A-Za-z0-9])train(?![A-Za-z0-9])", "train_test"),
        (r"(?<![A-Za-z0-9])test(?![A-Za-z0-9])", "train_test"),
    ):
        if re.search(pattern, directory_name):
            return parent + re.sub(pattern, replacement, directory_name, count=1)
    return f"{parent}{directory_name}_TrainTest"


def _role_output_dir(output_dir: str, role: str) -> str:
    parent, directory_name = _split_output_dir(output_dir)
    role_pattern = (
        r"(?<![A-Za-z0-9])(?:Train|train)(?![A-Za-z0-9])"
        if role == "train"
        else r"(?<![A-Za-z0-9])(?:Test|test)(?![A-Za-z0-9])"
    )
    if re.search(role_pattern, directory_name):
        return output_dir
    replacements = (
        ((r"(?<![A-Za-z0-9])Train(?![A-Za-z0-9])", "Test"),
         (r"(?<![A-Za-z0-9])train(?![A-Za-z0-9])", "test"))
        if role == "test" else
        ((r"(?<![A-Za-z0-9])Test(?![A-Za-z0-9])", "Train"),
         (r"(?<![A-Za-z0-9])test(?![A-Za-z0-9])", "train"))
    )
    for pattern, replacement in replacements:
        if re.search(pattern, directory_name):
            return parent + re.sub(pattern, replacement, directory_name, count=1)
    suffix = "Test" if role == "test" else "Train"
    return f"{parent}{directory_name}_{suffix}" if directory_name else f"Features_{suffix}"


def _split_output_dir(output_dir: str) -> tuple[str, str]:
    separator_index = max(output_dir.rfind("/"), output_dir.rfind("\\"))
    if separator_index < 0:
        return "", output_dir
    return output_dir[:separator_index + 1], output_dir[separator_index + 1:]
