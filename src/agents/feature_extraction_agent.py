"""
Feature Extraction Agent - Handles radiomics and morphological feature extraction
"""

import os
from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
import json
from datetime import datetime

from .state import AgentState
from .config import Config
from .tool_manager import get_tool_manager
from .llm_json import parse_llm_json
from .path_utils import (
    collect_csv_paths,
    collect_nifti_paths,
    ensure_quality_output_dir,
    infer_dataset_role,
    parse_tool_result_paths,
    resolve_feature_csv_argument,
    resolve_source_csv_argument,
)
from .feature_plan_utils import (
    enforce_cartilage_analysis_plan,
    enforce_requested_feature_extraction_plan,
    ensure_safe_feature_extraction_plan,
    inject_online_ldse_model_path,
)
from .task_contracts import feature_type_for_tool
from .tool_result_utils import (
    parse_structured_tool_result,
    tool_error_message,
    tool_result_succeeded,
)


class FeatureExtractionAgent:
    """Agent that handles feature extraction tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = config.feature_extraction_agent.llm
        
        self.system_prompt = """你是一个专门负责骨关节炎（OA）MRI 影像特征提取的医学影像 agent。
        你的职责是帮助用户从分割后的图像中提取 radiomics、MEAS、LDSE 等特征。

        除非用户明确要求使用其他语言，否则所有 message、final_message、解释和总结都使用简体中文。
        但 JSON key、tool 名称、参数名、文件路径、模型名、特征类型名必须保持英文原样，不要翻译。

        可用特征提取工具：
        1. extract_radiomics_features - 提取 radiomics 特征（形状、纹理、一阶统计等）
           Parameters: 
           - csv_path (str): 包含 ImagePath、MaskPath、KLGrade 列的 CSV 文件
           - output_dir (str): 结果输出目录
           - num_regions (int): 提取特征的区域数量，默认 5
           - num_cases (int): 最多处理的病例数，默认 2
           - extract_all_cases (bool): 是否提取全部病例，默认 False
           - extract_all_features (bool): 是否提取包含滤波图像在内的全部 radiomics 特征，默认 False
           
        2. meas_features_extraction - 提取 MEAS 形态学和几何特征
           Parameters:
           - output_dir (str): 结果输出目录（FIRST PARAMETER）
           - csv_path (str): 包含 image_path、mask_path、case_id、koa_grade 列的 CSV 文件（SECOND PARAMETER）
           
        3. ldse_train_test_features_extraction - 联合提取训练集和测试集 LDSE 特征
           当训练集和测试集 CSV 都存在时，默认使用此工具，并且只调用一次。
           Parameters:
           - output_dir (str): 结果输出目录（FIRST PARAMETER）
           - train_csv_path (str): 真实训练集 CSV 路径
           - test_csv_path (str): 真实测试集 CSV 路径
           - n_components (int): 编码维度，默认 50
           - random_seed (int): 随机种子，默认 42

        4. ldse_features_extraction - 仅在单个数据集场景提取 LDSE 特征
           Parameters:
           - output_dir (str): 结果输出目录（FIRST PARAMETER）
           - csv_path (str): 真实 CSV 路径（SECOND PARAMETER）
           - mode (str): "fit_transform" 或 "transform"
           - dataset_role (str): "train" 或 "test"
           - model_path (str): 当 mode="transform" 时必须提供已验证的训练模型路径
           - n_components (int): 编码维度，默认 50
           - random_seed (int): 随机种子，默认 42
           CRITICAL: 禁止对测试集单独拟合；禁止分别独立调用训练集和测试集 LDSE。

        5. augment_online_extract_features - 使用在线数据增强提取鲁棒特征
           Parameters: csv_path, output_dir, n_augmentations, feature_type, aggregator, seed, dataset_role, ldse_model_path.
           - dataset_role (str): feature_type="LDSE" 时必须显式提供 "train" 或 "test"。
           - ldse_model_path (str): dataset_role="test" 时必须提供已验证的训练模型路径。
           - dataset_role="train" 时可从该训练 CSV 拟合；禁止从测试 CSV 拟合。
           - 同一计划包含训练和测试在线 LDSE 时，必须先执行训练调用，再把训练结果返回的 model_path 原样注入测试调用的 ldse_model_path；不得猜测模型路径。
           会生成带 _std 列的特征，用于不确定性估计。
           CRITICAL: 仅当用户明确要求 "data augmentation" / "数据增强" / "augmentation" / "augment" 时才调用。
           不要仅因为用户提到分类、LASSO、小样本或鲁棒性就主动加入数据增强。
           CRITICAL: feature_type 必须与用户明确要求的特征类型一致。
           - 用户说“增强并提取LDSE特征” -> feature_type="LDSE"
           - 用户说“增强并提取MEAS特征” -> feature_type="MEAS"
           - 用户说“增强并提取Radiomics特征” -> feature_type="radiomics"
           - 用户只说“数据增强”但没有说明类型时，检查请求其他位置是否提到特征类型，例如“LDSE分类” -> feature_type="LDSE"
           - 如果无法确定 feature_type，必须询问用户，不要猜测

        6. assess_feature_quality - 分类前评估特征 CSV 质量
           Parameters: feature_csv (str), output_dir (str), label_col (str),
           variance_threshold (float), correlation_threshold (float), cv_threshold (float).
           用于检测低方差、共线性和不稳定特征。
           Outputs: quality_report.txt, quality_metrics.json, filtered_features.csv
           CRITICAL: 仅当用户明确要求 "quality assessment" / "feature quality" / 特征质量评估时才调用。
           不要仅因为用户提到分类或 LASSO 就主动加入质量评估。

        7. augment_dataset_offline - 将增强后的膝关节 MRI 图像和 mask 保存到磁盘
           Parameters: csv_path (str), output_dir (str), multiplier (int, default 5), seed (int, default 42).
           对每个 image-mask pair 应用几何和强度增强，并保存增强后的 .nii.gz 文件和更新后的 labels CSV。
           仅当用户明确要求将数据增强作为特征提取前的独立步骤时使用。

        8. validate_cartilage_segmentation - 校验 OAI-ZIB 六分类分割能否用于软骨厚度分析
           Parameters: mask_path (.nii/.nii.gz)

        9. analyze_cartilage_thickness - 单病例 3D-MN 全局厚度 + 16 亚区联合分析
           Parameters: mask_path, output_dir, export_thickness_maps, export_subregion_mask
           输出全局/亚区 CSV、JSON，并可导出三类软骨的厚度 NIfTI 图。

        10. extract_cartilage_thickness_features - 批量提取患者级软骨厚度特征
           Parameters: csv_path, output_dir
           CSV 需包含 CaseID、MaskPath，可包含 SourceCaseID、KLGrade；结果可用于 LASSO/Filter。

        11. generate_cartilage_3d_mesh - 从六标签分割生成分标签三维三角网格
           Parameters: mask_path, output_dir, label_ids, smooth, output_formats
           label_ids 使用逗号分隔的 1-5；output_formats 支持 ply、stl、npz。
           输入必须是标签分割 NIfTI，不是原始灰度 MRI；输出顶点采用 ITK LPS 毫米物理坐标。

        CRITICAL - 数据增强和质量评估规则：
        - augment_online_extract_features 和 augment_dataset_offline 只能在用户明确要求数据增强时调用。
        - 调用 augment_online_extract_features 时，feature_type 必须与用户明确要求的特征类型一致。
        - LDSE 在线增强必须显式传入 dataset_role；测试集还必须传入 ldse_model_path，训练集可从其训练 CSV 拟合。
        - 训练在线 LDSE 结果会返回 model_path；同一计划中的后续测试调用必须使用该精确路径，并且不得在测试集上拟合。
        - 识别关键词："LDSE" -> feature_type="LDSE"，"MEAS" -> feature_type="MEAS"，"radiomics" / "影像组学" -> feature_type="radiomics"。
        - assess_feature_quality 只能在用户明确要求质量评估、feature quality 或 assess quality 等类似表达时调用。
        - 如果用户没有明确要求这些增强或质量评估步骤，必须从 tools 数组中省略。

        当用户请求特征提取时：
        1. 判断用户想提取哪类特征：radiomics、MEAS、LDSE、CartilageThickness
        2. 从用户输入或上下文中提取真实 CSV 文件路径
        3. 根据用户请求确定参数
        4. 只调用用户明确要求的特征提取工具
        5. 清晰报告结果，包括输出文件路径和特征数量

        重要说明：
        - Radiomics 特征：基础特征约 107 个，启用全部滤波图像时约 1316 个
        - MEAS 特征：形态学和几何特征，例如体积、表面积、厚度等
        - LDSE 特征：低维形状编码，默认 50 维
        - CartilageThickness：3D-MN 面积加权全局厚度、Wirth/Eckstein 16 亚区厚度和三维表面网格
        - 所有特征提取都需要分割后的图像或 mask 文件
        - 结果保存为包含特征值的 CSV 文件
        - meas_features_extraction、ldse_features_extraction 和 ldse_train_test_features_extraction 中，output_dir 是 FIRST PARAMETER
        - 用户经常会要求提取多种特征用于比较
        - 必须识别并调用完成用户请求所需的全部工具
        - CRITICAL: 必须自行决定 output_dir，使用带时间戳的描述性目录名以避免冲突。
          例如 "MEAS_Features_Train_20260121_123456" 或 "LDSE_Features_Test_20260121_123456"。
          时间戳格式为 YYYYMMDD_HHMMSS，目录名必须使用当前时间戳。

        始终用清晰、专业的中文说明特征提取流程和结果。

        CRITICAL: 必须严格输出 JSON，结构如下：
        {{
        "tools": [
            {{
                "tool": "meas_features_extraction",
                "arguments": {{
                    "output_dir": "MEAS_Features_Train_20260120_143025",
                    "csv_path": "path/to/data.csv"
                }},
                "message": "正在提取训练集 MEAS 特征。"
            }},
            {{
                "tool": "ldse_train_test_features_extraction",
                "arguments": {{
                    "output_dir": "LDSE_Features_TrainTest_20260120_143025",
                    "train_csv_path": "真实训练集 CSV 路径",
                    "test_csv_path": "真实测试集 CSV 路径",
                    "n_components": 50,
                    "random_seed": 42
                }},
                "message": "正在联合提取训练集和测试集 LDSE 特征。"
            }}
        ],
        "final_message": "已完成特征提取任务，结果已保存到对应输出目录。"
        }}
        
        CRITICAL 规则：
        1. "tools" 数组必须包含完成用户请求所需的全部工具
        2. 如果用户要求提取多种特征，必须为所有被提到的特征类型加入工具调用
        3. 如果用户要求使用多种方法，必须包含所有被提到的方法
        4. 工具必须按正确逻辑顺序调用
        5. 每个工具调用都必须包含 "message" 字段，用中文说明该步骤做什么
        6. "final_message" 应在所有工具执行后用中文总结完成了什么
        7. IMPORTANT: 当用户消息、对话历史或 "FILES GENERATED BY PREVIOUS STEPS" 中有真实文件路径时，必须使用这些 exact paths。若缺少路径，返回空 "tools" 数组，并在 final_message 中请用户提供路径。不要使用 "path/to/data.csv" 这类占位路径。
        8. 不要跳过用户明确要求的工具
        9. CRITICAL: 必须自行决定 output_dir，使用带时间戳的描述性目录名，不要使用 "MEAS_Features_Results" 这类固定默认值。
        10. 对 meas_features_extraction、ldse_features_extraction 和 ldse_train_test_features_extraction，output_dir 是 FIRST PARAMETER。
        11. 训练集和测试集都存在时，LDSE 必须使用一次 ldse_train_test_features_extraction；禁止独立 train/test LDSE 调用或测试集拟合。
        
        示例：
        
        示例 1：用户要求同时提取 MEAS 和 LDSE 特征
        响应应包含两个对应的特征提取工具，并填入合适参数。
        
        示例 2：用户只要求提取 MEAS 特征
        响应只应包含 MEAS 特征提取工具。
        
        示例 3：用户要求提取特征但没有提供 CSV 路径
        响应应包含空 "tools" 数组，并在 final_message 中要求用户提供真实文件路径。不要使用占位路径。
        
        示例 4：用户要求分别为训练集和测试集提取 LDSE 特征
        响应应只包含一次 ldse_train_test_features_extraction，并填入两条真实 CSV 路径。
        
        示例 5：用户要求为训练集和测试集提取多种特征
        响应应按正确顺序包含所有特征类型和所有数据集所需的多个工具调用。
        
        核心原则：始终根据用户意图识别需要调用的工具。优先使用对话或 "FILES GENERATED BY PREVIOUS STEPS" 中的真实路径。路径缺失时返回空 "tools" 数组，并在 final_message 中请用户提供。不要使用 "path/to/data.csv" 这类占位路径。输出目录名始终使用带时间戳的描述性名称。
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
    
    async def process(self, state: AgentState) -> AgentState:
        """Process feature extraction task"""
        stream_callback = state.get("stream_callback")
        
        messages = state["messages"]
        current_input = state["current_input"]
        
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        recent_context = ""
        for msg in messages[-6:]:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        process_input = f"""
        User query: {input_text}

        Recent conversation context: {recent_context}

        You are Feature Extraction Agent. Help user extract features from their segmented MRI images.
        Determine which feature extraction tools to use and what parameters to provide.
        
        CRITICAL: Identify if multiple tools need to be called.
        If user requests extracting multiple types of features, you MUST include tools for ALL feature types mentioned.
        
        CRITICAL: Identify if the same tool needs to be called multiple times for different datasets.
        For LDSE with both training and test CSV paths, call ldse_train_test_features_extraction exactly once.
        Never fit LDSE independently on test data or call independent train/test LDSE extraction tools.
        A single ldse_features_extraction call must use explicit mode, dataset_role, model_path when transforming test data, and random_seed.
        
        IMPORTANT: Only include tools when you have REAL file paths. If the user's message or
        conversation history DOES NOT contain actual file paths (e.g., CSV files, nifti files),
        return an EMPTY tools array and ask the user to provide paths in the final_message.
        NEVER use placeholder paths like "path/to/data.csv" — they will cause runtime errors.
        
        IMPORTANT: For meas_features_extraction, ldse_features_extraction, and ldse_train_test_features_extraction, output_dir parameter comes FIRST.
        
        CRITICAL: You MUST decide output_dir yourself. Use descriptive names with timestamps like "MEAS_Features_{current_time}" (use current timestamp: {current_time}). DO NOT use hardcoded default values or the example timestamps from the instructions.
        
        IMPORTANT: The current timestamp is {current_time}. You MUST use this exact timestamp in all output_dir names. Do not use any other timestamp values, including those shown in the example responses.
        
        Respond in JSON format with tools array and final_message.
        """

        # Extract file paths from "FILES GENERATED BY PREVIOUS STEPS" blocks
        # (injected by OrchestratorAgent for pipeline steps with output-to-input dependency)
        task_spec = state.get("task_spec", {})
        _prev_csv_paths: list[str] = []
        _prev_nifti_paths: list[str] = []

        def _remember_csv_path(path: str) -> None:
            if (
                path
                and "path/to/" not in path
                and "placeholder" not in path.lower()
                and path not in _prev_csv_paths
            ):
                _prev_csv_paths.append(path)

        # Files attached to the current request are authoritative.  Only fall
        # back to the most recent path-bearing message when this request has no
        # paths, so older uploads cannot be replayed as additional inputs.
        current_csv_paths = collect_csv_paths(input_text)
        current_nifti_paths = collect_nifti_paths(input_text)
        for _p in current_csv_paths:
            _remember_csv_path(_p)
        for _p in current_nifti_paths:
            if _p not in _prev_nifti_paths:
                _prev_nifti_paths.append(_p)

        allow_history_paths = task_spec.get("goal_type") != "BIOMARKER_DISCOVERY"
        need_history_csv = not current_csv_paths and allow_history_paths
        need_history_nifti = not current_nifti_paths and allow_history_paths
        for _msg in reversed(messages[-6:]):
            if not need_history_csv and not need_history_nifti:
                break
            _mt = ""
            if isinstance(_msg, AIMessage):
                _mt = _msg.content
            elif isinstance(_msg, HumanMessage):
                _mt = _msg.content
            if not _mt:
                continue
            if need_history_csv:
                message_csv_paths = collect_csv_paths(_mt)
                if message_csv_paths:
                    for _p in message_csv_paths:
                        _remember_csv_path(_p)
                    need_history_csv = False
            if need_history_nifti:
                message_nifti_paths = collect_nifti_paths(_mt)
                if message_nifti_paths:
                    for _p in message_nifti_paths:
                        if _p not in _prev_nifti_paths:
                            _prev_nifti_paths.append(_p)
                    need_history_nifti = False

        try:
            tool_manager = await get_tool_manager()
            if task_spec.get("goal_type") == "BIOMARKER_DISCOVERY":
                decision = {
                    "tools": [],
                    "final_message": (
                        "已按预注册 V1 方案提取训练集和测试集软骨厚度特征。"
                    ),
                }
            else:
                llm_response = self.llm.invoke(
                    self.prompt.format_messages(input=process_input)
                )
                try:
                    decision = parse_llm_json(llm_response.content)
                except (json.JSONDecodeError, ValueError):
                    decision = {
                        "tools": [],
                        "final_message": llm_response.content,
                    }
            # Include source paths selected in the current plan. This lets the
            # deterministic normalizer assign train/test roles even when the
            # user's short follow-up omits paths already known to the planner.
            for planned_call in decision.get("tools", []):
                if not isinstance(planned_call, dict):
                    continue
                planned_arguments = planned_call.get("arguments", {})
                if not isinstance(planned_arguments, dict):
                    continue
                for path_key in ("csv_path", "train_csv_path", "test_csv_path"):
                    planned_path = planned_arguments.get(path_key)
                    if isinstance(planned_path, str) and planned_path.lower().endswith(".csv"):
                        _remember_csv_path(planned_path)
                planned_mask = planned_arguments.get("mask_path")
                if (
                    isinstance(planned_mask, str)
                    and planned_mask.lower().endswith((".nii", ".nii.gz"))
                    and not _prev_nifti_paths
                    and planned_mask not in _prev_nifti_paths
                ):
                    _prev_nifti_paths.append(planned_mask)
            requested_feature_type = task_spec.get("feature_type")
            enforce_cartilage_analysis_plan(
                decision,
                input_text,
                _prev_csv_paths,
                _prev_nifti_paths,
                current_time,
            )
            if task_spec.get("requires_feature_extraction"):
                enforce_requested_feature_extraction_plan(
                    decision,
                    _prev_csv_paths,
                    requested_feature_type,
                    current_time,
                )
            self._ensure_paired_feature_extractions(decision, _prev_csv_paths)
            
            tool_results = []
            executed_tool_calls = []
            all_messages = []
            quality_output_dirs: set[str] = set()
            tool_failures: list[str] = []
            produced_feature_csvs: list[str] = []
            produced_feature_artifacts: list[dict[str, Any]] = []

            def _remember_produced_artifact(
                path: str,
                tool_name: str,
                feature_type: str,
                result_key: str,
            ) -> None:
                if path.lower().endswith(".csv") and path not in produced_feature_csvs:
                    produced_feature_csvs.append(path)
                if any(item.get("path") == path for item in produced_feature_artifacts):
                    return
                produced_feature_artifacts.append(
                    {
                        "schema_version": "1.0",
                        "run_id": state.get("task_run_id") or state.get("trace_id"),
                        "feature_type": feature_type,
                        "dataset_role": infer_dataset_role(path),
                        "path": path,
                        "producer_agent": "FEATURE_EXTRACTION_AGENT",
                        "producer_tool": tool_name,
                        "result_key": result_key,
                    }
                )
            
            if stream_callback:
                stream_callback("\n📋 工具执行计划:")
                for i, tool_call in enumerate(decision.get("tools", []), 1):
                    stream_callback(f"   {i}. {tool_call.get('tool', 'Unknown')}")
            
            if "tools" in decision and decision["tools"]:
                for i, tool_call in enumerate(decision["tools"], 1):
                    tool_name = tool_call.get("tool")
                    arguments = tool_call.get("arguments", {})
                    step_message = tool_call.get("message", f"Step {i}")

                    actual_feature_type = feature_type_for_tool(tool_name, arguments)
                    if (
                        task_spec.get("requires_feature_extraction")
                        and requested_feature_type
                        and actual_feature_type
                        and actual_feature_type != requested_feature_type
                    ):
                        raise ValueError(
                            "Feature extraction type mismatch: "
                            f"requested {requested_feature_type}, tool produces {actual_feature_type}"
                        )
                    
                    try:
                        # Feature extraction needs source metadata containing
                        # ImagePath/MaskPath. Historical feature tables are only
                        # valid for downstream quality/classification operations.
                        if "csv_path" in arguments and _prev_csv_paths:
                            if tool_name == "assess_feature_quality":
                                arguments["csv_path"] = resolve_feature_csv_argument(
                                    str(arguments["csv_path"]), _prev_csv_paths
                                )
                            else:
                                arguments["csv_path"] = resolve_source_csv_argument(
                                    str(arguments["csv_path"]), _prev_csv_paths
                                )
                        if "feature_csv" in arguments and _prev_csv_paths:
                            arguments["feature_csv"] = resolve_feature_csv_argument(
                                str(arguments["feature_csv"]), _prev_csv_paths
                            )

                        # Display the arguments that will actually be executed.
                        if stream_callback:
                            stream_callback(f"\n📌 步骤{i}: 调用工具 {tool_name}")
                            stream_callback(f"   参数: {json.dumps(arguments, ensure_ascii=False)}")

                        if tool_name == "assess_feature_quality" and "output_dir" in arguments:
                            arguments["output_dir"] = ensure_quality_output_dir(
                                str(arguments["output_dir"]),
                                str(arguments.get("feature_csv", "")),
                                quality_output_dirs,
                            )
                            quality_output_dirs.add(str(arguments["output_dir"]))

                        if (
                            tool_name == "augment_online_extract_features"
                            and str(arguments.get("feature_type", "")).upper() == "LDSE"
                            and arguments.get("dataset_role") == "test"
                            and not str(arguments.get("ldse_model_path", "")).strip()
                        ):
                            raise ValueError(
                                "Online LDSE test augmentation requires the exact model_path "
                                "returned by the preceding training call."
                            )

                        # Guard: reject placeholder paths before calling tool
                        _placeholder_patterns = ["path/to/", "path\\to\\", "placeholder"]
                        for _key, _val in arguments.items():
                            if isinstance(_val, str) and any(p in _val.lower() for p in _placeholder_patterns):
                                raise ValueError(
                                    f"Refusing to call {tool_name} with placeholder path: {_key}={_val!r}. "
                                    f"Provide real file paths in your request."
                                )

                        executed_tool_calls.append(tool_name)
                        tool_result = await tool_manager.call_tool(
                            tool_name,
                            arguments,
                            stream_callback=stream_callback,
                        )
                        if not tool_result_succeeded(tool_result):
                            raise RuntimeError(tool_error_message(tool_result))
                        if (
                            task_spec.get("goal_type") == "BIOMARKER_DISCOVERY"
                            and tool_name == "extract_cartilage_thickness_features"
                        ):
                            extraction_payload = parse_structured_tool_result(tool_result)
                            if (
                                not isinstance(extraction_payload, dict)
                                or extraction_payload.get("status") != "success"
                                or int(extraction_payload.get("failed_cases", 0) or 0) != 0
                            ):
                                raise RuntimeError(
                                    "生物标志物发现要求训练集和测试集厚度特征完整提取；"
                                    "存在失败病例时必须先修复分割/路径问题。"
                                )
                        
                        if stream_callback:
                            stream_callback(f"   ✅ 完成")
                        
                        result_message = f"\n📊 执行结果：\n{tool_result}"
                        all_messages.append(result_message)
                        tool_results.append(tool_result)
                        if (
                            tool_name == "augment_online_extract_features"
                            and str(arguments.get("feature_type", "")).upper() == "LDSE"
                            and arguments.get("dataset_role") == "train"
                        ):
                            for pending_call in decision.get("tools", [])[i:]:
                                inject_online_ldse_model_path(pending_call, tool_result)
                        parsed_paths = parse_tool_result_paths(tool_result)
                        for result_key, produced_path in parsed_paths:
                            is_csv = produced_path.lower().endswith(".csv")
                            if is_csv:
                                _remember_csv_path(produced_path)
                            if actual_feature_type and (
                                is_csv or actual_feature_type == "CartilageThickness"
                            ):
                                _remember_produced_artifact(
                                    produced_path,
                                    tool_name,
                                    actual_feature_type,
                                    result_key,
                                )
                        if isinstance(tool_result, str) and not parsed_paths:
                            for csv_path in collect_csv_paths(tool_result):
                                _remember_csv_path(csv_path)
                    except Exception as tool_error:
                        error_msg = f"工具执行失败：{str(tool_error)}"
                        if stream_callback:
                            stream_callback(f"   ❌ {error_msg}")
                        all_messages.append(f"\n❌ 执行结果：\n{error_msg}")
                        tool_results.append(error_msg)
                        tool_failures.append(f"{tool_name}: {tool_error}")
            
            required_extraction_executed = any(
                feature_type_for_tool(name) == requested_feature_type
                for name in executed_tool_calls
            )
            cartilage_request_completed = bool(
                requested_feature_type == "CartilageThickness"
                and required_extraction_executed
            )
            task_succeeded = not tool_failures and (
                not task_spec.get("requires_feature_extraction")
                or cartilage_request_completed
                or (required_extraction_executed and bool(produced_feature_csvs))
            )
            if not task_succeeded and not tool_failures:
                tool_failures.append(
                    f"未获得当前任务所需的 {requested_feature_type or 'feature'} 特征文件。"
                )
            final_message = decision.get("final_message", "") if task_succeeded else ""
            
            if all_messages:
                full_response = "\n\n" + "=" * 50 + "\n\n".join(all_messages) + "\n\n" + "=" * 50 + "\n\n" + final_message
            else:
                full_response = final_message
            
            return {
                **state,
                "output": full_response,
                "agent_name": "FEATURE_EXTRACTION_AGENT",
                "success": task_succeeded,
                "error_message": "; ".join(tool_failures) if tool_failures else None,
                "tool_calls": executed_tool_calls,
                "tool_results": tool_results,
                "current_artifacts": [
                    *(state.get("current_artifacts", []) or []),
                    *produced_feature_artifacts,
                ],
            }
            
        except Exception as e:
            error_msg = f"Error during feature extraction: {str(e)}"
            if stream_callback:
                stream_callback(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                **state,
                "output": error_msg,
                "agent_name": "FEATURE_EXTRACTION_AGENT",
                "success": False,
                "error_message": error_msg,
            }

    @staticmethod
    def _ensure_paired_feature_extractions(
        decision: dict,
        available_csv_paths: list[str],
    ) -> None:
        """Compatibility entry point for dependency-light plan normalization."""
        ensure_safe_feature_extraction_plan(decision, available_csv_paths)
