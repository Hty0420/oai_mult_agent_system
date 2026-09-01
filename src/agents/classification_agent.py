"""
Classification Agent - Handles KOA grading and classification tasks
"""

from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
import os
import json
from datetime import datetime

from .state import AgentState
from .config import Config
from .tool_manager import get_tool_manager
from .llm_json import parse_llm_json
from .path_utils import (
    collect_csv_paths,
    ensure_quality_output_dir,
    resolve_feature_csv_argument,
    select_train_test_feature_paths,
)
from .task_contracts import validate_classification_artifacts
from .tool_result_utils import tool_error_message, tool_result_succeeded


def _structured_tool_status(result: object) -> str | None:
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("status")
    if not isinstance(status, str):
        return None
    normalized = status.strip().lower()
    return normalized or None


def _structured_tool_output_dir(result: object) -> str | None:
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(payload, dict):
        return None
    output_dir = payload.get("output_dir")
    if not isinstance(output_dir, str):
        return None
    normalized = output_dir.strip()
    return normalized or None


class ClassificationAgent:
    """Agent that handles KOA grading and classification tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = config.classification_agent.llm
        
        self.system_prompt = """你是一个专门负责骨关节炎（OA）分类和 Kellgren-Lawrence（KL）分级的医学影像 agent。
        你的职责是帮助用户基于膝关节 OA MRI 特征数据执行分类、KL 分级预测，并读取或解释分类结果。

        除非用户明确要求使用其他语言，否则所有 message、final_message、解释和总结都使用简体中文。
        但 JSON key、tool 名称、参数名、文件路径、模型名、指标名必须保持英文原样，不要翻译。
        工具调用协议必须严格保持英文标识，例如 "tools"、"arguments"、"train_path"、"test_path"、"output_dir"。

        可用分类工具：
        1. koa_lasso_classification - 基于 LASSO 的特征选择和二分类
           Parameters: train_path (str), test_path (str), output_dir (str)
           
        2. koa_filter_classification - 基于 Filter 方法的特征选择和分类
           Parameters: train_path (str), test_path (str), output_dir (str), feature_type (str)
           IMPORTANT: 实际输出目录就是你提供的 output_dir，不会额外创建子目录。
           读取 Filter 分类结果时，必须使用分类工具调用时的同一个 output_dir。
           调用该工具时必须包含 feature_type 参数，并使用 DETECTED FEATURE TYPE 的值。
           
        3. combine_radiomics_features_by_region - 按区域组合 radiomics 特征
           Parameters: csv_path (str), output_dir (str)
           
        4. combine_radiomics_features_all - 组合全部 radiomics 特征
           Parameters: csv_path (str), output_dir (str)
           
        5. lasso_comb_radiomics_features_by_region - 使用按区域组合后的特征进行 LASSO 分类
           Parameters: train_path (str), test_path (str), output_dir (str)
           
        6. lasso_comb_radiomics_features_all - 使用全部组合特征进行 LASSO 分类
           Parameters: train_path (str), test_path (str), output_dir (str)
           
        7. read_classification_results - 读取并展示分类结果
           Parameters: results_dir (str) - 注意：实际参数名是 'classification_result_dir'

        8. assess_feature_quality - 分类前评估特征质量
           Parameters: feature_csv (str), output_dir (str), label_col (str),
           variance_threshold (float), correlation_threshold (float), cv_threshold (float)
           Outputs: quality_report.txt, quality_metrics.json, filtered_features.csv
           用于在分类前检测和过滤低方差、共线性或不稳定特征，以改善分类结果。

        当用户请求分类时：
        0. 检查特征数据集是否已有 filtered_features.csv
           -> 如果存在，优先使用 filtered features 进行分类
           -> 如果不存在且用户明确要求质量评估，可先调用 assess_feature_quality
        1. 判断用户想使用的分类方法；未指定时默认使用 LASSO
        2. 从用户输入或上下文中提取 train/test 数据集路径
        3. 判断是否需要多个工具，例如质量评估 -> 分类 -> 读取结果
        4. 按正确顺序调用工具，质量评估必须早于分类
        5. 清晰报告结果，包括 accuracy、confusion matrix、feature importance 等

        重要说明：
        - 分类需要训练集和测试集数据
        - LASSO 会在分类前执行特征选择
        - Filter 方法使用统计检验进行特征选择
        - 组合特征可能改善分类表现
        - 结果包括 accuracy、precision、recall、F1-score 和 confusion matrix
        - 用户经常会在分类后要求读取分类结果
        - 必须识别并调用完成用户请求所需的全部工具
        - CRITICAL: 必须自行决定 output_dir，使用带时间戳的描述性目录名以避免冲突。
          例如 "LASSO_Classification_20260121_123456" 或 "Filter_Classification_20260121_123456"。
          时间戳格式为 YYYYMMDD_HHMMSS，目录名必须使用当前时间戳。

        始终用清晰、专业的中文说明分类流程和结果。

        CRITICAL: 必须严格输出 JSON，结构如下：
        {{
        "tools": [
            {{
                "tool": "tool_name",
                "arguments": {{
                    "train_path": "path/to/train.csv",
                    "test_path": "path/to/test.csv",
                    "output_dir": "LASSO_Classification_20260121_123456"
                }},
                "message": "正在使用 LASSO 方法进行分类。"
            }},
            {{
                "tool": "read_classification_results",
                "arguments": {{
                    "classification_result_dir": "LASSO_Classification_20260121_123456"
                }},
                "message": "正在读取分类结果。"
            }}
        ],
        "final_message": "已完成分类并读取结果，请查看上方分类报告摘要。"
        }}
        
        CRITICAL 规则：
        1. "tools" 数组必须包含完成用户请求所需的全部工具
        2. 如果用户要求分类并读取结果，必须同时包含：
           - 一个分类工具，例如 koa_lasso_classification 或 koa_filter_classification
           - read_classification_results 工具
        3. 如果用户要求使用多个模型，必须包含所有被提到的模型工具
        4. 工具必须按正确逻辑顺序调用，例如先分类再读取结果
        5. 每个工具调用都必须包含 "message" 字段，用中文说明该步骤做什么
        6. "final_message" 应在所有工具执行后用中文总结完成了什么
        7. IMPORTANT: 只有在用户请求或对话历史中存在真实文件路径时才加入工具调用。如果缺少真实路径，返回空 "tools" 数组，并在 final_message 中请用户提供路径。不要使用 "path/to/train.csv" 这类占位路径。
        8. 不要跳过用户明确要求的工具
        9. CRITICAL: 必须自行决定 output_dir，使用带时间戳的描述性目录名，不要使用 "KOA_LASSO_Results" 这类固定默认值。
        10. 调用 read_classification_results 时，必须使用分类工具使用过的同一个 output_dir。
        
        示例：
        
        示例 1：用户要求执行 LASSO 分类并读取分类报告
        响应应同时包含分类工具和读取结果工具，并填入合适参数。
        
        示例 2：用户只要求执行 LASSO 分类
        响应只应包含分类工具。
        
        示例 3：用户只要求读取分类结果
        响应只应包含 read_classification_results 工具。
        
        示例 4：用户要求执行 Filter 分类并读取结果
        响应应同时包含 Filter 分类工具和读取结果工具。
        
        示例 5：用户要求使用已有训练集和测试集特征进行分类
        响应应使用对应 train_path 和 test_path 调用分类工具。
        
        核心原则：始终根据用户意图识别需要调用的工具。参数缺失时，在 final_message 中请用户补充。不要使用占位路径。输出目录名始终使用带时间戳的描述性名称。
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
    
    async def process(self, state: AgentState) -> AgentState:
        """Process classification task"""
        stream_callback = state.get("stream_callback")
        
        messages = state["messages"]
        current_input = state["current_input"]
        
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        recent_context = ""
        last_classification_output_dir = None
        feature_type = "MEAS"
        collected_csv_paths: list[str] = []
        current_input_csv_paths: list[str] = []
        scoped_csv_paths: list[str] = []
        train_feature_dir = None
        test_feature_dir = None
        task_spec = state.get("task_spec", {})
        selected_pipeline = dict(state.get("selected_pipeline", {}) or {})
        task_run_id = state.get("task_run_id") or state.get("trace_id")
        current_artifacts = state.get("current_artifacts", []) or []
        allow_history_reuse = bool(task_spec.get("allow_history_reuse"))
        current_artifact_paths = [
            str(item.get("path"))
            for item in current_artifacts
            if item.get("path") and item.get("run_id") == task_run_id
        ]

        def _remember_csv_path(path: str) -> None:
            if (
                path
                and "path/to/" not in path
                and "placeholder" not in path.lower()
                and path not in collected_csv_paths
            ):
                collected_csv_paths.append(path)

        def _infer_feature_type(text: str) -> str | None:
            upper_text = text.upper()
            if "LDSE" in upper_text:
                return "LDSE"
            if "MEAS" in upper_text:
                return "MEAS"
            if "RADIOMICS" in upper_text or "RADIOLOGY" in upper_text:
                return "Radiomics"
            return None
        
        for msg in messages[-6:]:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
                for _p in collect_csv_paths(msg.content):
                    _remember_csv_path(_p)
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
                for _p in collect_csv_paths(msg.content):
                    _remember_csv_path(_p)
                
                import re
                
                patterns = [
                    r'(LDSE_Features_[^\s]+|MEAS_Features_[^\s]+|Radiomics_Features_[^\s]+|KOA_LASSO_Results[^\s]+|KOA_Filter_Results[^\s]+)',
                    r'output_dir["\s]*[:=]["\s]*["\s]*([^"\s,)]+)'
                ]
                
                direct_dir_pattern = r'保存在\s+([^\s]+)\s+目录中'
                direct_matches = re.findall(direct_dir_pattern, msg.content)
                
                if direct_matches:
                    for match in direct_matches:
                        directory_path = match
                        
                        if "LDSE_Features" in directory_path:
                            feature_type = "LDSE"
                            if "Train" in directory_path:
                                train_feature_dir = directory_path
                            elif "Test" in directory_path:
                                test_feature_dir = directory_path
                            if last_classification_output_dir is None:
                                last_classification_output_dir = directory_path
                        elif "Radiomics_Features" in directory_path:
                            feature_type = "Radiomics"
                            if "Train" in directory_path:
                                train_feature_dir = directory_path
                            elif "Test" in directory_path:
                                test_feature_dir = directory_path
                            if last_classification_output_dir is None:
                                last_classification_output_dir = directory_path
                        elif "MEAS_Features" in directory_path:
                            feature_type = "MEAS"
                            if "Train" in directory_path:
                                train_feature_dir = directory_path
                            elif "Test" in directory_path:
                                test_feature_dir = directory_path
                            if last_classification_output_dir is None:
                                last_classification_output_dir = directory_path
                else:
                    for pattern in patterns:
                        matches = re.findall(pattern, msg.content)
                        if matches:
                            for match in matches:
                                directory_path = match
                                
                                if "LDSE_Features" in directory_path:
                                    feature_type = "LDSE"
                                    if last_classification_output_dir is None:
                                        last_classification_output_dir = directory_path
                                elif "Radiomics_Features" in directory_path:
                                    feature_type = "Radiomics"
                                    if last_classification_output_dir is None:
                                        last_classification_output_dir = directory_path
                                elif "MEAS_Features" in directory_path:
                                    feature_type = "MEAS"
                                    if last_classification_output_dir is None:
                                        last_classification_output_dir = directory_path
                            break

        current_feature_type = _infer_feature_type(input_text)
        if current_feature_type:
            feature_type = current_feature_type

        for _p in collect_csv_paths(input_text):
            _remember_csv_path(_p)
            if _p not in current_input_csv_paths:
                current_input_csv_paths.append(_p)

        for _p in current_artifact_paths:
            _remember_csv_path(_p)
            if _p not in current_input_csv_paths:
                current_input_csv_paths.append(_p)

        current_train_path, current_test_path = select_train_test_feature_paths(current_input_csv_paths)
        if current_train_path and current_test_path:
            train_feature_dir = current_train_path
            test_feature_dir = current_test_path
            scoped_csv_paths = current_input_csv_paths.copy()
        elif allow_history_reuse:
            collected_train_path, collected_test_path = select_train_test_feature_paths(collected_csv_paths)
            if collected_train_path:
                train_feature_dir = collected_train_path
            if collected_test_path:
                test_feature_dir = collected_test_path
            scoped_csv_paths = collected_csv_paths
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check if filtered features exist for the detected feature directories
        filtered_train_path = None
        filtered_test_path = None
        if train_feature_dir:
            potential_filtered = os.path.join(train_feature_dir, "filtered_features.csv")
            if os.path.exists(potential_filtered):
                filtered_train_path = potential_filtered
        if test_feature_dir:
            potential_filtered = os.path.join(test_feature_dir, "filtered_features.csv")
            if os.path.exists(potential_filtered):
                filtered_test_path = potential_filtered

        quality_hint = ""
        if filtered_train_path and filtered_test_path:
            quality_hint = (
                f"\nQUALITY-ASSESSED FEATURES AVAILABLE:\n"
                f"  Train filtered: {filtered_train_path}\n"
                f"  Test filtered: {filtered_test_path}\n"
                f"  PREFER using these filtered feature files for classification "
                f"instead of the original files.\n"
            )
        elif feature_type:
            quality_hint = (
                f"\nNOTE: Features are {feature_type} type. "
                f"Consider suggesting assess_feature_quality before classification "
                f"to detect and filter problematic features.\n"
            )
        
        process_input = f"""
        User query: {input_text}

        Recent conversation context: {recent_context}

        You are Classification Agent. Help user perform KOA classification and KL grading.
        Determine which classification tools to use and what parameters to provide.
        
        CRITICAL: Identify if multiple tools need to be called.
        If user requests classification and reading results, you MUST include BOTH:
           - A classification tool (koa_lasso_classification or koa_filter_classification)
           - The read_classification_results tool
        
        IMPORTANT: If user requests reading classification results without specifying a directory, 
        check the conversation history for a recent classification operation and use its output directory.
        
        IMPORTANT: If user requests classification after feature extraction, 
        check the conversation history for recent feature extraction operations and use their output directories as input for classification.
        
        RECENT OPERATION OUTPUT DIRECTORY: {last_classification_output_dir if last_classification_output_dir else "No recent output directory found"}
        
        DETECTED FEATURE TYPE: {feature_type}
        {quality_hint}

        DETECTED GENERATED FEATURE CSV FILES:
        Train: {train_feature_dir if train_feature_dir else "Not found"}
        Test: {test_feature_dir if test_feature_dir else "Not found"}
        
        IMPORTANT: Only include tools when you have REAL file paths. If actual paths are not found
        in the conversation history, return an EMPTY tools array and ask the user to provide
        them. NEVER use placeholder paths like "path/to/train.csv" — they cause runtime errors.
        
        CRITICAL: When processing multi-step tasks that involve both feature extraction and classification:
        1. First identify the feature extraction results directory from the conversation history
        2. Use that directory as input for the classification tool
        3. For Filter classification, ALWAYS include the feature_type parameter using the DETECTED FEATURE TYPE value
        4. For Filter classification, the output directory is the same as the one you provide, no subdirectory is created
        5. When reading Filter classification results, use the same output_dir that was used for the classification tool
        6. Ensure all steps are executed in the correct order
        
        CRITICAL: For read_classification_results tool, the actual parameter name is 'classification_result_dir', not 'results_dir'.
        
        CRITICAL: You MUST decide the output_dir yourself. Use descriptive names with timestamps like "LASSO_Classification_{current_time}" (use current timestamp: {current_time}). DO NOT use hardcoded default values or the example timestamps from the instructions.
        
        IMPORTANT: The current timestamp is {current_time}. You MUST use this exact timestamp in all output_dir names. Do not use any other timestamp values, including those shown in the example responses.
        
        Respond in JSON format with tools array and final_message.
        """
        
        try:
            tool_manager = await get_tool_manager()
            
            llm_response = self.llm.invoke(self.prompt.format_messages(input=process_input))
            
            print(f"[DEBUG] ClassificationAgent - LLM 原始响应: {llm_response.content[:500] if len(llm_response.content) > 500 else llm_response.content}")
            
            try:
                decision = parse_llm_json(llm_response.content)
                print(f"[DEBUG] ClassificationAgent - 解析后的 decision: {json.dumps(decision, ensure_ascii=False, indent=2)}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[DEBUG] ClassificationAgent - JSON 解析失败: {e}")
                decision = {
                    "tools": [],
                    "final_message": llm_response.content
                }

            planned_tools = decision.get("tools")
            if not isinstance(planned_tools, list):
                planned_tools = []
                decision["tools"] = planned_tools
            classifier_names = {"koa_lasso_classification", "koa_filter_classification"}
            if task_spec.get("requires_classification") and selected_pipeline:
                planned_tools[:] = [
                    call
                    for call in planned_tools
                    if not (
                        isinstance(call, dict)
                        and call.get("tool") in classifier_names
                    )
                ]
                selected_method = str(
                    selected_pipeline.get("classification_method") or ""
                ).strip().upper()
                classifier = (
                    "koa_filter_classification"
                    if selected_method == "FILTER"
                    else "koa_lasso_classification"
                )
                arguments: dict[str, Any] = {
                    "output_dir": (
                        f"Filter_Classification_{current_time}"
                        if classifier == "koa_filter_classification"
                        else f"LASSO_Classification_{current_time}"
                    )
                }
                if classifier == "koa_filter_classification":
                    recommended = dict(
                        selected_pipeline.get("validation_recommended_params")
                        or task_spec.get("model_recipe_params")
                        or {}
                    )
                    arguments.update(
                        {
                            "feature_type": selected_pipeline.get(
                                "feature_type", feature_type
                            ),
                            "model_recipe": selected_pipeline.get(
                                "model_recipe", "ANOVA_TOPK_L2"
                            ),
                            "k": recommended.get("selector__k", 10),
                            "model_c": recommended.get("model__C", 1.0),
                        }
                    )
                planned_tools.append(
                    {
                        "tool": classifier,
                        "arguments": arguments,
                        "message": "严格执行训练集内部选出的唯一原子配方。",
                    }
                )
            elif task_spec.get("requires_classification") and not any(
                call.get("tool") in classifier_names
                for call in planned_tools if isinstance(call, dict)
            ):
                classifier = (
                    "koa_filter_classification"
                    if "filter" in input_text.lower()
                    else "koa_lasso_classification"
                )
                planned_tools.append(
                    {
                        "tool": classifier,
                        "arguments": {
                            "output_dir": (
                                f"Filter_Classification_{current_time}"
                                if classifier == "koa_filter_classification"
                                else f"LASSO_Classification_{current_time}"
                            )
                        },
                        "message": "使用当前任务产生的训练集和测试集特征执行分类。",
                    }
                )
            if task_spec.get("requires_report_read"):
                read_calls = [
                    call for call in planned_tools
                    if isinstance(call, dict)
                    and call.get("tool") == "read_classification_results"
                ]
                planned_tools[:] = [
                    call for call in planned_tools
                    if not (
                        isinstance(call, dict)
                        and call.get("tool") == "read_classification_results"
                    )
                ]
                planned_tools.append(
                    read_calls[0] if read_calls else {
                        "tool": "read_classification_results",
                        "arguments": {},
                        "message": "读取本次分类产生的报告。",
                    }
                )
            
            tool_results = []
            executed_tool_calls = []
            all_messages = []
            classification_output_dir = None  # Track LASSO/Filter output_dir for read_classification_results
            classification_succeeded = False
            classification_failed = False
            report_failed = False
            quality_output_dirs: set[str] = set()

            import re
            for msg in messages[-6:]:
                msg_text = ""
                if isinstance(msg, AIMessage):
                    msg_text = msg.content
                elif isinstance(msg, HumanMessage):
                    msg_text = msg.content
                if not msg_text:
                    continue

                # Pattern 1: Chinese "saved in XXX directory" format
                direct_matches = re.findall(r'保存在\s+([^\s]+)\s+目录中', msg_text)
                for directory_path in direct_matches:
                    if "Train" in directory_path:
                        train_feature_dir = directory_path
                    elif "Test" in directory_path:
                        test_feature_dir = directory_path

                # Pattern 2: "FILES GENERATED BY PREVIOUS STEPS" injected by OrchestratorAgent
                # Format: "- label: /path/to/file.csv"
                injected_matches = re.findall(
                    r'^\s*-\s+\S+\s*:\s*(\S+\.csv)',
                    msg_text, re.MULTILINE
                )
                for csv_path in injected_matches:
                    if "path/to/" in csv_path or "placeholder" in csv_path.lower():
                        continue  # skip placeholders
                    # Infer train/test from the label on the same line
                    line_match = re.search(
                        rf'^\s*-\s+(\S+)\s*:\s*{re.escape(csv_path)}',
                        msg_text, re.MULTILINE
                    )
                    label = line_match.group(1).lower() if line_match else ""
                    if "train" in label or "train" in csv_path.lower():
                        train_feature_dir = csv_path
                    elif "test" in label or "test" in csv_path.lower():
                        test_feature_dir = csv_path
                    elif train_feature_dir is None:
                        train_feature_dir = csv_path  # default to train
                    _remember_csv_path(csv_path)

                # Pattern 3: Explicit CSV file paths (from tool execution results)
                csv_matches = re.findall(
                    r'(?:feature_csv|output_csv|filtered_csv|ldse_features_summary|meas_features_summary|radiomics_features_summary|augmented_features)[^\s]*\.csv',
                    msg_text, re.IGNORECASE
                )
                for csv_path in csv_matches:
                    if "path/to/" in csv_path or "placeholder" in csv_path.lower():
                        continue  # skip placeholders
                    if re.search(r'(?:train|Train)', csv_path):
                        train_feature_dir = csv_path
                    elif re.search(r'(?:test|Test)', csv_path):
                        test_feature_dir = csv_path
                    elif train_feature_dir is None:
                        train_feature_dir = csv_path  # default to train
                    _remember_csv_path(csv_path)

                # Pattern 4: Generic CSV paths in "execution result" blocks
                exec_matches = re.findall(
                    r'执行结果[：:]\s*\n\s*([^\s]*\.csv)',
                    msg_text
                )
                for csv_path in exec_matches:
                    if "path/to/" in csv_path or "placeholder" in csv_path.lower():
                        continue
                    if re.search(r'(?:train|Train)', csv_path):
                        train_feature_dir = csv_path
                    elif re.search(r'(?:test|Test)', csv_path):
                        test_feature_dir = csv_path
                    elif train_feature_dir is None:
                        train_feature_dir = csv_path
                    _remember_csv_path(csv_path)

                for csv_path in collect_csv_paths(msg_text):
                    _remember_csv_path(csv_path)

            current_train_path, current_test_path = select_train_test_feature_paths(current_input_csv_paths)
            if current_train_path and current_test_path:
                train_feature_dir = current_train_path
                test_feature_dir = current_test_path
                scoped_csv_paths = current_input_csv_paths.copy()
            elif allow_history_reuse:
                collected_train_path, collected_test_path = select_train_test_feature_paths(collected_csv_paths)
                if collected_train_path:
                    train_feature_dir = collected_train_path
                if collected_test_path:
                    test_feature_dir = collected_test_path
                scoped_csv_paths = collected_csv_paths
            
            if stream_callback:
                stream_callback("\n📋 工具执行计划:")
                for i, tool_call in enumerate(decision.get("tools", []), 1):
                    stream_callback(f"   {i}. {tool_call.get('tool', 'Unknown')}")
            
            print(f"[DEBUG] ClassificationAgent - decision.get('tools'): {decision.get('tools', [])}")
            
            if "tools" in decision and decision["tools"]:
                print(f"[DEBUG] ClassificationAgent - 进入工具循环，共 {len(decision['tools'])} 个工具")
                for i, tool_call in enumerate(decision["tools"], 1):
                    tool_name = tool_call.get("tool")
                    print(f"[DEBUG] ClassificationAgent - 处理工具 {i}: {tool_name}")
                    arguments = tool_call.get("arguments", {})
                    step_message = tool_call.get("message", f"Step {i}")

                    if tool_name == "read_classification_results" and not (
                        classification_succeeded and classification_output_dir
                    ):
                        continue
                    
                    if stream_callback:
                        stream_callback(f"\n📌 步骤{i}: 调用工具 {tool_name}")
                        stream_callback(f"   参数: {json.dumps(arguments, ensure_ascii=False)}")
                    
                    if (tool_name == "koa_lasso_classification" or tool_name == "koa_filter_classification") and train_feature_dir and test_feature_dir:
                        arguments["train_path"] = train_feature_dir
                        arguments["test_path"] = test_feature_dir

                    if tool_name in ("koa_lasso_classification", "koa_filter_classification"):
                        if selected_pipeline:
                            expected_tool = (
                                "koa_filter_classification"
                                if str(
                                    selected_pipeline.get("classification_method")
                                    or ""
                                ).strip().upper() == "FILTER"
                                else "koa_lasso_classification"
                            )
                            if tool_name != expected_tool:
                                raise ValueError(
                                    "classification tool does not match the selected pipeline"
                                )
                            if tool_name == "koa_filter_classification":
                                recommended = dict(
                                    selected_pipeline.get(
                                        "validation_recommended_params"
                                    )
                                    or task_spec.get("model_recipe_params")
                                    or {}
                                )
                                arguments.update(
                                    {
                                        "feature_type": selected_pipeline.get(
                                            "feature_type", feature_type
                                        ),
                                        "model_recipe": selected_pipeline.get(
                                            "model_recipe", "ANOVA_TOPK_L2"
                                        ),
                                        "k": recommended.get("selector__k", 10),
                                        "model_c": recommended.get("model__C", 1.0),
                                    }
                                )
                        validate_classification_artifacts(
                            train_path=str(arguments.get("train_path", "")),
                            test_path=str(arguments.get("test_path", "")),
                            task_spec=task_spec,
                            current_artifacts=current_artifacts,
                            run_id=task_run_id,
                        )
                    
                    if tool_name == "read_classification_results":
                        arguments["classification_result_dir"] = classification_output_dir

                    for _csv_key in ("train_path", "test_path", "feature_csv"):
                        if _csv_key in arguments and scoped_csv_paths:
                            arguments[_csv_key] = resolve_feature_csv_argument(
                                str(arguments[_csv_key]),
                                scoped_csv_paths,
                            )

                    if tool_name == "assess_feature_quality" and "output_dir" in arguments:
                        arguments["output_dir"] = ensure_quality_output_dir(
                            str(arguments["output_dir"]),
                            str(arguments.get("feature_csv", "")),
                            quality_output_dirs,
                        )
                        quality_output_dirs.add(str(arguments["output_dir"]))
                    
                    execution_registered = False
                    failed_tool_result = None
                    try:
                        # Guard: reject placeholder paths before calling tool
                        _placeholder_patterns = ["path/to/", "path\\to\\", "placeholder"]
                        for _key, _val in arguments.items():
                            if isinstance(_val, str) and any(p in _val.lower() for p in _placeholder_patterns):
                                raise ValueError(
                                    f"Refusing to call {tool_name} with placeholder path: {_key}={_val!r}. "
                                    f"Provide real file paths in your request."
                                )

                        executed_tool_calls.append(tool_name)
                        execution_registered = True
                        tool_result = await tool_manager.call_tool(
                            tool_name,
                            arguments,
                            stream_callback=stream_callback,
                        )
                        if not tool_result_succeeded(tool_result):
                            failed_tool_result = tool_result
                            raise RuntimeError(tool_error_message(tool_result))
                        
                        # Track output_dir from LASSO or Filter for downstream read_classification_results
                        if tool_name in ("koa_lasso_classification", "koa_filter_classification"):
                            classification_succeeded = _structured_tool_status(tool_result) == "success"
                            classification_failed = (
                                classification_failed or not classification_succeeded
                            )
                            classification_output_dir = (
                                _structured_tool_output_dir(tool_result)
                                if classification_succeeded
                                else None
                            )
                        
                        # 处理分类报告，将结果传给大模型进行总结
                        if tool_name == "read_classification_results":
                            print(f"[DEBUG] 进入 read_classification_results 处理分支")
                            print(f"[DEBUG] tool_result 类型: {type(tool_result)}")
                            # 将分类报告传给大模型进行智能总结
                            if isinstance(tool_result, str):
                                print(f"[DEBUG] tool_result 是字符串，长度: {len(tool_result)}")
                                print(f"[DEBUG] tool_result 前500字符: {tool_result[:500]}")
                                summary_prompt = f"""请对以下分类报告进行总结，挑重点告知给用户：

分类报告内容：
{tool_result}

请用中文简要总结：
1. 模型的整体性能（准确率、精确率、召回率、F1分数等关键指标）
2. 模型的表现评价（良好/一般/需要改进）
3. 任何值得注意的发现或建议

请用简洁、专业的语言进行总结。"""
                                
                                try:
                                    print(f"[DEBUG] 正在调用大模型进行总结...")
                                    summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
                                    print(f"[DEBUG] 大模型调用成功，返回类型: {type(summary_response)}")
                                    summary = summary_response.content
                                    print(f"[DEBUG] 大模型总结结果: {summary[:200] if len(summary) > 200 else summary}")
                                except Exception as e:
                                    print(f"[DEBUG] 大模型调用失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    summary = tool_result
                            else:
                                print(f"[DEBUG] tool_result 不是字符串，转换为字符串")
                                summary = str(tool_result)
                            
                            result_message = f"\n📊 分类报告总结：\n{summary}"
                        else:
                            result_message = f"\n📊 执行结果：\n{tool_result}"
                        
                        if stream_callback:
                            stream_callback(f"   ✅ 完成")
                        
                        all_messages.append(result_message)
                        tool_results.append(tool_result)
                        if isinstance(tool_result, str):
                            for _p in collect_csv_paths(tool_result):
                                _remember_csv_path(_p)
                        elif isinstance(tool_result, dict):
                            for _val in tool_result.values():
                                if isinstance(_val, str) and _val.lower().endswith(".csv"):
                                    _remember_csv_path(_val)
                    except Exception as tool_error:
                        if tool_name in ("koa_lasso_classification", "koa_filter_classification"):
                            classification_succeeded = False
                            classification_failed = True
                            classification_output_dir = None
                        if tool_name == "read_classification_results":
                            report_failed = True
                        failure_detail = (
                            str(failed_tool_result)
                            if failed_tool_result is not None
                            else str(tool_error)
                        )
                        error_msg = f"工具执行失败：{failure_detail}"
                        if stream_callback:
                            stream_callback(f"   ❌ {error_msg}")
                        all_messages.append(f"\n❌ 执行结果：\n{error_msg}")
                        if execution_registered:
                            tool_results.append(
                                failed_tool_result
                                if failed_tool_result is not None
                                else error_msg
                            )
            
            final_message = decision.get("final_message", "")
            if classification_failed or report_failed:
                final_message = ""
            
            if all_messages:
                full_response = "\n\n" + "=" * 50 + "\n\n".join(all_messages) + "\n\n" + "=" * 50 + "\n\n" + final_message
            else:
                full_response = final_message
            
            return {
                **state,
                "output": full_response,
                "agent_name": "CLASSIFICATION_AGENT",
                "success": not classification_failed and not report_failed and (
                    not task_spec.get("requires_classification")
                    or any(
                        name in ("koa_lasso_classification", "koa_filter_classification")
                        for name in executed_tool_calls
                    )
                ) and (
                    not task_spec.get("requires_report_read")
                    or "read_classification_results" in executed_tool_calls
                ),
                "tool_calls": executed_tool_calls,
                "tool_results": tool_results
            }
            
        except Exception as e:
            error_msg = f"Error during classification: {str(e)}"
            if stream_callback:
                stream_callback(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                **state,
                "output": error_msg,
                "agent_name": "CLASSIFICATION_AGENT",
                "success": False,
                "error_message": error_msg,
            }
