"""
Feature Combination Agent - Handles feature combination tasks using radiomics features
"""

from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
import json
from datetime import datetime

from .state import AgentState
from .config import Config
from .tool_manager import get_tool_manager
from .llm_json import parse_llm_json


class FeatureCombinationAgent:
    """Agent that handles feature combination tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = config.feature_combination_agent.llm
        
        self.system_prompt = """你是一个专门负责 radiomics 特征组合的医学影像 agent。
        你的职责是帮助用户将不同 radiomics 特征文件按区域或不按区域进行组合，用于骨关节炎（OA）分析。

        除非用户明确要求使用其他语言，否则所有 message、final_message、解释和总结都使用简体中文。
        但 JSON key、tool 名称、参数名、文件路径、模型名、特征列名必须保持英文原样，不要翻译。
        工具调用协议必须严格保持英文标识，例如 "tools"、"arguments"、"feature_files"、"output_root"。

        可用特征组合工具：
        1. combine_radiomics_features_by_region - 按区域组合特征
           Parameters:
           - feature_files (list): 特征文件路径列表
           - output_root (str): 输出根目录
           - regions (int): 区域数量，默认 5
           - key_cols (list): 用于合并的非特征列，默认 ["CaseID", "Region", "KLGrade"]
           
        2. combine_radiomics_features_all - 不按区域划分，组合全部特征
           Parameters:
           - feature_files (list): 特征文件路径列表
           - output_root (str): 输出根目录
           - key_cols (list): 用于合并的非特征列，默认 ["CaseID", "Region", "KLGrade"]

        当用户请求特征组合时：
        1. 判断用户需要哪种组合方式：按区域组合或不按区域组合
        2. 从用户输入中提取特征文件路径
        3. 根据用户请求确定参数
        4. 调用合适的特征组合工具
        5. 清晰报告结果，包括输出文件路径

        重要说明：
        - 特征组合支持两两、三元和四元组合
        - 结果会以 CSV 文件保存到指定目录
        - key_cols 用于确保不同特征文件按同一病例或区域正确对齐
        - 系统会从文件名自动识别 train/test split
        - 一次只能处理一个 split，即 train 或 test

        CRITICAL: 必须严格输出 JSON，结构如下：
        {{
        "tools": [
            {{
                "tool": "combine_radiomics_features_by_region",
                "arguments": {{
                    "feature_files": ["path/to/feature1.csv", "path/to/feature2.csv"],
                    "output_root": "comb_datasets",
                    "regions": 5,
                    "key_cols": ["CaseID", "Region", "KLGrade"]
                }},
                "message": "正在按区域组合 radiomics 特征。"
            }}
        ],
        "final_message": "已完成特征组合，结果已保存到输出目录。"
        }}
        
        CRITICAL 规则：
        1. "tools" 数组必须包含完成用户请求所需的全部工具
        2. 如果用户提供多个特征文件，必须全部放入 feature_files 列表
        3. 如果用户要求按区域组合，使用 combine_radiomics_features_by_region
        4. 如果用户要求不按区域划分组合，使用 combine_radiomics_features_all
        5. 每个工具调用都必须包含 "message" 字段，用中文说明该步骤做什么
        6. "final_message" 应在所有工具执行后用中文总结完成了什么
        7. IMPORTANT: 即使缺少部分参数，也要识别需要调用的工具；缺少路径时可使用 ["path/to/feature1.csv"] 这类占位值并在 final_message 中请用户提供真实路径。
        8. 不要跳过用户明确要求的工具
        9. CRITICAL: 使用描述性输出目录名，并使用 process_input 提供的真实当前时间戳。格式如 "comb_datasets_TIMESTAMP"，其中 TIMESTAMP 替换为实际时间戳。
        
        示例：
        
        示例 1：用户要求按区域组合特征
        响应应包含 combine_radiomics_features_by_region 工具，并填入合适参数。
        
        示例 2：用户要求不按区域划分地组合特征
        响应应包含 combine_radiomics_features_all 工具。
        
        示例 3：用户要求组合多个特征文件但没有说明组合方式
        响应可包含两种组合工具，并在 final_message 中询问用户希望按区域组合还是不按区域组合。
        
        示例 4：用户要求分别组合训练集和测试集特征
        响应应为每个数据集包含相应工具调用。
        
        核心原则：始终根据用户意图识别需要调用的工具。参数缺失时使用占位值并在 final_message 中请用户提供真实信息。不要仅因为参数缺失就返回空 "tools" 数组。始终使用描述性输出目录名。
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
    
    async def process(self, state: AgentState) -> AgentState:
        """Process feature combination task"""
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

        You are Feature Combination Agent. Help user combine radiomics features from different files.
        Determine which feature combination tools to use and what parameters to provide.
        
        CRITICAL: Identify if multiple tools need to be called.
        If user requests combining features for both training and test sets separately, you MUST call the appropriate combination tool multiple times with appropriate parameters.
        
        IMPORTANT: ALWAYS identify and include the tools that need to be called, even if some parameters are missing. Use placeholder values like ["path/to/feature1.csv"] for missing parameters. NEVER return an empty tools array just because parameters are missing.
        
        If feature file paths are provided, use them. If not provided, use placeholders and ask user to provide them in final_message.
        
        IMPORTANT: Use descriptive output directory names. You MUST use the actual current timestamp provided here, not example timestamps. Use format like "comb_datasets_TIMESTAMP" where TIMESTAMP is replaced with the actual timestamp: {current_time}.
        
        Respond in JSON format with tools array and final_message.
        """
        
        try:
            tool_manager = await get_tool_manager()
            
            llm_response = self.llm.invoke(self.prompt.format_messages(input=process_input))
            
            try:
                decision = parse_llm_json(llm_response.content)
            except (json.JSONDecodeError, ValueError):
                decision = {
                    "tools": [],
                    "final_message": llm_response.content
                }
            
            tool_results = []
            all_messages = []
            
            if stream_callback:
                stream_callback("\n📋 执行步骤:")
                for i, tool_call in enumerate(decision.get("tools", []), 1):
                    stream_callback(f"   {i}. {tool_call.get('tool', 'Unknown')}")
                stream_callback("\n🧠 推理过程:")
                stream_callback("   分析用户请求，确定需要使用的特征组合方法")
            
            if "tools" in decision and decision["tools"]:
                for i, tool_call in enumerate(decision["tools"], 1):
                    tool_name = tool_call.get("tool")
                    arguments = tool_call.get("arguments", {})
                    step_message = tool_call.get("message", f"Step {i}")
                    
                    if stream_callback:
                        stream_callback(f"\n📌 步骤{i}: 调用工具 {tool_name}")
                        stream_callback(f"   参数: {json.dumps(arguments, ensure_ascii=False)}")
                    
                    try:
                        tool_result = await tool_manager.call_tool(
                            tool_name,
                            arguments,
                            stream_callback=stream_callback,
                        )
                        
                        if stream_callback:
                            stream_callback(f"   ✅ 完成")
                        
                        result_message = f"\n📊 执行结果：\n{tool_result}"
                        all_messages.append(result_message)
                        tool_results.append(tool_result)
                    except Exception as tool_error:
                        error_msg = f"工具执行失败：{str(tool_error)}"
                        if stream_callback:
                            stream_callback(f"   ❌ {error_msg}")
                        all_messages.append(f"\n❌ 执行结果：\n{error_msg}")
                        tool_results.append(error_msg)
            
            final_message = decision.get("final_message", "")
            
            if all_messages:
                full_response = "\n\n" + "=" * 50 + "\n\n".join(all_messages) + "\n\n" + "=" * 50 + "\n\n" + final_message
            else:
                full_response = final_message
            
            return {
                **state,
                "output": full_response,
                "agent_name": "FEATURE_COMBINATION_AGENT",
                "tool_calls": [t.get("tool") for t in decision.get("tools", [])],
                "tool_results": tool_results
            }
            
        except Exception as e:
            error_msg = f"Error during feature combination: {str(e)}"
            if stream_callback:
                stream_callback(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                **state,
                "output": error_msg,
                "agent_name": "FEATURE_COMBINATION_AGENT"
            }
