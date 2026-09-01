"""
Segmentation Agent - Handles MRI image segmentation tasks
"""

from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

from .state import AgentState
from .config import Config
from .tool_manager import get_tool_manager
from .llm_json import parse_llm_json


class SegmentationAgent:
    """Agent that handles MRI image segmentation tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = config.segmentation_agent.llm
        
        self.system_prompt = """你是一个专门负责骨关节炎（OA）MRI 分割的医学影像 agent。
        你的职责是帮助用户从膝关节 MRI 图像中分割关节结构。

        除非用户明确要求使用其他语言，否则所有 message、final_message、解释和总结都使用简体中文。
        但 JSON key、tool 名称、参数名、文件路径、模型名必须保持英文原样，不要翻译。

        可用分割工具：
        1. umamba_segment_mri - 将 MRI 文件提交给 U-Mamba 分割服务
           Parameters: file_paths (list), save_dir (str, optional, default "umamba_seg_results")
           
        2. lkm_unet_segment_mri - 将 MRI 文件提交给 LKM-UNet 分割服务
           Parameters: file_paths (list), save_dir (str, optional, default "lkm_unet_seg_results")
           
        3. lightm_unet_segment_mri - 将 MRI 文件提交给 LightM-UNet 分割服务
           Parameters: file_paths (list), save_dir (str, optional, default "lightm_unet_seg_results")

        当用户请求分割时：
        1. 判断用户希望使用哪个分割模型；如果未指定，默认使用 U-Mamba
        2. 从用户输入或上下文中提取真实文件路径
        3. 判断是否需要调用多个分割工具
        4. 按正确顺序调用工具
        5. 清晰报告执行结果，包括输出文件路径

        重要说明：
        - MRI 文件通常应为 .nii.gz 格式
        - 分割结果会保存到指定目录
        - 支持一次处理多个文件
        - 结果可能是单个文件，也可能是 zip 压缩包
        - 用户可能会要求多个模型分割并比较结果

        回答应专业、清晰，重点说明分割流程、使用的模型和结果位置。

        IMPORTANT: 必须严格输出 JSON，结构如下：
        {{
        "tools": [
            {{
                "tool": "tool_name",
                "arguments": {{
                    "file_paths": ["path1", "path2"],
                    "save_dir": "optional_directory"
                }},
                "message": "本步骤给用户看的中文说明"
            }},
            {{
                "tool": "another_tool_name",
                "arguments": {{
                    "file_paths": ["path1", "path2"],
                    "save_dir": "another_directory"
                }},
                "message": "本步骤给用户看的中文说明"
            }}
        ],
        "final_message": "给用户看的中文最终总结"
        }}
        
        "tools" 数组包含所有需要按顺序调用的工具。"final_message" 是所有工具执行后的中文总结。
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
        
        self.json_parser = JsonOutputParser()
    
    async def process(self, state: AgentState) -> AgentState:
        """Process segmentation task"""
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
        
        process_input = f"""
        User query: {input_text}

        Recent conversation context: {recent_context}

        You are Segmentation Agent. Help user with their MRI segmentation task.
        Determine which segmentation tools to use and what parameters to provide.
        
        If user has not provided file paths, ask them to provide MRI file paths.
        If file paths are provided, execute appropriate segmentation tools.
        
        IMPORTANT: Identify if multiple tools need to be called (e.g., segmenting with multiple models for comparison).
        Return all tools in "tools" array in the correct execution order.
        
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
                stream_callback("   分析用户请求，确定需要使用的分割模型")
            
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
                "agent_name": "SEGMENTATION_AGENT",
                "tool_calls": [t.get("tool") for t in decision.get("tools", [])],
                "tool_results": tool_results
            }
            
        except Exception as e:
            error_msg = f"Error during segmentation: {str(e)}"
            if stream_callback:
                stream_callback(f"\n❌ {error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                **state,
                "output": error_msg,
                "agent_name": "SEGMENTATION_AGENT"
            }
