"""
Web Search Agent - Handles web search tasks using Zhipu AI's search API
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


class WebSearchAgent:
    """Agent that handles web search tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = config.web_search_agent.llm
        
        self.system_prompt = """你是一个专门负责网页搜索的 agent。
        你的职责是帮助用户构造搜索请求、调用 web_search 工具，并把搜索结果整理成清晰、有重点的答复。

        除非用户明确要求使用其他语言，否则所有 message、final_message、解释和总结都使用简体中文。
        但 JSON key、tool 名称、参数名、URL、域名、文件路径必须保持英文原样，不要翻译。
        工具调用协议必须严格保持英文标识，例如 "tools"、"arguments"、"search_query"、"count"。

        可用搜索工具：
        1. web_search - 使用 Zhipu AI 的搜索 API 执行网页搜索
           Parameters:
           - search_query (str): 搜索查询字符串
           - count (int): 返回结果数量，范围 1-50，默认 10
           - search_domain_filter (str): 只访问指定域名内容，例如 'www.sohu.com'
           - search_recency_filter (str): 指定搜索时间范围，默认 'noLimit'
           - content_size (str): 控制网页摘要长度，默认 'medium'

        当用户请求网页搜索时：
        1. 从用户输入中提取搜索查询
        2. 判断是否需要额外参数，例如域名过滤、时间范围等
        3. 使用合适参数调用 web_search 工具
        4. 将搜索结果清晰总结给用户

        重要说明：
        - 始终提供清晰、简洁的搜索结果摘要
        - 如果返回多个结果，按结构化方式组织
        - 突出每条结果中最相关的信息
        - 如果没有找到结果，礼貌告知用户

        CRITICAL: 必须严格输出 JSON，结构如下：
        {{
        "tools": [
            {{
                "tool": "web_search",
                "arguments": {{
                    "search_query": "用户要搜索的查询词",
                    "count": 10,
                    "search_domain_filter": "www.example.com",
                    "search_recency_filter": "noLimit",
                    "content_size": "medium"
                }},
                "message": "正在为你执行网页搜索。"
            }}
        ],
        "final_message": "已完成网页搜索，请查看上方搜索结果摘要。"
        }}
        
        CRITICAL 规则：
        1. "tools" 数组必须包含 web_search 工具调用
        2. 必须始终包含 search_query 参数
        3. 用户未指定其他参数时，使用合适默认值
        4. "final_message" 应在工具执行后用中文总结搜索结果
        5. 不要返回空 "tools" 数组
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])
    
    async def process(self, state: AgentState) -> AgentState:
        """Process web search task"""
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

        You are Web Search Agent. Help user perform web search and provide relevant information.
        Determine the search query and any additional parameters needed.

        IMPORTANT: ALWAYS include the web_search tool call with appropriate parameters.
        Use default values for parameters not specified by the user.

        Respond in JSON format with tools array and final_message.
        """
        
        try:
            # Get tool manager with correct config path
            tool_manager = await get_tool_manager()
            # Re-initialize with correct path
            import os
            server_config_path = os.path.join(os.path.dirname(__file__), '..', 'server_config.json')
            await tool_manager.initialize(server_config_path=server_config_path)
            
            llm_response = self.llm.invoke(self.prompt.format_messages(input=process_input))
            
            print(f"[DEBUG] WebSearchAgent - LLM 原始响应: {llm_response.content[:500] if len(llm_response.content) > 500 else llm_response.content}")
            
            try:
                decision = parse_llm_json(llm_response.content)
                print(f"[DEBUG] WebSearchAgent - 解析后的 decision: {json.dumps(decision, ensure_ascii=False, indent=2)}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[DEBUG] WebSearchAgent - JSON 解析失败: {e}")
                decision = {
                    "tools": [],
                    "final_message": llm_response.content
                }
            
            tool_results = []
            all_messages = []
            
            if stream_callback:
                stream_callback("\n执行步骤:")
                for i, tool_call in enumerate(decision.get("tools", []), 1):
                    stream_callback(f"   {i}. {tool_call.get('tool', 'Unknown')}")
                stream_callback("\n推理过程:")
                stream_callback("   分析用户请求，确定搜索查询和参数")
            
            print(f"[DEBUG] WebSearchAgent - decision.get('tools'): {decision.get('tools', [])}")
            
            if "tools" in decision and decision["tools"]:
                print(f"[DEBUG] WebSearchAgent - 进入工具循环，共 {len(decision['tools'])} 个工具")
                for i, tool_call in enumerate(decision["tools"], 1):
                    tool_name = tool_call.get("tool")
                    print(f"[DEBUG] WebSearchAgent - 处理工具 {i}: {tool_name}")
                    arguments = tool_call.get("arguments", {})
                    step_message = tool_call.get("message", f"Step {i}")
                    
                    if stream_callback:
                        stream_callback(f"\n步骤{i}: 调用工具 {tool_name}")
                        stream_callback(f"   参数: {json.dumps(arguments, ensure_ascii=False)}")
                    
                    try:
                        tool_result = await tool_manager.call_tool(
                            tool_name,
                            arguments,
                            stream_callback=stream_callback,
                        )
                        
                        # 处理搜索结果，将结果传给大模型进行总结
                        if tool_name == "web_search":
                            print(f"[DEBUG] 进入 web_search 处理分支")
                            print(f"[DEBUG] tool_result 类型: {type(tool_result)}")
                            # 将搜索结果传给大模型进行智能总结
                            if isinstance(tool_result, dict):
                                print(f"[DEBUG] tool_result 是字典")
                                # 检查搜索结果是否成功
                                if tool_result.get("status") == "success":
                                    print(f"[DEBUG] 搜索结果成功，共 {tool_result.get('total', 0)} 条")
                                    # 构建搜索结果文本
                                    results_text = ""
                                    for i, result in enumerate(tool_result.get('results', []), 1):
                                        title = result.get('title', '')
                                        url = result.get('url', '')
                                        summary = result.get('summary', '')
                                        results_text += f"{i}. 标题: {title}\n"
                                        if url:
                                            results_text += f"   链接: {url}\n"
                                        if summary:
                                            results_text += f"   摘要: {summary}\n"
                                        results_text += "\n"
                                    
                                    summary_prompt = f"""请对以下网页搜索结果进行总结，挑重点告知给用户：

搜索结果内容：
{results_text}

请用中文简要总结：
1. 搜索结果的主要内容和关键点
2. 相关信息的重要摘要
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
                                        summary = "搜索结果解析失败，请稍后重试"
                                else:
                                    print(f"[DEBUG] 搜索结果失败: {tool_result.get('message', '未知错误')}")
                                    summary = tool_result.get('message', '搜索失败，请稍后重试')
                            elif isinstance(tool_result, str):
                                print(f"[DEBUG] tool_result 是字符串，长度: {len(tool_result)}")
                                print(f"[DEBUG] tool_result 前500字符: {tool_result[:500]}")
                                summary_prompt = f"""请对以下网页搜索结果进行总结，挑重点告知给用户：

搜索结果内容：
{tool_result}

请用中文简要总结：
1. 搜索结果的主要内容和关键点
2. 相关信息的重要摘要
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
                            
                            result_message = f"\n搜索结果总结：\n{summary}"
                        else:
                            result_message = f"\n执行结果：\n{tool_result}"
                        
                        if stream_callback:
                            stream_callback(f"   完成")
                        
                        all_messages.append(result_message)
                        tool_results.append(tool_result)
                    except Exception as tool_error:
                        error_msg = f"工具执行失败：{str(tool_error)}"
                        if stream_callback:
                            stream_callback(f"   {error_msg}")
                        all_messages.append(f"\n执行结果：\n{error_msg}")
                        tool_results.append(error_msg)
            
            final_message = decision.get("final_message", "")
            
            if all_messages:
                full_response = "\n\n" + "=" * 50 + "\n\n".join(all_messages) + "\n\n" + "=" * 50 + "\n\n" + final_message
            else:
                full_response = final_message
            
            return {
                **state,
                "output": full_response,
                "agent_name": "WEB_SEARCH_AGENT",
                "tool_calls": [t.get("tool") for t in decision.get("tools", [])],
                "tool_results": tool_results
            }
            
        except Exception as e:
            error_msg = f"Error during web search: {str(e)}"
            if stream_callback:
                stream_callback(f"\n{error_msg}")
            import traceback
            traceback.print_exc()
            
            return {
                **state,
                "output": error_msg,
                "agent_name": "WEB_SEARCH_AGENT"
            }
