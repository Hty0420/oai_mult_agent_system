from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict, Optional, Union
from contextlib import AsyncExitStack
import json
import asyncio
import inspect
import os
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv()
import uuid
from db_utils.db_utils import DatabaseManager, set_request_context, get_trace_id
from db_utils.trajectory import (
    assign_trajectory_partition,
    build_trajectory_event,
    current_agent_version,
    stable_json_hash,
)
from db_utils.error_categorizer import categorize_error
from db_utils.workspace_memory import render_workspace_memory_context
from llm_config import get_llm_config
import time

# Import multi-agent system
from agents.orchestrator import aprocess_query as multi_agent_process_query
from agents.tool_progress import build_tool_progress_callback


class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict


class OAI_ChatBot:
    def __init__(
        self,
        model=None,
        username="default_user",
        use_multi_agent=True,
        workspace_id=None,
        agent_version=None,
    ):
        llm_config = get_llm_config()
        if model is not None and model != llm_config.model:
            raise ValueError("请在 llm_config.json 中统一配置模型，避免单智能体与多智能体配置不一致。")
        # 初始化会话和客户端对
        self.sessions: List[ClientSession] = []  # 存储所有服务器会话
        self.exit_stack = AsyncExitStack()  # 用于管理异步资源的上下文栈
        self.available_tools: List[ToolDefinition] = []  # 可用工具列表
        self.tool_to_session: Dict[str, ClientSession] = {}  # 工具到会话的映射
        self.model = llm_config.model
        self.client = llm_config.create_client()
        
        # 添加对话历史属性
        self.chat_history = []
        self.username = username
        self.session_id = str(uuid.uuid4())  # 生成唯一会话ID
        self.db_manager = DatabaseManager()
        self.user_id = self.db_manager.get_or_create_user_id(username)
        if workspace_id:
            self.workspace_id = self.db_manager.ensure_workspace(
                self.user_id, workspace_id, workspace_id
            )
        else:
            self.workspace_id = self.db_manager.ensure_personal_workspace(
                self.user_id
            )
        self.agent_version = agent_version or current_agent_version()
        self.last_run_metadata: Dict[str, object] = {}
        
        # 是否使用多智能体系统
        self.use_multi_agent = use_multi_agent
        
        # 加载历史会话（如果存在）
        self._load_history()

    async def connect_to_server(
        self, server_name: str, server_config: dict, config_dir: Path = None
    ) -> None:
        """连接到单个MCP服务器"""
        try:
            # 创建服务器参数
            config_dir = config_dir or Path(__file__).resolve().parent
            configured_command = str(server_config["command"])
            command_name = os.path.basename(configured_command).lower()
            command = (
                sys.executable
                if command_name in {
                    "python", "python.exe", "python3", "python3.exe"
                }
                else configured_command
            )
            server_params = StdioServerParameters(
                command=command,
                args=server_config.get("args", []),
                env=server_config.get("env"),
                cwd=str(config_dir),
            )
            # 建立标准输入输出传输
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            # 创建并初始化会话
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)

            # 获取此会话的可用工具列表
            response = await session.list_tools()
            tools = response.tools
            print(f"\n已连接到 {server_name}，可用工具:", [t.name for t in tools])
            # 注册工具及其对应的会话
            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })

        except Exception as e:
            print(f"连接到 {server_name} 失败: {e}")

    async def connect_to_servers(self):
        """连接到所有配置的MCP服务器"""
        try:
            # 从配置文件加载服务器信息
            config_path = Path(__file__).resolve().parent / "server_config.json"
            with config_path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})

            # 连接到所有服务器
            for server_name, server_config in servers.items():
                await self.connect_to_server(
                    server_name, server_config, config_path.parent
                )
        except Exception as e:
            print(f"加载服务器配置时出错: {e}")
            raise

    def _load_history(self):
        history = self.db_manager.load_chat_history(self.user_id, self.session_id)
        self.chat_history = []
        for item in history:
            role = item["role"]
            content = item["content"] if item["content"] is not None else ""
            if role == "tool":
                continue  # 关键：重启后不要把 tool 消息喂给 OpenAI
            self.chat_history.append({"role": role, "content": content})

    def _load_workspace_memory_context(self) -> str:
        db = DatabaseManager()
        try:
            records = db.list_workspace_memories(
                self.user_id,
                self.workspace_id,
                active_only=True,
                limit=100,
            )
            return render_workspace_memory_context(records)
        finally:
            db.close()

    def _find_last_n_round_spans(self, n: int):
        """
        返回最近 n 轮对话的 [(start_idx, end_idx), ...]，每轮定义为：
        - start_idx：一轮中"最后一个 user"的索引；
        - end_idx：在该 user 之后、下一个 user 之前出现的"最后一个 assistant"的索引（期间允许有 tool/system）。
        若该 user 之后没有 assistant，则该轮不计入。
        """
        # 收集所有 user 的位置
        user_positions = [i for i, m in enumerate(self.chat_history) if m.get("role") == "user"]
        if not user_positions:
            return []

        spans = []
        for k, uidx in enumerate(user_positions):
            # 下一个 user 的索引作为边界（若没有，则到末尾）
            next_user_idx = user_positions[k + 1] if k + 1 < len(user_positions) else len(self.chat_history)
            # 在 (uidx, next_user_idx) 范围内找最后一个 assistant
            last_assistant_idx = None
            for j in range(next_user_idx - 1, uidx, -1):
                if self.chat_history[j].get("role") == "assistant":
                    last_assistant_idx = j
                    break
            if last_assistant_idx is not None:
                spans.append((uidx, last_assistant_idx))

        # 取最近 n 轮
        return spans[-n:]

    def _count_completed_rounds(self) -> int:
        """统计已完成的 (user -> assistant) 轮数，忽略 tool/system 消息。"""
        rounds = 0
        waiting_user = False
        for m in self.chat_history:
            r = m.get("role")
            if r == "user":
                waiting_user = True
            elif r == "assistant" and waiting_user:
                rounds += 1
                waiting_user = False
        return rounds

    def summarize_memory(self) -> str:
        """
        规则（新版）：
        - 只保留最近 2 轮完整原文（每轮 user→...→assistant，期间包含 tool/system）。
        - 其余更早的所有对话合并成一条 system 摘要。
        - 如果历史完成轮数 <= 2，则不做摘要。
        """
        completed = self._count_completed_rounds()
        if completed <= 2:
            return ""

        # 取最近 2 轮
        last_two_spans = self._find_last_n_round_spans(2)
        if not last_two_spans:
            return ""

        # 需要保留的两段原文（按顺序拼接）
        retained_msgs = []
        first_retained_start = last_two_spans[0][0]
        last_retained_end = last_two_spans[-1][1]
        for (s, e) in last_two_spans:
            retained_msgs.extend(self.chat_history[s:e + 1])

        # 需要被摘要的"更早对话"
        prior = self.chat_history[:first_retained_start]
        if not prior:
            return ""

        # 生成摘要（把提示里的文字改成"最近2轮"）
        summary_prompt = [
            {
                "role": "system",
                "content": r"""
                           你是一个专为骨关节炎MRI影像分析Agent服务的摘要助手。请将以下历史多轮对话内容进行结构化总结，生成JSON格式摘要，输出格式必须严格如下：
                           {
                             "任务描述": {
                               "1": "描述用户在该阶段的主要任务描述，如MRI分割、特征提取、KL分级预测、特征组合等",
                               "2": "...(可多条)"
                             },
                             "调用工具名称": {
                               "1": "列出被调用的工具名称，如umamba_segment_mri、extract_radiomics_features、koa_lasso_classification、combine_radiomics_features_all等",
                               "2": "...(可多条)"
                             },
                             "工具执行结果": {
                               "1": "对应工具返回的主要结果、数值、路径或日志摘要，如'分割文件已保存成功'或'LASSO分类完成'",
                               "2": "...(可多条)"
                             },
                             "待办事项": {
                               "1": "推测用户下一步可能要进行的操作，如'提取特征'、'做KL分级'等",
                               "2": "...(可多条)"
                             }
                           }
                           严格按照以下要求进行总结：
                           - "任务描述"部分，用一句话来总结当次任务的描述。
                           - "调用工具名称"部分，当次任务调用了什么工具，就加入什么工具。
                           - "工具执行结果"部分，保留工具返回的结果，能正确识别出返回结果中可能涉及的文件路径、工具结果描述等。
                           - "待办事项"部分，根据本次工具调用结果，推测用户最可能进行的下一步操作。
                           - 之前摘要所显示的内容（任务描述、调用工具名称、工具执行结果、用户可能的待办事项），必须保留，不要清除和修改。
                           - 确保输出为合法JSON，内容简洁、科研化、条理清晰。
                           """
            },
            {"role": "user", "content": json.dumps(prior, ensure_ascii=False)}
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=summary_prompt
            )
            summary_text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"⚠️ 摘要生成失败：{e}")
            return ""

        if summary_text:
            new_history = [{"role": "system", "content": f"摘要记忆：{summary_text}"}] + retained_msgs
            self.chat_history = new_history

            try:
                self.db_manager.replace_conversation_with_messages(
                    self.user_id, self.session_id, new_history
                )
            except Exception as e:
                print(f"⚠️ 把摘要历史写回数据库失败：{e}")

            # 记录会话摘要
            try:
                self.db_manager.save_conversation_summary(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    summary_content=summary_text,
                    summarized_message_count=len(prior)
                )
            except Exception as e:
                print(f"⚠️ 保存会话摘要记录失败：{e}")

            return summary_text
        return ""

    async def process_query(
        self,
        query,
        stream_callback=None,
        user_message_persisted: bool = False,
    ):
        """处理用户查询，支持多智能体系统和单智能体模式
        
        Args:
            query: 用户查询
            stream_callback: 流式输出回调函数，用于实时输出工具调用信息
        """
        
        # 如果启用了多智能体系统，使用多智能体处理
        if self.use_multi_agent:
            return await self._process_with_multi_agent(
                query,
                stream_callback,
                user_message_persisted=user_message_persisted,
            )
        else:
            # 否则使用原有的单智能体模式
            return await self._process_with_single_agent(query, stream_callback)
    
    async def _process_with_multi_agent(
        self,
        query,
        stream_callback=None,
        user_message_persisted: bool = False,
    ):
        """使用多智能体系统处理查询"""
        # 设置请求上下文，用于数据库日志记录
        set_request_context(
            self.user_id,
            self.session_id,
            workspace_id=self.workspace_id,
            agent_version=self.agent_version,
            data_partition=assign_trajectory_partition(
                stable_json_hash(str(query))
            ),
        )

        # 将聊天历史转换为 LangChain 消息格式
        langchain_history = []
        workspace_memory_context = self._load_workspace_memory_context()
        if workspace_memory_context:
            langchain_history.append(
                SystemMessage(content=workspace_memory_context)
            )
        history_for_request = list(self.chat_history)
        if user_message_persisted and history_for_request:
            latest = history_for_request[-1]
            if isinstance(query, str):
                current_query_text = query
            elif isinstance(query, dict):
                current_query_text = str(query.get("text", ""))
            else:
                current_query_text = str(query)
            if (
                latest.get("role") == "user"
                and str(latest.get("content", "")) == current_query_text
            ):
                history_for_request = history_for_request[:-1]
        for msg in history_for_request:
            if msg["role"] == "user":
                langchain_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_history.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                langchain_history.append(SystemMessage(content=msg["content"]))

        # Persist the submitted task before orchestration starts. This makes
        # the running request visible after a page reload and ensures the
        # conversation title is created even when execution takes a long time.
        if isinstance(query, str):
            persisted_query = query
        elif isinstance(query, dict):
            persisted_query = str(query.get("text", ""))
        else:
            persisted_query = str(query)
        if not user_message_persisted:
            self.chat_history.append({"role": "user", "content": persisted_query})
            self.db_manager.save_message(
                self.user_id, self.session_id, "user", persisted_query
            )
        
        # Keep orchestration and every MCP call on the persistent background
        # event loop. MCP ClientSession/AnyIO streams are loop-affine; moving
        # orchestration into a worker thread previously created and destroyed
        # a new loop per Agent node, leaving the shared transport closed.
        orchestration_result = multi_agent_process_query(
            query, langchain_history, stream_callback
        )
        if inspect.isawaitable(orchestration_result):
            result = await orchestration_result
        else:
            # Compatibility for injected synchronous test/dry-run handlers.
            result = orchestration_result
        self.last_run_metadata = {
            "trace_id": result.get("trace_id"),
            "task_id": result.get("task_id"),
            "task_fingerprint": result.get("task_fingerprint"),
            "outcome_summary": result.get("outcome_summary", {}),
            "selected_pipeline_id": (
                result.get("selected_pipeline", {}) or {}
            ).get("candidate_id"),
            "workspace_id": self.workspace_id,
            "agent_version": self.agent_version,
            "policy_id": result.get("policy_id"),
            "data_partition": result.get("data_partition"),
        }
        
        # 获取响应
        response_text = result.get("response", "")
        agent_name = result.get("agent_name", "UNKNOWN")
        
        # 输出信息
        agent_info = f"🤖 [多智能体系统] 使用 {agent_name} 处理您的请求"
        print(agent_info)
        if stream_callback:
            stream_callback(agent_info)
        
        print(f"🤖 :{response_text}")
        if stream_callback:
            stream_callback(f"🤖 :{response_text}")
        
        # 添加助手回复到对话历史
        self.chat_history.append({'role': 'assistant', 'content': response_text})
        # 保存到数据库
        self.db_manager.save_message(self.user_id, self.session_id, 'assistant', response_text)
        
        # 自动触发摘要
        self.summarize_memory()
        return result

    def record_latest_task_feedback(
        self, outcome_label: str, notes: str = ""
    ) -> dict:
        """Attach authenticated UI feedback to the latest run in this session."""
        if outcome_label not in {"correct", "partial", "incorrect"}:
            raise ValueError("outcome_label must be correct, partial, or incorrect")
        trace_id = str(self.last_run_metadata.get("trace_id") or "")
        if not trace_id:
            raise ValueError("No completed task is available for feedback")
        task_id = self.last_run_metadata.get("task_id")
        accepted = outcome_label == "correct"
        reviewer_id = f"user:{stable_json_hash(self.username)[:16]}"
        set_request_context(
            self.user_id,
            self.session_id,
            workspace_id=self.workspace_id,
            agent_version=self.agent_version,
            data_partition=(
                str(self.last_run_metadata.get("data_partition"))
                if self.last_run_metadata.get("data_partition")
                else assign_trajectory_partition(
                    str(
                        self.last_run_metadata.get("task_fingerprint")
                        or trace_id
                    )
                )
            ),
        )
        self.db_manager.save_task_feedback(
            trace_id=trace_id,
            task_id=task_id if isinstance(task_id, int) else None,
            feedback_source="human",
            outcome_label=outcome_label,
            accepted=accepted,
            correction_required=not accepted,
            confidence=None,
            expected_pipeline_id=str(
                self.last_run_metadata.get("selected_pipeline_id") or ""
            ) or None,
            error_types=[],
            reviewer_id=reviewer_id,
            notes=notes or None,
        )
        self.db_manager.save_trajectory_event(
            build_trajectory_event(
                event_type="human_feedback_recorded",
                trace_id=trace_id,
                task_id=task_id if isinstance(task_id, int) else None,
                actor_type="human",
                actor_name=reviewer_id,
                status="success",
                payload={
                    "outcome_label": outcome_label,
                    "accepted": accepted,
                    "notes_sha256": stable_json_hash(notes or ""),
                },
            )
        )
        return {
            "trace_id": trace_id,
            "outcome_label": outcome_label,
            "recorded": True,
        }
    
    async def _process_with_single_agent(self, query, stream_callback=None):
        """使用原有的单智能体模式处理查询（保留向后兼容性）"""
        # 设置请求上下文
        set_request_context(
            self.user_id,
            self.session_id,
            agent_name="SINGLE_AGENT",
            workspace_id=self.workspace_id,
            agent_version=self.agent_version,
            data_partition=assign_trajectory_partition(
                stable_json_hash(str(query))
            ),
        )

        # 添加用户查询到对话历史
        self.chat_history.append({'role': 'user', 'content': query})
        # 保存到数据库
        self.db_manager.save_message(self.user_id, self.session_id, 'user', query)
        request_messages = list(self.chat_history)
        workspace_memory_context = self._load_workspace_memory_context()
        if workspace_memory_context:
            request_messages.insert(
                0,
                {"role": "system", "content": workspace_memory_context},
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=request_messages,
            tools=self.available_tools,
        )
        choice = response.choices[0]
        process_query = True
        while process_query:
            # 非工具调用，直接输出回复
            if choice.finish_reason != 'tool_calls':
                final_content = choice.message.content
                print(f"🤖 :{final_content}")
                if stream_callback:
                    stream_callback(f"🤖 :{final_content}")
                # 添加助手回复到对话历史
                self.chat_history.append({'role': 'assistant', 'content': final_content})
                # 保存到数据库
                self.db_manager.save_message(self.user_id, self.session_id, 'assistant', final_content)
                # —— 自动触发摘要（严格规则）：只保留最近1轮 ——
                self.summarize_memory()
                return
            # 工具调用情况

            ### 1. 非推理模型
            msg = choice.message.model_dump()
            if msg.get("content") is None:
                msg["content"] = ""  # 关键：避免写库 NULL
            self.chat_history.append(msg)
            # print(response)
            # print(f"🤔 :{choice.message.content}")
            # self.db_manager.save_message(self.user_id, self.session_id, 'assistant', choice.message.model_dump())

            ### 2.推理模型
            # self.chat_history.append({'role': 'assistant', 'content': choice.message.reasoning})
            # print(f"🤔 :{choice.message.reasoning}")
            # self.db_manager.save_message(self.user_id, self.session_id, 'assistant',choice.message.reasoning)
            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                session = self.tool_to_session[tool_name]
                tool_info = f"🔧 我需要调用工具 {tool_name} 来分析数据，马上给你结果\n 工具参数: {tool_args}"
                print(tool_info)
                if stream_callback:
                    stream_callback(tool_info)

                tool_start = time.time()
                tool_status = "success"
                tool_error_msg = None
                try:
                    progress_callback = build_tool_progress_callback(
                        tool_name,
                        stream_callback,
                    )
                    if progress_callback is None:
                        result = await session.call_tool(
                            tool_name,
                            json.loads(tool_args),
                        )
                    else:
                        result = await session.call_tool(
                            tool_name,
                            json.loads(tool_args),
                            progress_callback=progress_callback,
                        )
                    tool_content = result.content[0].text
                except Exception as tool_exc:
                    tool_status = "error"
                    tool_error_msg = str(tool_exc)
                    tool_content = f"Tool execution failed: {tool_exc}"

                tool_exec_ms = int((time.time() - tool_start) * 1000)

                # 记录工具调用日志（单智能体模式之前缺失此记录）
                try:
                    trace_id = get_trace_id()
                    error_category = None
                    if tool_status == 'error' and tool_error_msg:
                        error_category = categorize_error(tool_error_msg, tool_name)
                    result_summary = tool_content[:500] if len(tool_content) > 500 else tool_content
                    parsed_args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                    self.db_manager.save_tool_call_log(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        agent_name="SINGLE_AGENT",
                        tool_name=tool_name,
                        tool_arguments=parsed_args,
                        execution_time_ms=tool_exec_ms,
                        status=tool_status,
                        error_message=tool_error_msg,
                        result_summary=result_summary,
                        trace_id=trace_id,
                        error_category=error_category,
                    )
                except Exception as log_error:
                    print(f"[单智能体] 记录工具调用日志失败: {log_error}")

                # 直接显示工具结果给用户
                tool_result = f"\n📊 工具名称：{tool_name}\n 工具id:{tool_call.id} \n 执行结果:\n {tool_content}"
                print(tool_result)
                if stream_callback:
                    stream_callback(tool_result)

                tool_result_message = f"工具{tool_name}的返回结果为:\n{tool_content}"
                self.chat_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result_message
                })
                # 保存工具结果到数据库
                self.db_manager.save_message(self.user_id, self.session_id, 'tool', tool_result_message)
            
            # 继续与模型对话，让模型决定是否需要调用更多工具
            response = self.client.chat.completions.create(
                model=self.model,
                messages=(
                    [
                        {
                            "role": "system",
                            "content": workspace_memory_context,
                        }
                    ]
                    if workspace_memory_context
                    else []
                )
                + self.chat_history,
                tools=self.available_tools,
            )
            choice = response.choices[0]
            print(f"🤖 Response: {choice.message.content}")
            if stream_callback:
                stream_callback(f"🤖 Response: {choice.message.content}")
            
            # 如果模型不再调用工具，则保存最终回复并退出循环
            if choice.finish_reason == 'stop':
                final_res = choice.message.content or ""
                # 添加助手最终回复到对话历史
                self.chat_history.append({'role': 'assistant', 'content': final_res})
                # 保存到数据库
                self.db_manager.save_message(self.user_id, self.session_id, 'assistant', final_res)
                # —— 自动触发摘要：只保留最近1轮 ——
                self.summarize_memory()
                process_query = False
            # 如果模型还需要调用工具，继续循环
            elif choice.finish_reason == 'tool_calls':
                # 保存模型的中间回复（如果有）到对话历史
                if choice.message.content:
                    self.chat_history.append({'role': 'assistant', 'content': choice.message.content})
                    self.db_manager.save_message(self.user_id, self.session_id, 'assistant', choice.message.content)

    def clear_chat_history(self):
        """清除对话历史"""
        self.chat_history = []
        print("🗑️ 对话历史已清除")

    async def chat_loop(self):
        """运行交互式聊天循环"""
        print("\nMCP聊天机器人已启动！")
        print("输入您的问题或输入'quit'退出，输入'clear'清除对话历史。")

        while True:
            try:
                query = input("\n问题: ").strip()
                if query.lower() == 'quit':
                    break
                elif query.lower() == 'clear':
                    self.clear_chat_history()
                    continue
                await self.process_query(query)
                print("\n")
            except Exception as e:
                print(f"\n错误: {str(e)}")

    async def cleanup(self):
        """使用AsyncExitStack干净地关闭所有资源"""
        self.db_manager.close()  # 关闭数据库连接
        await self.exit_stack.aclose()


async def main():
    chatbot = OAI_ChatBot(use_multi_agent=True)  # 启用多智能体系统
    try:
        # MCP客户端和会话不是使用"with"语句初始化的
        # 因此需要手动处理清理工作
        await chatbot.connect_to_servers()  # 连接到所有服务器
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()  # 确保资源被正确清理


if __name__ == "__main__":
    asyncio.run(main())
