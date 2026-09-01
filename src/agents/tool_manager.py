"""
MCP Tool Manager - Manages MCP tool connections and execution
"""

import json
import asyncio
import os
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from db_utils.db_utils import get_request_context, get_trace_id, DatabaseManager
from db_utils.trajectory import build_trajectory_event, stable_json_hash
from db_utils.experiment_etl import (
    CLASSIFICATION_TOOLS,
    ingest_classification_result,
    parse_classification_tool_result,
)
from db_utils.error_categorizer import categorize_error
from .tool_policy import ensure_tool_allowed
from .tool_progress import build_tool_progress_callback


class MCPToolManager:
    """Manages MCP tool connections and execution"""
    
    def __init__(self):
        self.sessions: List[ClientSession] = []
        self.exit_stack = None
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.available_tools: List[Dict[str, Any]] = []
        self._initialized = False
        self._init_error = None
        self._owner_loop_ref = None
        self._initialize_lock = None
        self._reconnect_lock = None
        self._connection_task = None
        self._shutdown_event = None
        self._ready_event = None
        self._server_config_path = str(
            Path(__file__).resolve().parents[1] / "server_config.json"
        )

    def _bind_to_running_loop(self):
        loop = asyncio.get_running_loop()
        owner = self._owner_loop_ref() if self._owner_loop_ref else None
        if owner is None:
            self._owner_loop_ref = weakref.ref(loop)
            self._initialize_lock = asyncio.Lock()
            self._reconnect_lock = asyncio.Lock()
        elif owner is not loop:
            raise RuntimeError(
                "MCP Tool Manager cannot be shared across event loops"
            )
        return loop
    
    async def initialize(self, server_config_path: str = None):
        """Initialize connections to all MCP servers"""
        self._bind_to_running_loop()
        config_path = Path(
            server_config_path or self._server_config_path
        ).resolve()
        self._server_config_path = str(config_path)
        async with self._initialize_lock:
            if (
                self._initialized
                and self._connection_task is not None
                and not self._connection_task.done()
            ):
                return

            if self._connection_task is not None:
                await self._close_connections()

            self._init_error = None
            self._ready_event = asyncio.Event()
            self._shutdown_event = asyncio.Event()
            self._connection_task = asyncio.create_task(
                self._connection_lifecycle(config_path),
                name="mcp-connection-lifecycle",
            )
            await self._ready_event.wait()
            if not self._initialized:
                init_error = self._init_error
                await self._close_connections()
                raise RuntimeError(
                    "MCP initialization failed: "
                    f"{init_error or 'unknown connection error'}"
                )

    async def _connection_lifecycle(self, config_path: Path):
        """Own every MCP context from enter through exit in one task."""
        try:
            async with AsyncExitStack() as stack:
                self.exit_stack = stack
                with config_path.open("r", encoding="utf-8") as file:
                    data = json.load(file)

                servers = data.get("mcpServers", {})
                if not servers:
                    raise RuntimeError("No MCP servers are configured")

                for server_name, server_config in servers.items():
                    await self._connect_to_server(
                        server_name,
                        server_config,
                        config_path.parent,
                    )

                if not self.sessions:
                    raise RuntimeError("Unable to connect to any MCP server")

                self._initialized = True
                self._init_error = None
                print(
                    f"[MCP Tool Manager] Initialized with "
                    f"{len(self.sessions)} servers"
                )
                print(
                    "[MCP Tool Manager] Available tools: "
                    f"{[t['function']['name'] for t in self.available_tools]}"
                )
                self._ready_event.set()
                await self._shutdown_event.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._init_error = str(exc) or type(exc).__name__
            print(f"[MCP Tool Manager] Initialization failed: {exc}")
        finally:
            self._initialized = False
            self.exit_stack = None
            if self._ready_event is not None:
                self._ready_event.set()
    
    async def _connect_to_server(
        self,
        server_name: str,
        server_config: Dict[str, Any],
        config_dir: Path,
    ):
        """Connect to a single MCP server"""
        try:
            print(f"[MCP Tool Manager] Connecting to server: {server_name}")

            configured_command = str(server_config["command"])
            command_name = os.path.basename(configured_command).lower()
            command = (
                sys.executable
                if command_name in {"python", "python.exe", "python3", "python3.exe"}
                else configured_command
            )
            server_params = StdioServerParameters(
                command=command,
                args=server_config["args"],
                env=None,
                cwd=str(config_dir),
            )
            
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            
            await session.initialize()
            
            # List available tools from this server
            tools_response = await session.list_tools()
            
            for tool in tools_response.tools:
                tool_name = tool.name
                self.tool_to_session[tool_name] = session
                
                # Convert to OpenAI function format
                tool_dict = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                self.available_tools.append(tool_dict)
                print(f"[MCP Tool Manager] Registered tool: {tool_name}")
            
            self.sessions.append(session)
            print(f"[MCP Tool Manager] Connected to {server_name} successfully")
            
        except Exception as e:
            print(f"[MCP Tool Manager] Failed to connect to {server_name}: {e}")
            # Don't raise error for individual server failures, continue with other servers

    @staticmethod
    def _is_transport_error(error: Exception) -> bool:
        transport_error_names = {
            "BrokenResourceError",
            "ClosedResourceError",
            "EndOfStream",
            "IncompleteReadError",
        }
        current = error
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if type(current).__name__ in transport_error_names:
                return True
            message = str(current).lower()
            if any(
                marker in message
                for marker in (
                    "closed resource",
                    "broken resource",
                    "end of stream",
                    "connection closed",
                    "session is closed",
                )
            ):
                return True
            current = current.__cause__ or current.__context__
        return False

    async def _close_connections(self):
        connection_task = self._connection_task
        if connection_task is not None and not connection_task.done():
            if self._shutdown_event is not None:
                self._shutdown_event.set()
            try:
                await connection_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(f"[MCP Tool Manager] Connection cleanup warning: {exc}")
        elif connection_task is not None:
            try:
                connection_task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(f"[MCP Tool Manager] Connection cleanup warning: {exc}")

        self._connection_task = None
        self._shutdown_event = None
        self._ready_event = None
        self.exit_stack = None
        self.sessions.clear()
        self.tool_to_session.clear()
        self.available_tools.clear()
        self._initialized = False

    async def _reconnect(self):
        self._bind_to_running_loop()
        async with self._reconnect_lock:
            print("[MCP Tool Manager] Transport closed; reconnecting once")
            await self._close_connections()
            self._init_error = None
            await self.initialize(self._server_config_path)

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools"""
        return self.available_tools
    
    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get description of a specific tool"""
        for tool in self.available_tools:
            if tool["function"]["name"] == tool_name:
                return tool["function"]["description"]
        return None
    
    def get_tool_parameters(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get parameters schema of a specific tool"""
        for tool in self.available_tools:
            if tool["function"]["name"] == tool_name:
                return tool["function"]["parameters"]
        return None
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        progress_callback=None,
        stream_callback=None,
    ) -> Any:
        """Call a specific tool with arguments"""
        if hasattr(self, "_bind_to_running_loop"):
            self._bind_to_running_loop()
        start_time = time.time()
        status = "success"
        error_message = None
        result_text = ""

        try:
            agent_name = get_request_context().get("agent_name")
            ensure_tool_allowed(agent_name, tool_name)
            if not self._initialized:
                if self._init_error:
                    raise RuntimeError(
                        f"MCP Tool Manager not initialized. Initialization error: {self._init_error}"
                    )
                raise RuntimeError("MCP Tool Manager not initialized. Call initialize() first.")

            if tool_name not in self.tool_to_session:
                available_tools = list(self.tool_to_session.keys())
                raise ValueError(
                    f"Tool {tool_name} not found. Available tools: {available_tools}"
                )

            session = self.tool_to_session[tool_name]

            if progress_callback is None and stream_callback is not None:
                progress_callback = build_tool_progress_callback(
                    tool_name,
                    stream_callback,
                )

            print(f"[MCP Tool Manager] Calling tool: {tool_name}")
            print(f"[MCP Tool Manager] Arguments: {arguments}")

            try:
                try:
                    if progress_callback is None:
                        result = await session.call_tool(tool_name, arguments)
                    else:
                        result = await session.call_tool(
                            tool_name,
                            arguments,
                            progress_callback=progress_callback,
                        )
                except Exception as first_error:
                    if (
                        not hasattr(self, "_is_transport_error")
                        or not self._is_transport_error(first_error)
                    ):
                        raise
                    await self._reconnect()
                    if tool_name not in self.tool_to_session:
                        raise RuntimeError(
                            f"Tool {tool_name} is unavailable after MCP reconnect"
                        ) from first_error
                    if progress_callback is None:
                        result = await self.tool_to_session[tool_name].call_tool(
                            tool_name, arguments
                        )
                    else:
                        result = await self.tool_to_session[tool_name].call_tool(
                            tool_name,
                            arguments,
                            progress_callback=progress_callback,
                        )

                structured_content = getattr(result, "structuredContent", None)
                if structured_content is not None:
                    result_text = json.dumps(structured_content, ensure_ascii=False)
                elif result.content:
                    content = result.content[0]
                    if hasattr(content, 'text'):
                        result_text = content.text
                    else:
                        result_text = str(content)
                else:
                    result_text = "Tool executed successfully (no output)"

                return result_text

            except Exception as e:
                status = "error"
                error_message = str(e)
                result_text = f"Tool execution failed: {e}"
                print(f"[MCP Tool Manager] {result_text}")
                import traceback
                traceback.print_exc()
                return result_text

        except (RuntimeError, ValueError) as e:
            status = "error"
            error_message = str(e)
            result_text = f"Tool execution failed: {e}"
            print(f"[MCP Tool Manager] {result_text}")
            return result_text

        finally:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._log_tool_call(
                tool_name, arguments, execution_time_ms,
                status, error_message, result_text
            )

    def _log_tool_call(self, tool_name: str, arguments: dict, execution_time_ms: int,
                       status: str, error_message: str, result_text: str):
        """记录工具调用到数据库"""
        db = None
        try:
            ctx = get_request_context()
            user_id = ctx.get('user_id')
            session_id = ctx.get('session_id')
            agent_name = ctx.get('agent_name', 'UNKNOWN')
            trace_id = ctx.get('trace_id') or get_trace_id()

            if tool_name in CLASSIFICATION_TOOLS and status == "success":
                structured_result = parse_classification_tool_result(result_text)
                application_status = (
                    str(structured_result.get("status", "")).strip().lower()
                    if structured_result is not None
                    else ""
                )
                if application_status != "success":
                    status = "error"
                    if structured_result is None:
                        error_message = "Classification tool returned an unstructured result"
                    else:
                        error_message = str(
                            structured_result.get("message")
                            or "Classification tool reported failure"
                        )

            error_category = None
            if status == 'error' and error_message:
                error_category = categorize_error(error_message, tool_name)

            should_auto_etl = tool_name in CLASSIFICATION_TOOLS and status == "success"
            if (user_id and session_id) or should_auto_etl:
                db = DatabaseManager()

            if user_id and session_id:
                result_summary = result_text[:500] if len(result_text) > 500 else result_text
                db.save_tool_call_log(
                    user_id=user_id,
                    session_id=session_id,
                    agent_name=agent_name,
                    tool_name=tool_name,
                    tool_arguments=arguments,
                    execution_time_ms=execution_time_ms,
                    status=status,
                    error_message=error_message,
                    result_summary=result_summary,
                    trace_id=trace_id,
                    error_category=error_category,
                )
                db.save_trajectory_event(
                    build_trajectory_event(
                        event_type="tool_completed",
                        trace_id=trace_id,
                        actor_type="tool",
                        actor_name=tool_name,
                        status="success" if status == "success" else "failed",
                        duration_ms=execution_time_ms,
                        payload={
                            "agent_name": agent_name,
                            "arguments": arguments,
                            "result_sha256": stable_json_hash(result_text or ""),
                            "result_length": len(result_text or ""),
                            "error_category": error_category,
                        },
                    )
                )

            if should_auto_etl and db is not None:
                try:
                    etl_result = ingest_classification_result(
                        tool_name,
                        result_text,
                        db=db,
                        expected_output_dir=arguments.get("output_dir"),
                        trace_id=trace_id,
                        session_id=session_id,
                    )
                    etl_status = etl_result.get("status", "failed")
                    db.save_trajectory_event(
                        build_trajectory_event(
                            event_type="evaluation_ingestion_completed",
                            trace_id=trace_id,
                            actor_type="evaluator",
                            actor_name="experiment_etl",
                            status=(
                                "success"
                                if etl_status == "success"
                                else "skipped"
                                if etl_status in {"skipped", "not_applicable"}
                                else "failed"
                            ),
                            payload={
                                "source_tool": tool_name,
                                "reason": etl_result.get("reason"),
                                "loaded": etl_result.get("loaded", 0),
                                "skipped": etl_result.get("skipped", 0),
                                "artifacts_registered": etl_result.get(
                                    "artifacts_registered", 0
                                ),
                                "evaluations_loaded": etl_result.get(
                                    "evaluations_loaded", 0
                                ),
                            },
                        )
                    )
                    if etl_status != "success":
                        print(
                            "[MCP Tool Manager] Automatic classification ETL "
                            f"did not load records: {etl_result.get('reason')}"
                        )
                except Exception as etl_error:
                    # Classification results remain available even if database
                    # ingestion has an incidental failure.  Record the failure
                    # separately so it can be retried or alerted on.
                    try:
                        db.save_trajectory_event(
                            build_trajectory_event(
                                event_type="evaluation_ingestion_completed",
                                trace_id=trace_id,
                                actor_type="evaluator",
                                actor_name="experiment_etl",
                                status="failed",
                                payload={
                                    "source_tool": tool_name,
                                    "reason": "etl_exception",
                                    "error_sha256": stable_json_hash(str(etl_error)),
                                },
                            )
                        )
                    except Exception:
                        pass
                    print(f"[MCP Tool Manager] Automatic classification ETL failed: {etl_error}")
        except Exception as e:
            print(f"[MCP Tool Manager] 记录工具调用日志失败: {e}")
        finally:
            if db is not None:
                try:
                    db.close()
                except Exception:
                    pass
    
    async def close(self):
        """Close all connections"""
        self._bind_to_running_loop()
        await self._close_connections()
        self._init_error = None
        print("[MCP Tool Manager] All connections closed")


# MCP transports are event-loop affine. Keep one manager per long-lived loop
# instead of a process-global singleton that can be reused from the wrong loop.
_tool_managers = weakref.WeakKeyDictionary()
_tool_managers_guard = threading.Lock()


async def get_tool_manager() -> MCPToolManager:
    """Get or create the MCP manager owned by the current event loop."""
    loop = asyncio.get_running_loop()
    with _tool_managers_guard:
        manager = _tool_managers.get(loop)
        if manager is None:
            manager = MCPToolManager()
            _tool_managers[loop] = manager
    await manager.initialize()
    return manager


async def reset_tool_manager():
    """Close and forget the MCP manager for the current event loop."""
    loop = asyncio.get_running_loop()
    with _tool_managers_guard:
        manager = _tool_managers.pop(loop, None)
    if manager is not None:
        await manager.close()
