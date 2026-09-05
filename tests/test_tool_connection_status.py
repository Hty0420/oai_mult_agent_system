"""Regression tests for slow MCP startup on Python 3.10 and newer."""

import sys
import unittest
from concurrent.futures import Future, TimeoutError as FutureTimeoutError, wait
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app as ui
from mesh_visualization import UIRefreshTracker


class ToolConnectionStatusTests(unittest.TestCase):
    def setUp(self):
        self.future = Future()
        self.bot = SimpleNamespace(
            user_id=1,
            username="test-user",
            workspace_id="test-workspace",
            session_id="test-session",
            available_tools=[],
            _load_history=Mock(),
        )
        self.db = Mock()
        self.db.get_or_create_user_id.return_value = 1
        self.db.ensure_personal_workspace.return_value = "test-workspace"
        self.db.get_user_default_session_id.return_value = "test-session"
        self.db.get_active_background_task.return_value = None
        self.db.session_belongs_to_workspace.return_value = True
        self.runtime = Mock(return_value=([], None, gr.update(), [], gr.update()))

        def submit(coroutine, loop):
            # Keep MCP subprocesses out of the test; completion is controlled by
            # a real concurrent Future, including its Python 3.10 exceptions.
            coroutine.close()
            return self.future

        for mocked in (
            patch.object(ui, "global_chatbot", None),
            patch.object(ui, "is_initialized", False),
            patch.object(ui, "current_user", "test-user"),
            patch.object(ui, "current_session_id", None),
            patch.object(ui, "current_workspace_id", None),
            patch.object(ui, "loop", None),
            patch.object(ui, "_ensure_execution_loop"),
            patch.object(ui, "OAI_ChatBot", return_value=self.bot),
            patch.object(ui, "DatabaseManager", return_value=self.db),
            patch.object(ui, "_workspace_choices", return_value=[]),
            patch.object(ui, "workspace_dashboard_markdown", return_value=""),
            patch.object(ui, "_session_selector_state", return_value=gr.update()),
            patch.object(ui, "_session_runtime_view", self.runtime),
            patch.object(ui, "gallery_refresh_tracker", UIRefreshTracker()),
            patch.object(ui.asyncio, "run_coroutine_threadsafe", side_effect=submit),
            patch.object(ui, "wait_for_futures", side_effect=lambda futures, timeout: wait(futures, timeout=0)),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_expired_ui_wait_is_connecting_then_poll_reports_success(self):
        # Reproduce the original exception on the actual interpreter.
        with self.assertRaises(FutureTimeoutError):
            self.future.result(timeout=0)
        result = ui.initialize_chatbot()
        self.assertEqual(len(result), 7)
        self.assertIn("正在后台连接", result[0])
        self.assertNotIn("失败", result[0])
        self.assertNotIn("TimeoutError", result[0])
        self.assertFalse(self.future.cancelled())
        self.assertIs(self.bot.tool_connection_future, self.future)
        self.bot.available_tools = [{"function": {"name": "one"}}, {"function": {"name": "two"}}]
        self.future.set_result(None)
        polled = ui.poll_current_session("test-session")
        self.assertEqual(len(polled), 8)
        self.assertEqual(polled[1], "工作台已就绪 · 工具已连接（2 个）")

    def test_fast_connection_is_ready_on_initial_load(self):
        self.bot.available_tools = [{}]
        self.future.set_result(None)
        self.assertIn("工具已连接（1 个）", ui.initialize_chatbot()[0])

    def test_reopening_workspace_uses_the_current_connection_outcome(self):
        ui.initialize_chatbot()
        self.assertIn("正在后台连接", ui.initialize_chatbot()[0])
        self.future.set_result(None)
        self.assertIn("工具已连接", ui.initialize_chatbot()[0])

    def test_actual_timeout_failure_is_not_confused_with_wait_expiration(self):
        self.future.set_exception(FutureTimeoutError())
        with patch("builtins.print"):
            result = ui.initialize_chatbot()
        self.assertIn("工具连接失败", result[0])
        self.assertIn("TimeoutError", result[0])
        self.assertNotIn("正在后台连接", result[0])

    def test_late_connection_failure_replaces_connecting_status(self):
        ui.initialize_chatbot()
        with patch("builtins.print"):
            self.future.set_exception(RuntimeError("MCP handshake failed"))
        result = ui.poll_current_session("test-session")
        self.assertIn("工具连接失败：MCP handshake failed", result[1])

    def test_connection_completion_updates_status_without_a_selected_session(self):
        ui.initialize_chatbot()
        self.future.set_result(None)
        self.assertIn("工具已连接", ui.poll_current_session(None)[1])

    def test_active_task_status_keeps_priority(self):
        ui.initialize_chatbot()
        self.future.set_result(None)
        self.db.get_active_background_task.return_value = {
            "status": "running", "session_id": "test-session", "progress_text": "",
        }
        self.assertIn("任务后台执行中", ui.poll_current_session("test-session")[1])

    def test_completed_task_status_is_preserved(self):
        ui.initialize_chatbot()
        self.future.set_result(None)
        self.runtime.return_value = ([], "后台任务已完成", gr.update(), [], gr.update())
        self.assertEqual(ui.poll_current_session("test-session")[1], "后台任务已完成")


if __name__ == "__main__":
    unittest.main()
