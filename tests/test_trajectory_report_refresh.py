"""Exercise report polling through Gradio's actual State change detection."""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr
from gradio.state_holder import SessionState

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import app as ui
from mesh_visualization import UIRefreshTracker


class TrajectoryReportRefreshTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.report = self.root / "scientific_trajectory.md"
        self.report.write_text("# Scientific trajectory\n\nCompleted report.", encoding="utf-8")
        self.run = {
            "job_id": "run-a",
            "status": "completed",
            "output_files": [str(self.report)],
        }
        self.db = Mock()
        self.db.session_belongs_to_workspace.return_value = True
        self.db.get_latest_session_background_task.side_effect = lambda *args: self.run
        for mocked in (
            patch.dict(os.environ, {"SCIENTIFIC_TRAJECTORY_DIR": str(self.root)}),
            patch.object(ui, "DatabaseManager", return_value=self.db),
            patch.object(ui, "is_initialized", True),
            patch.object(ui, "global_chatbot", SimpleNamespace(user_id=1, workspace_id="workspace-a")),
            patch.object(ui, "trajectory_report_refresh_tracker", UIRefreshTracker()),
        ):
            mocked.start()
            self.addCleanup(mocked.stop)

        with gr.Blocks() as self.blocks:
            self.session = gr.Textbox(value="session-a")
            self.payload = gr.State(value={"session_id": "", "text": ""})
            self.markdown = gr.Markdown(visible=False)
            self.timer = gr.Timer(1)
            self.timer.tick(
                ui.refresh_scientific_trajectory_report,
                inputs=self.session,
                outputs=self.payload,
                queue=True,
                show_progress="hidden",
            )
            self.payload.change(
                ui.render_scientific_trajectory_report,
                inputs=[self.payload, self.session],
                outputs=self.markdown,
                queue=False,
                show_progress="hidden",
            )
        self.state = SessionState(self.blocks)
        self.request = gr.Request(session_hash="browser-a")
        self.event_loop = asyncio.new_event_loop()
        self.addCleanup(self.event_loop.close)

    def poll(self, session_id="session-a"):
        return self.event_loop.run_until_complete(self.blocks.process_api(
            0, [session_id], state=self.state, request=self.request,
        ))

    def render(self, session_id="session-a"):
        return self.event_loop.run_until_complete(self.blocks.process_api(
            1, [None, session_id], state=self.state, request=self.request,
        ))["data"][0]

    def test_unchanged_ticks_do_not_trigger_the_visible_report(self):
        first = self.poll()
        self.assertEqual(first["changed_state_ids"], [self.payload._id])
        self.assertTrue(self.render()["visible"])
        for _ in range(4):
            self.assertEqual(self.poll()["changed_state_ids"], [])

    def test_rewritten_report_updates_once(self):
        self.poll()
        self.report.write_text("# Updated report\n\nNew result.", encoding="utf-8")
        self.assertEqual(self.poll()["changed_state_ids"], [self.payload._id])
        self.assertIn("New result.", self.render()["value"])
        self.assertEqual(self.poll()["changed_state_ids"], [])

    def test_touching_identical_content_does_not_refresh_markdown(self):
        self.poll()
        stat = self.report.stat()
        os.utime(self.report, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertEqual(self.poll()["changed_state_ids"], [])

    def test_new_running_task_hides_old_report_once(self):
        self.poll()
        self.run = {"job_id": "run-b", "status": "running", "output_files": []}
        self.assertEqual(self.poll()["changed_state_ids"], [self.payload._id])
        self.assertFalse(self.render()["visible"])
        self.assertEqual(self.poll()["changed_state_ids"], [])
        self.run.update(status="completed", output_files=[str(self.report)])
        self.assertEqual(self.poll()["changed_state_ids"], [self.payload._id])
        self.assertTrue(self.render()["visible"])

    def test_session_switch_clears_report_and_rejects_a_stale_response(self):
        self.poll()
        self.assertEqual(self.render("session-b"), {"__type__": "update"})
        self.run = None
        self.assertEqual(self.poll("session-b")["changed_state_ids"], [self.payload._id])
        self.assertFalse(self.render("session-b")["visible"])
        self.assertEqual(self.poll("session-b")["changed_state_ids"], [])

    def test_lost_workspace_access_clears_report(self):
        self.poll()
        self.db.session_belongs_to_workspace.return_value = False
        self.assertEqual(self.poll()["changed_state_ids"], [self.payload._id])
        self.assertFalse(self.render()["visible"])
        self.assertEqual(self.poll()["changed_state_ids"], [])

    def test_another_browser_receives_its_own_initial_report(self):
        self.poll()
        self.state = SessionState(self.blocks)
        self.request = gr.Request(session_hash="browser-b")
        self.assertEqual(self.poll()["changed_state_ids"], [self.payload._id])
        self.assertTrue(self.render()["visible"])

    def test_production_timer_targets_state_instead_of_markdown(self):
        poll = next(f for f in ui.chat_app.fns.values() if f.fn is ui.refresh_scientific_trajectory_report)
        render = next(f for f in ui.chat_app.fns.values() if f.fn is ui.render_scientific_trajectory_report)
        self.assertEqual(len(poll.outputs), 1)
        self.assertIsInstance(poll.outputs[0], gr.State)
        # Gradio 5.x drops changed_state_ids from non-queued completion events;
        # backend-only process_api checks cannot catch that browser regression.
        self.assertTrue(poll.queue)
        self.assertIn((poll.outputs[0]._id, "change"), render.targets)
        self.assertTrue(all(isinstance(component, gr.Markdown) for component in render.outputs))
        for callback in ui.chat_app.fns.values():
            if any(event == "tick" for _, event in callback.targets):
                self.assertFalse(set(callback.outputs) & set(render.outputs))


if __name__ == "__main__":
    unittest.main()
