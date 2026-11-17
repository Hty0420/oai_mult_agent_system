from unittest.mock import Mock, patch

import pytest

import app as ui


def test_gradio6_history_preserves_messages_without_system_memory():
    history = [
        {"role": "system", "content": "private summary"},
        {"role": "assistant", "content": "Welcome"},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Pending question"},
    ]
    with patch.object(ui, "GRADIO_MAJOR_VERSION", 6):
        displayed = ui._chatbot_display_history(ui._chatbot_history_pairs(history))
    assert displayed == [
        {"role": "assistant", "content": "Welcome"},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Pending question"},
    ]


def test_gradio5_history_keeps_legacy_pairs():
    pairs = [("Question", "Answer"), ("Pending question", "")]
    with patch.object(ui, "GRADIO_MAJOR_VERSION", 5):
        assert ui._chatbot_display_history(pairs) == pairs


@pytest.mark.parametrize("status", ["running", "completed"])
def test_session_history_passes_real_chatbot_postprocessing(status):
    db = Mock()
    db.load_chat_history.return_value = [{"role": "user", "content": "Question"}]
    db.get_latest_session_background_task.return_value = {
        "status": status,
        "query_text": "Question",
        "progress_text": "Working",
        "response_text": "Answer" if status == "completed" else "",
        "output_files": [],
    }
    db.list_session_output_files.return_value = []
    with patch.object(ui, "DatabaseManager", return_value=db):
        history, *_ = ui._session_runtime_view(1, "workspace", "session")
    serialized = ui.chatbot.postprocess(history)
    assert serialized is not None
    if ui.GRADIO_MAJOR_VERSION >= 6:
        assert history[0] == {"role": "user", "content": "Question"}
        assert history[1]["role"] == "assistant"
        assert ("Answer" if status == "completed" else "Working") in history[1]["content"]
