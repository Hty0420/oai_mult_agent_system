"""Offline regression checks for the shared LLM configuration boundary."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import llm_config


@pytest.fixture
def configured_llm(tmp_path, monkeypatch):
    config_path = tmp_path / "llm_config.json"
    values = {
        "model": "selected-model",
        "base_url": "https://selected.invalid/v1",
        "api_key_env": "TEST_LLM_API_KEY",
        "timeout": 45,
        "max_retries": 0,
    }
    config_path.write_text(json.dumps(values), encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_LLM_API_KEY=file-test-key\n", encoding="utf-8")
    monkeypatch.setattr(llm_config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(llm_config, "ENV_PATH", env_path)
    # Register the original value with monkeypatch before dotenv mutates it.
    monkeypatch.setenv("TEST_LLM_API_KEY", "")
    monkeypatch.delenv("TEST_LLM_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "stale-openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://stale.invalid/v1")
    monkeypatch.setenv("MODEL_NAME", "stale-model")
    llm_config.get_llm_config.cache_clear()
    yield config_path, env_path, values
    llm_config.get_llm_config.cache_clear()


def test_all_clients_send_the_selected_model_endpoint_and_key(configured_llm, monkeypatch):
    from agents.config import Config
    import oai_client

    monkeypatch.setattr(oai_client, "DatabaseManager", Mock())
    monkeypatch.setattr(oai_client.OAI_ChatBot, "_load_history", lambda self: None)
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "test-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "selected-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
            },
        )

    monkeypatch.setattr(httpx.Client, "send", lambda self, request, **kw: respond(request))

    async def async_send(self, request, **kwargs):
        return respond(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", async_send)
    agents = [value.llm for value in vars(Config()).values() if hasattr(value, "llm")]
    assert len(agents) == 9
    for agent in agents:
        assert agent.invoke("ping").content == "ok"
        assert agent.request_timeout == 45
        assert agent.max_retries == 0
    assert asyncio.run(agents[0].ainvoke("ping")).content == "ok"
    assert sorted(agent.temperature for agent in agents) == [0.1] * 7 + [0.65, 0.7]

    for multi_agent in (False, True):
        chatbot = oai_client.OAI_ChatBot(use_multi_agent=multi_agent)
        with chatbot.client as client:
            client.chat.completions.create(
                model=chatbot.model,
                messages=[{"role": "user", "content": "ping"}],
            )
            assert client.timeout == 45
            assert client.max_retries == 0

    assert len(requests) == 12
    for request in requests:
        assert str(request.url) == "https://selected.invalid/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer file-test-key"
        assert json.loads(request.content)["model"] == "selected-model"

    with pytest.raises(ValueError, match="llm_config.json"):
        oai_client.OAI_ChatBot(model="conflicting-model")


def test_loading_ignores_working_directory(configured_llm, tmp_path, monkeypatch):
    launch_dir = tmp_path / "different-launch-directory"
    launch_dir.mkdir()
    (launch_dir / ".env").write_text("TEST_LLM_API_KEY=wrong-key\n", encoding="utf-8")
    (launch_dir / "llm_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(launch_dir)
    settings = llm_config.get_llm_config()
    assert settings.model == "selected-model"
    assert settings.api_key == "file-test-key"
    assert settings.api_key not in repr(settings)


def test_process_key_takes_precedence(configured_llm, monkeypatch):
    monkeypatch.setenv("TEST_LLM_API_KEY", "process-test-key")
    assert llm_config.get_llm_config().api_key == "process-test-key"


def test_missing_selected_key_does_not_use_openai_key(configured_llm):
    _, env_path, _ = configured_llm
    env_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="TEST_LLM_API_KEY"):
        llm_config.get_llm_config()


def test_configuration_is_consistent_until_restart(configured_llm):
    config_path, _, values = configured_llm
    first = llm_config.get_llm_config()
    values["model"] = "next-model"
    config_path.write_text(json.dumps(values), encoding="utf-8")
    assert llm_config.get_llm_config() is first
    llm_config.get_llm_config.cache_clear()
    assert llm_config.get_llm_config().model == "next-model"


@pytest.mark.parametrize("changes", [
    {"model": ""},
    {"base_url": "not-a-url"},
    {"api_key_env": ""},
    {"timeout": 0},
    {"timeout": True},
    {"max_retries": -1},
    {"api_key": "do-not-store-keys-here"},
])
def test_invalid_settings_fail_before_creating_clients(configured_llm, changes):
    config_path, _, values = configured_llm
    config_path.write_text(json.dumps({**values, **changes}), encoding="utf-8")
    with pytest.raises(ValueError):
        llm_config.get_llm_config()
