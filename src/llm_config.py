"""Shared LLM configuration for the chatbot and every specialized agent.

Edit the project-root llm_config.json to switch models. Credentials stay in
the project-root .env (or process environment). Restart after configuration
changes so all clients use the same settings for the lifetime of the process.
"""

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "llm_config.json"
ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key_env: str
    api_key: str = field(repr=False)
    timeout: float = 600.0
    max_retries: int = 2

    def _client_kwargs(self) -> dict:
        return {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    def create_client(self) -> OpenAI:
        return OpenAI(**self._client_kwargs())

    def create_chat_model(self, *, temperature: float = 0.1) -> ChatOpenAI:
        return ChatOpenAI(
            model=self.model,
            temperature=temperature,
            **self._client_kwargs(),
        )


@lru_cache(maxsize=1)
def get_llm_config() -> LLMConfig:
    """Load one configuration snapshot; never fall back to another provider."""
    try:
        values = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"缺少 LLM 配置文件：{CONFIG_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"llm_config.json 格式错误：第 {exc.lineno} 行，第 {exc.colno} 列。"
        ) from exc

    if not isinstance(values, dict):
        raise ValueError("llm_config.json 必须是 JSON 对象。")
    required = {"model", "base_url", "api_key_env"}
    unknown = values.keys() - required - {"timeout", "max_retries"}
    if unknown:
        raise ValueError(f"llm_config.json 包含未知配置项：{', '.join(sorted(unknown))}")
    for name in sorted(required):
        value = values.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"llm_config.json 的 {name} 必须是非空字符串。")
        values[name] = value.strip()

    url = urlsplit(values["base_url"])
    if url.scheme not in {"http", "https"} or not url.hostname:
        raise ValueError("llm_config.json 的 base_url 必须是完整的 HTTP/HTTPS 接口地址。")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("base_url 不应包含凭据、查询参数或片段；密钥请通过 api_key_env 指定。")
    timeout = values.get("timeout", 600.0)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("llm_config.json 的 timeout 必须是大于 0 的秒数。")
    max_retries = values.get("max_retries", 2)
    if type(max_retries) is not int or max_retries < 0:
        raise ValueError("llm_config.json 的 max_retries 必须是非负整数。")

    load_dotenv(dotenv_path=ENV_PATH, override=False)
    api_key_env = values["api_key_env"]
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"未配置 LLM 密钥：请在项目根目录 .env 或进程环境中设置 {api_key_env}。"
        )

    return LLMConfig(
        model=values["model"],
        base_url=values["base_url"].rstrip("/"),
        api_key_env=api_key_env,
        api_key=api_key,
        timeout=float(timeout),
        max_retries=max_retries,
    )
