"""DeepSeek OpenAI-compatible Chat Completions client.

The API key is read from DEEPSEEK_API_KEY. Never commit secrets to this project.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def chat_completion(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 900,
    timeout: int = 240,
    response_format: dict[str, str] | None = None,
    thinking: bool | None = None,
) -> str:
    key = api_key or get_deepseek_api_key()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Set it in your shell before using DeepSeek.")

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if response_format:
        payload["response_format"] = response_format
    if thinking is not None:
        payload["thinking"] = {"type": "enabled" if thinking else "disabled"}

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek API error {exc.code}: {body}") from exc

    choices = result.get("choices")
    if not choices:
        raise RuntimeError(f"DeepSeek response missing choices: {result}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"DeepSeek response missing message content: {result}")
    return content.strip()


def get_deepseek_api_key() -> str | None:
    key = os.getenv("DEEPSEEK_API_KEY")
    if key:
        return key
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as env_key:
            value, _ = winreg.QueryValueEx(env_key, "DEEPSEEK_API_KEY")
        return str(value).strip() or None
    except Exception:
        return None
