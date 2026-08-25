"""Minimal OpenAI-compatible Chat Completions client for temporary Web use."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


class OpenAICompatibleAPIError(RuntimeError):
    """A safe user-facing remote API failure without response-body leakage."""


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.2,
    max_tokens: int = 900,
    timeout: int = 240,
) -> str:
    endpoint = chat_completions_url(base_url)
    clean_model = model.strip()
    clean_key = api_key.strip()
    if not clean_model:
        raise ValueError("Remote API model is required")
    if not clean_key:
        raise ValueError("Remote API key is required")

    payload: dict[str, Any] = {
        "model": clean_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {clean_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise OpenAICompatibleAPIError(remote_http_error(exc.code)) from exc
    except URLError as exc:
        raise OpenAICompatibleAPIError("Remote API connection failed") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenAICompatibleAPIError("Remote API returned invalid JSON") from exc

    choices = result.get("choices") if isinstance(result, dict) else None
    if not isinstance(choices, list) or not choices:
        raise OpenAICompatibleAPIError("Remote API response is missing choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OpenAICompatibleAPIError("Remote API returned an empty answer")
    return content.strip()


def chat_completions_url(base_url: str) -> str:
    value = base_url.strip()
    if not value:
        raise ValueError("Remote API URL is required")

    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").casefold()
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("Remote API URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("Remote API URL must not contain credentials")
    if parsed.fragment or parsed.query:
        raise ValueError("Remote API URL must not contain a query or fragment")
    if parsed.scheme == "http" and hostname not in LOCAL_HOSTS:
        raise ValueError("Remote API URL must use HTTPS")

    path = parsed.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path = f"{path}/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def remote_http_error(status: int) -> str:
    details = {
        400: "request rejected; check the model and endpoint",
        401: "authentication failed",
        403: "access denied",
        404: "endpoint or model not found",
        408: "request timed out",
        409: "request conflict",
        422: "request format rejected",
        429: "rate limit reached",
    }
    detail = details.get(status, "provider request failed")
    return f"Remote API error {status}: {detail}"
