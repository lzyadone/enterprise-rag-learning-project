"""Small Ollama HTTP helpers used by local RAG experiments."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any
from urllib.request import Request, urlopen


EMBEDDING_CACHE_MAX_SIZE = 256
_embedding_cache: OrderedDict[tuple[str, str, str], tuple[float, ...]] = OrderedDict()
_embedding_cache_lock = threading.RLock()
_embedding_cache_hits = 0
_embedding_cache_misses = 0
_embedding_api_requests = 0
_embedding_cache_bypasses = 0


def post_json(url: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_input(text: str) -> str:
    return " ".join(text.replace("\r", "\n").split())


def embed_texts(texts: list[str], model: str, host: str) -> list[list[float]]:
    clean_texts = [normalize_input(text) for text in texts]
    response = post_json(f"{host.rstrip('/')}/api/embed", {"model": model, "input": clean_texts})
    embeddings = response.get("embeddings")
    if isinstance(embeddings, list) and len(embeddings) == len(clean_texts):
        return embeddings

    embeddings_out: list[list[float]] = []
    for text in clean_texts:
        response = post_json(f"{host.rstrip('/')}/api/embeddings", {"model": model, "prompt": text})
        embedding = response.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Ollama embedding response missing embedding")
        embeddings_out.append(embedding)
    return embeddings_out


def embed_query(
    query: str,
    model: str,
    host: str,
    *,
    use_cache: bool = True,
) -> list[float]:
    if not use_cache:
        global _embedding_api_requests, _embedding_cache_bypasses
        with _embedding_cache_lock:
            _embedding_api_requests += 1
            _embedding_cache_bypasses += 1
        return embed_texts([query], model=model, host=host)[0]
    return embed_queries([query], model=model, host=host)[0]


def embed_queries(queries: list[str], model: str, host: str) -> list[list[float]]:
    """Embed queries with bounded in-process caching and batched cache misses."""
    global _embedding_cache_hits, _embedding_cache_misses, _embedding_api_requests
    if not queries:
        return []

    normalized_host = host.rstrip("/")
    normalized_queries = [normalize_input(query) for query in queries]
    keys = [(model, normalized_host, query) for query in normalized_queries]
    missing_keys: list[tuple[str, str, str]] = []
    seen_missing: set[tuple[str, str, str]] = set()
    with _embedding_cache_lock:
        for key in keys:
            if key in _embedding_cache:
                _embedding_cache_hits += 1
                _embedding_cache.move_to_end(key)
            elif key in seen_missing:
                _embedding_cache_hits += 1
            else:
                seen_missing.add(key)
                missing_keys.append(key)
                _embedding_cache_misses += 1

    if missing_keys:
        vectors = embed_texts(
            [key[2] for key in missing_keys],
            model=model,
            host=normalized_host,
        )
        with _embedding_cache_lock:
            _embedding_api_requests += 1
            for key, vector in zip(missing_keys, vectors, strict=True):
                _embedding_cache[key] = tuple(float(value) for value in vector)
                _embedding_cache.move_to_end(key)
            while len(_embedding_cache) > EMBEDDING_CACHE_MAX_SIZE:
                _embedding_cache.popitem(last=False)

    with _embedding_cache_lock:
        return [list(_embedding_cache[key]) for key in keys]


def clear_embedding_cache() -> None:
    global _embedding_cache_hits, _embedding_cache_misses, _embedding_api_requests
    global _embedding_cache_bypasses
    with _embedding_cache_lock:
        _embedding_cache.clear()
        _embedding_cache_hits = 0
        _embedding_cache_misses = 0
        _embedding_api_requests = 0
        _embedding_cache_bypasses = 0


def embedding_cache_info() -> dict[str, int]:
    with _embedding_cache_lock:
        return {
            "size": len(_embedding_cache),
            "max_size": EMBEDDING_CACHE_MAX_SIZE,
            "hits": _embedding_cache_hits,
            "misses": _embedding_cache_misses,
            "api_requests": _embedding_api_requests,
            "bypasses": _embedding_cache_bypasses,
        }


def unload_embedding_model(model: str, host: str) -> None:
    """Release an Ollama embedding model after a memory-sensitive offline phase."""
    post_json(
        f"{host.rstrip('/')}/api/embed",
        {"model": model, "input": "", "keep_alive": 0},
    )


def generate(prompt: str, model: str, host: str, num_ctx: int = 8192, num_predict: int = 700) -> str:
    response = post_json(
        f"{host.rstrip('/')}/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "temperature": 0.2,
            },
        },
        timeout=240,
    )
    return str(response.get("response", "")).strip()
