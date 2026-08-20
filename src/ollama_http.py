"""Small Ollama HTTP helpers used by local RAG experiments."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen


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


def embed_query(query: str, model: str, host: str) -> list[float]:
    return embed_texts([query], model=model, host=host)[0]


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
