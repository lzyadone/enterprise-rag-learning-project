"""Lightweight local reranking for retrieved chunks.

This module is intentionally dependency-free. It gives the project a realistic
two-stage retrieval shape now, while leaving a clean seam for replacing this
heuristic with a cross-encoder or API reranker later.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


TERM_ALIASES: dict[str, list[str]] = {
    "rag": ["retrieval", "augmented", "generation", "retrieval-augmented"],
    "检索增强": ["rag", "retrieval", "augmented", "generation"],
    "切分": ["chunk", "chunks", "chunking", "split", "splitter", "splitting"],
    "分块": ["chunk", "chunks", "chunking", "split", "splitter", "splitting"],
    "固定窗口": ["fixed", "window", "chunk_size", "chunk size", "fixed chunk size"],
    "边界": ["boundary", "boundaries", "sentence", "paragraph", "section", "heading"],
    "重排": ["rerank", "reranker", "reranking", "ranker", "postprocessor"],
    "召回": ["retrieve", "retrieval", "retriever", "candidate", "candidates"],
    "向量": ["vector", "embedding", "embeddings", "dense"],
    "评估": ["evaluation", "evaluate", "faithfulness", "groundedness", "relevance"],
    "元数据": ["metadata", "filter", "where"],
    "过滤": ["filter", "metadata", "where"],
    "上下文": ["context", "window", "synthesis"],
}


def lexical_rerank(query: str, chunks: Sequence[Any], top_k: int) -> list[Any]:
    """Rerank candidates using retrieval score plus lexical/schema evidence."""
    if not chunks:
        return []

    query_tokens = expand_query_tokens(query)
    base_scores = [base_retrieval_score(item) for item in chunks]
    normalized_base_scores = min_max_normalize(base_scores)

    scored: list[tuple[float, Any]] = []
    for item, base_score in zip(chunks, normalized_base_scores):
        text_score = lexical_match_score(query_tokens, searchable_text(item))
        heading_score = lexical_match_score(query_tokens, heading_text(item))
        category_score = lexical_match_score(query_tokens, str(getattr(item, "metadata", {}).get("category", "")))
        final_score = round((0.50 * base_score) + (0.30 * text_score) + (0.15 * heading_score) + (0.05 * category_score), 6)

        setattr(item, "rerank_score", final_score)
        setattr(
            item,
            "rerank_reason",
            f"base={base_score:.3f}; lexical={text_score:.3f}; heading={heading_score:.3f}; category={category_score:.3f}",
        )
        item.score = final_score
        scored.append((final_score, item))

    return [item for _, item in sorted(scored, key=lambda pair: (-pair[0], pair[1].distance))[:top_k]]


def base_retrieval_score(item: Any) -> float:
    score = float(getattr(item, "score", 0.0) or 0.0)
    if score > 0:
        return score
    distance = max(0.0, float(getattr(item, "distance", 0.0) or 0.0))
    return 1.0 / (1.0 + distance)


def min_max_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def searchable_text(item: Any) -> str:
    metadata = getattr(item, "metadata", {}) or {}
    fields = [
        str(metadata.get("title", "")),
        str(metadata.get("category", "")),
        str(metadata.get("heading_path", "")),
        str(getattr(item, "document", "")),
    ]
    return "\n".join(fields).lower()


def heading_text(item: Any) -> str:
    metadata = getattr(item, "metadata", {}) or {}
    return " ".join(
        [
            str(metadata.get("title", "")),
            str(metadata.get("category", "")),
            str(metadata.get("heading_path", "")),
        ]
    ).lower()


def expand_query_tokens(query: str) -> set[str]:
    tokens = tokenize(query)
    normalized_query = query.lower()
    for key, aliases in TERM_ALIASES.items():
        if key.lower() in normalized_query:
            tokens.update(alias.lower() for alias in aliases)
    return {token for token in tokens if len(token) >= 2}


def tokenize(text: str) -> set[str]:
    text = text.lower()
    tokens = set(re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", text))

    for cjk_span in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(cjk_span) == 1:
            tokens.add(cjk_span)
            continue
        tokens.update(cjk_span[i : i + 2] for i in range(len(cjk_span) - 1))
        tokens.update(cjk_span[i : i + 3] for i in range(len(cjk_span) - 2))
    return tokens


def lexical_match_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens or not text:
        return 0.0
    normalized_text = text.lower()
    matched = sum(1 for token in query_tokens if token in normalized_text)
    return matched / len(query_tokens)
