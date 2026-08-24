"""Blind LLM relevance judge shared by audit and benchmark labeling."""

from __future__ import annotations

import json
from typing import Any

from src.deepseek_client import DEFAULT_MODEL, chat_completion
from src.relevance_audit import parse_llm_judgments


SYSTEM_PROMPT = """你是独立的信息检索相关性评测员。请判断每个候选知识块对给定问题的证据价值。

统一评分标准：
3 = 直接证据：知识块直接回答问题，或直接回答复合问题中的至少一个明确子问题。
2 = 辅助证据：包含形成答案所需的重要背景、方法或解释，但自身不是直接答案。
1 = 仅主题相关：主题相近，但缺少回答当前问题所需的具体证据。
0 = 无关或误导：不能帮助回答当前问题，或可能把回答引向错误方向。

必须阅读正文后判断，不得只根据标题、分类或专业术语评分。不要假设候选的排列顺序代表质量。
只返回 JSON 对象，格式为：
{"judgments":[{"chunk_id":"...","relevance":0,"reason":"不超过50个中文字符的依据"}]}
必须为每个输入 chunk_id 返回且只返回一次结果。"""


def judge_pool(
    pool: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    max_document_chars: int = 3200,
) -> list[dict[str, Any]]:
    """Judge one query's candidates without retrieval or prior-label signals."""
    candidates = []
    chunk_ids = []
    for candidate in pool["candidates"]:
        chunk_id = str(candidate["chunk_id"])
        chunk_ids.append(chunk_id)
        metadata = candidate.get("metadata") or {}
        document = str(candidate.get("document") or "")
        if len(document) > max_document_chars:
            document = document[:max_document_chars] + "\n[正文已截断]"
        candidates.append(
            {
                "chunk_id": chunk_id,
                "title": metadata.get("title") or "",
                "heading": metadata.get("heading_path") or "",
                "document": document,
            }
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"question": pool["question"], "candidates": candidates},
                ensure_ascii=False,
            ),
        },
    ]
    last_error: ValueError | None = None
    for _ in range(3):
        content = chat_completion(
            messages=messages,
            model=model,
            temperature=0.0,
            max_tokens=3600,
            timeout=300,
            response_format={"type": "json_object"},
            thinking=False,
        )
        try:
            return parse_llm_judgments(content, chunk_ids)
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"LLM relevance judge returned invalid JSON after 3 attempts: {last_error}")
