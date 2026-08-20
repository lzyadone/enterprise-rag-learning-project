"""Repair RAG answers with a stronger chat model while preserving evidence."""

from __future__ import annotations

import json
from typing import Any

from src.deepseek_client import DEFAULT_BASE_URL, DEFAULT_MODEL, chat_completion


def repair_answer_with_deepseek(
    query: str,
    context: str,
    draft_answer: str,
    audit: dict[str, Any],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    prompt = f"""请把下面的 RAG 草稿答案修复为可交付答案。

修复规则：
1. 只能使用“检索资料”里的信息。
2. 不要补充资料外事实。
3. 每个关键结论和要点都必须在句末标注来源编号，例如 [1]。
4. 来源编号只能来自检索资料中的编号。
5. 最后必须输出“来源：”，列出用到的编号、标题和 URL。
6. 如果检索资料不足，直接说“根据当前知识库资料不足以回答”。

输出格式必须是：
结论：
- ...

要点：
1. ...
2. ...

来源：
[1] 标题 - URL

用户问题：
{query}

检索资料：
{context}

草稿答案：
{draft_answer}

审计结果：
{json.dumps(audit, ensure_ascii=False, indent=2)}

请输出修复后的中文答案：
"""
    return chat_completion(
        [
            {"role": "system", "content": "你是严谨的 RAG 答案修复器，只能基于证据重写答案。"},
            {"role": "user", "content": prompt},
        ],
        model=model,
        base_url=base_url,
        temperature=0.0,
        max_tokens=1800,
    )
