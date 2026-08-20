"""Answer faithfulness audit helpers for RAG responses."""

from __future__ import annotations

import json
import re
from typing import Any

from src.deepseek_client import DEFAULT_BASE_URL, chat_completion


DEFAULT_AUDIT_MODEL = "deepseek-chat"


def deterministic_audit(answer: str, source_count: int) -> dict[str, Any]:
    citation_ids = sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer) if int(match) <= source_count})
    out_of_range = sorted({int(match) for match in re.findall(r"\[(\d+)\]", answer) if int(match) > source_count})
    has_sources_section = bool(re.search(r"(?m)^\s*(来源|参考来源)\s*[:：]?", answer))
    insufficient_answer = "资料不足" in answer or "不足以回答" in answer

    issues: list[str] = []
    if not citation_ids and not insufficient_answer:
        issues.append("answer has no valid source citations")
    if out_of_range:
        issues.append(f"answer cites source ids outside retrieved range: {out_of_range}")
    if not has_sources_section and not insufficient_answer:
        issues.append("answer has no final source list")

    return {
        "valid_citation_ids": citation_ids,
        "out_of_range_citation_ids": out_of_range,
        "has_sources_section": has_sources_section,
        "insufficient_answer": insufficient_answer,
        "rule_pass": not issues,
        "rule_issues": issues,
    }


def audit_answer_with_deepseek(
    query: str,
    context: str,
    answer: str,
    model: str = DEFAULT_AUDIT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    prompt = build_audit_prompt(query, context, answer)
    raw = chat_completion(
        [
            {
                "role": "system",
                "content": "You are a strict RAG answer auditor. Return only a JSON object.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        base_url=base_url,
        temperature=0.0,
        max_tokens=2000,
        response_format=None,
    )
    if not raw.strip():
        raw = chat_completion(
            [
                {
                    "role": "system",
                    "content": "You are a strict RAG answer auditor. Return only a compact JSON object.",
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            base_url=base_url,
            temperature=0.0,
            max_tokens=3000,
            response_format=None,
        )
    parsed = parse_json_object(raw)
    parsed["raw_audit_response"] = raw
    return parsed


def build_audit_prompt(query: str, context: str, answer: str) -> str:
    return f"""请审计下面这个 RAG 回答是否忠实于检索资料。

你只能根据“检索资料”判断，不要使用资料外知识。

请返回 JSON 对象，字段必须包含：
- pass: boolean，答案是否可以接受
- faithfulness_score: 0 到 5，是否忠实于资料
- citation_score: 0 到 5，引用编号是否充分且正确
- relevance_score: 0 到 5，是否回答了用户问题
- unsupported_claims: string[]，答案中没有资料支持的判断
- missing_citations: string[]，应该引用但没有引用的关键判断
- answer_issues: string[]，其他问题
- suggested_fix: string，用中文说明如何改

用户问题：
{query}

检索资料：
{context}

待审计答案：
{answer}
"""


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {
            "pass": False,
            "faithfulness_score": 0,
            "citation_score": 0,
            "relevance_score": 0,
            "unsupported_claims": [],
            "missing_citations": [],
            "answer_issues": ["auditor did not return a JSON object"],
            "suggested_fix": "重新运行审计，或降低审计输出复杂度。",
        }
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Parsed audit JSON is not an object")
    return value


def combine_audits(rule_audit: dict[str, Any], llm_audit: dict[str, Any] | None) -> dict[str, Any]:
    if llm_audit is None:
        return {"rule_audit": rule_audit, "llm_audit": None, "overall_pass": rule_audit["rule_pass"]}

    llm_pass = bool(llm_audit.get("pass"))
    scores = [
        float(llm_audit.get("faithfulness_score", 0) or 0),
        float(llm_audit.get("citation_score", 0) or 0),
        float(llm_audit.get("relevance_score", 0) or 0),
    ]
    score_pass = all(score >= 4 for score in scores)
    return {
        "rule_audit": rule_audit,
        "llm_audit": llm_audit,
        "overall_pass": bool(rule_audit["rule_pass"] and llm_pass and score_pass),
    }
