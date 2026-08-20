"""Coverage audit for compound RAG questions.

Faithfulness tells us whether an answer is grounded. Coverage tells us whether
the answer addressed the important parts of the user's question.
"""

from __future__ import annotations

from typing import Any

from src.answer_audit import parse_json_object
from src.deepseek_client import DEFAULT_BASE_URL, chat_completion
from src.query_planning import QueryAspect


ASPECT_KEYWORDS: dict[str, list[str]] = {
    "classification": ["分类", "类别", "类型", "架构", "2-step", "agentic", "hybrid", "naive", "advanced"],
    "workflow": ["流程", "阶段", "接入", "切分", "索引", "检索", "生成", "评估"],
    "techniques": ["关键技术", "文档", "解析", "chunk", "切分", "embedding", "向量", "检索", "重排", "上下文", "评估"],
    "query_optimization": ["query rewrite", "query expansion", "改写", "扩展", "查询", "召回", "检索", "复杂", "模糊", "子问题"],
    "bottlenecks": ["瓶颈", "挑战", "难点", "召回", "噪声", "幻觉", "延迟", "成本", "更新", "权限", "引用"],
    "evaluation": ["评估", "指标", "相关性", "忠实", "正确性", "引用", "badcase", "judge"],
    "citation_quality": ["引用质量", "引用召回", "引用精度", "支撑", "可验证", "citation", "precision", "recall"],
    "badcase_analysis": ["badcase", "诊断", "检索错误", "生成错误", "上下文利用", "噪声", "幻觉", "failure", "error"],
}


def deterministic_coverage_audit(answer: str, aspects: list[QueryAspect]) -> dict[str, Any]:
    if not aspects:
        return {
            "pass": True,
            "coverage_score": 5,
            "covered_aspects": [],
            "missing_aspects": [],
            "thin_aspects": [],
            "suggested_fix": "单一问题无需多面覆盖审计。",
        }

    answer_lower = answer.lower()
    covered: list[str] = []
    thin: list[str] = []
    missing: list[str] = []
    for aspect in aspects:
        keywords = ASPECT_KEYWORDS.get(aspect.name, [aspect.name])
        hits = [keyword for keyword in keywords if keyword.lower() in answer_lower]
        if len(hits) >= 2:
            covered.append(aspect.name)
        elif hits:
            thin.append(aspect.name)
        else:
            missing.append(aspect.name)

    coverage_score = round(5 * len(covered) / max(1, len(aspects)), 1)
    return {
        "pass": not missing and coverage_score >= 4,
        "coverage_score": coverage_score,
        "covered_aspects": covered,
        "missing_aspects": missing,
        "thin_aspects": thin,
        "suggested_fix": build_suggested_fix(missing, thin),
    }


def audit_coverage_with_deepseek(
    query: str,
    context: str,
    answer: str,
    aspects: list[QueryAspect],
    model: str = "deepseek-chat",
    base_url: str = DEFAULT_BASE_URL,
) -> dict[str, Any]:
    prompt = build_coverage_prompt(query, context, answer, aspects)
    raw = chat_completion(
        [
            {
                "role": "system",
                "content": "You are a strict RAG completeness auditor. Return only a JSON object.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        base_url=base_url,
        temperature=0.0,
        max_tokens=1800,
        response_format=None,
    )
    parsed = parse_json_object(raw)
    parsed["raw_coverage_response"] = raw
    return parsed


def build_coverage_prompt(query: str, context: str, answer: str, aspects: list[QueryAspect]) -> str:
    aspect_lines = "\n".join(
        f"- {aspect.name}: {aspect.question}; categories={', '.join(aspect.categories)}" for aspect in aspects
    )
    return f"""请审计这个 RAG 回答是否完整覆盖了用户问题的各个子问题。

你只评估“覆盖度”和“资料支持是否足够”，不要引入资料外知识。

请返回 JSON 对象，字段必须包含：
- pass: boolean，复合问题是否覆盖完整
- coverage_score: 0 到 5，覆盖度评分
- covered_aspects: string[]，已覆盖的子问题名称
- missing_aspects: string[]，没有覆盖的子问题名称
- thin_aspects: string[]，提到了但证据或解释偏薄的子问题名称
- source_coverage_score: 0 到 5，检索资料是否足够支撑这些子问题
- suggested_fix: string，用中文说明应该补什么

用户问题：
{query}

本轮必须覆盖的子问题：
{aspect_lines}

检索资料：
{context}

待审计答案：
{answer}
"""


def build_suggested_fix(missing: list[str], thin: list[str]) -> str:
    parts: list[str] = []
    if missing:
        parts.append("补充未覆盖子问题：" + ", ".join(missing))
    if thin:
        parts.append("加强解释偏薄子问题：" + ", ".join(thin))
    return "；".join(parts) if parts else "覆盖度基本满足。"
