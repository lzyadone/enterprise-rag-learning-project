"""Independent relevance-label audit helpers."""

from __future__ import annotations

import json
from typing import Any


def parse_llm_judgments(content: str, expected_chunk_ids: list[str]) -> list[dict[str, Any]]:
    """Parse and strictly validate an LLM relevance-judgment response."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM audit response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("judgments"), list):
        raise ValueError("LLM audit response must contain a judgments array")

    expected = set(expected_chunk_ids)
    if len(expected) != len(expected_chunk_ids):
        raise ValueError("expected_chunk_ids contains duplicates")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for item in payload["judgments"]:
        if not isinstance(item, dict):
            raise ValueError("each LLM judgment must be an object")
        chunk_id = str(item.get("chunk_id") or "").strip()
        relevance = item.get("relevance")
        if chunk_id not in expected:
            raise ValueError(f"LLM audit returned unknown chunk_id: {chunk_id}")
        if chunk_id in seen:
            raise ValueError(f"LLM audit returned duplicate chunk_id: {chunk_id}")
        if isinstance(relevance, bool) or not isinstance(relevance, int) or relevance not in {0, 1, 2, 3}:
            raise ValueError(f"invalid LLM relevance for {chunk_id}: {relevance}")
        seen.add(chunk_id)
        normalized.append(
            {
                "chunk_id": chunk_id,
                "relevance": relevance,
                "reason": str(item.get("reason") or "").strip(),
            }
        )

    missing = expected - seen
    if missing:
        raise ValueError(f"LLM audit omitted chunk_ids: {sorted(missing)}")
    order = {chunk_id: index for index, chunk_id in enumerate(expected_chunk_ids)}
    return sorted(normalized, key=lambda row: order[row["chunk_id"]])


def agreement_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize exact, near, and severe human/LLM agreement."""
    total = len(rows)
    differences = [abs(int(row["human_relevance"]) - int(row["llm_relevance"])) for row in rows]
    exact = sum(1 for difference in differences if difference == 0)
    within_one = sum(1 for difference in differences if difference <= 1)
    severe = sum(1 for difference in differences if difference >= 2)
    return {
        "total": total,
        "exact_agreement": exact,
        "exact_agreement_rate": round(exact / total, 4) if total else 0.0,
        "within_one": within_one,
        "within_one_rate": round(within_one / total, 4) if total else 0.0,
        "severe_disagreements": severe,
        "severe_disagreement_rate": round(severe / total, 4) if total else 0.0,
    }


def refresh_human_judgments(
    audit_rows: list[dict[str, Any]],
    human_lookup: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Refresh cached audit rows after a human adjudication changes qrels."""
    refreshed: list[dict[str, Any]] = []
    for row in audit_rows:
        key = (str(row["query_id"]), str(row["chunk_id"]))
        human_row = human_lookup.get(key)
        if human_row is None:
            raise ValueError(f"audit row has no matching human judgment: {key}")
        item = dict(row)
        item["human_relevance"] = int(human_row["relevance"])
        item["human_note"] = str(human_row.get("note") or "")
        item["human_updated_at"] = str(human_row.get("updated_at") or "")
        item["absolute_difference"] = abs(
            item["human_relevance"] - int(item["llm_relevance"])
        )
        refreshed.append(item)
    return refreshed
