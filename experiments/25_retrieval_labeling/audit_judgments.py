"""Blindly audit human retrieval labels with an independent LLM judge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deepseek_client import DEFAULT_MODEL, chat_completion  # noqa: E402
from src.relevance_audit import (  # noqa: E402
    agreement_summary,
    parse_llm_judgments,
    refresh_human_judgments,
)
from src.retrieval_judgments import JudgmentStore, load_candidate_pools  # noqa: E402


DEFAULT_CANDIDATE_POOLS = PROJECT_ROOT / "eval" / "planned_reranker_full" / "candidate_pools.jsonl"
DEFAULT_QRELS = PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_v1" / "qrels.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_v1" / "llm_audit.jsonl"
DEFAULT_REVIEW_QUEUE = PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_v1" / "review_queue.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_v1" / "audit_summary.json"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blindly audit human retrieval labels with DeepSeek.")
    parser.add_argument("--candidate-pools", type=Path, default=DEFAULT_CANDIDATE_POOLS)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-document-chars", type=int, default=3200)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pools = load_candidate_pools(args.candidate_pools)
    human_store = JudgmentStore(args.qrels, pools)
    expected_count = sum(len(pool["candidates"]) for pool in pools)
    if human_store.progress()["labeled"] != expected_count:
        raise RuntimeError(
            f"human qrels are incomplete: {human_store.progress()['labeled']}/{expected_count}"
        )

    human = {(row["query_id"], row["chunk_id"]): row for row in human_store.rows()}
    audit = {} if args.force else load_existing(args.output)
    if audit:
        refreshed = refresh_human_judgments(list(audit.values()), human)
        audit = {(row["query_id"], row["chunk_id"]): row for row in refreshed}
    for index, pool in enumerate(pools, start=1):
        query_id = str(pool["case_id"])
        candidates = list(pool["candidates"])
        if all((query_id, str(candidate["chunk_id"])) in audit for candidate in candidates):
            print(f"[{index}/{len(pools)}] {query_id}: reused existing audit", flush=True)
            continue

        model_rows = judge_pool(pool, model=args.model, max_document_chars=args.max_document_chars)
        for model_row in model_rows:
            chunk_id = model_row["chunk_id"]
            human_row = human[(query_id, chunk_id)]
            audit[(query_id, chunk_id)] = {
                "query_id": query_id,
                "chunk_id": chunk_id,
                "human_relevance": human_row["relevance"],
                "llm_relevance": model_row["relevance"],
                "absolute_difference": abs(human_row["relevance"] - model_row["relevance"]),
                "human_note": human_row.get("note") or "",
                "human_updated_at": human_row.get("updated_at") or "",
                "llm_reason": model_row["reason"],
                "model": args.model,
            }
        write_jsonl(args.output, list(audit.values()), pools)
        print(f"[{index}/{len(pools)}] {query_id}: audited {len(candidates)} candidates", flush=True)

    rows = ordered_rows(list(audit.values()), pools)
    write_jsonl(args.output, rows, pools)
    severe_rows = [row for row in rows if int(row["absolute_difference"]) >= 2]
    review_rows = [
        row for row in severe_rows if not str(row.get("human_note") or "").startswith("复核：")
    ]
    write_jsonl(args.review_queue, review_rows, pools)
    summary = agreement_summary(rows)
    summary.update(
        {
            "adjudicated_severe_disagreements": len(severe_rows) - len(review_rows),
            "open_review_items": len(review_rows),
            "human_label_distribution": label_distribution(rows, "human_relevance"),
            "llm_label_distribution": label_distribution(rows, "llm_relevance"),
        }
    )
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"audit: {portable_path(args.output)}")
    print(f"review queue: {portable_path(args.review_queue)}")
    print(f"summary: {portable_path(args.summary)}")


def judge_pool(pool: dict[str, Any], model: str, max_document_chars: int) -> list[dict[str, Any]]:
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
    user_payload = {
        "question": pool["question"],
        "candidates": candidates,
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
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
    raise RuntimeError(f"LLM audit returned invalid JSON after 3 attempts: {last_error}")


def load_existing(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {(str(row["query_id"]), str(row["chunk_id"])): row for row in rows}


def write_jsonl(path: Path, rows: list[dict[str, Any]], pools: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in ordered_rows(rows, pools)
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def label_distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return {
        str(label): sum(1 for row in rows if int(row[field]) == label)
        for label in range(4)
    }


def ordered_rows(rows: list[dict[str, Any]], pools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {
        (str(pool["case_id"]), str(candidate["chunk_id"])): position
        for position, (pool, candidate) in enumerate(
            (pool, candidate) for pool in pools for candidate in pool["candidates"]
        )
    }
    return sorted(rows, key=lambda row: order[(str(row["query_id"]), str(row["chunk_id"]))])


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
