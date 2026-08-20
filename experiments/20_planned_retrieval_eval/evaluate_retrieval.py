"""Compare direct retrieval and planned retrieval on a small RAG knowledge eval set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import RetrievedChunk, direct_retrieve, planned_retrieve  # noqa: E402


DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_COLLECTION = "llm_rag_docs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "planned_retrieval_smoke"


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


SMOKE_CASES = [
    {
        "query": "RAG 的完整流程包括哪些阶段？",
        "expected_categories": ["RAG overview"],
    },
    {
        "query": "文档切分为什么不能只用固定窗口？",
        "expected_categories": ["chunking"],
    },
    {
        "query": "metadata filter 在 RAG 检索中有什么作用？",
        "expected_categories": ["vector db", "retrieval"],
    },
    {
        "query": "如何评估 RAG 回答是否忠实于检索上下文？",
        "expected_categories": ["evaluation"],
    },
    {
        "query": "bge-m3 这种 embedding 模型在知识库里负责什么？",
        "expected_categories": ["embedding"],
    },
    {
        "query": "rerank 和普通向量检索有什么关系？",
        "expected_categories": ["reranking", "retrieval"],
    },
    {
        "query": "Ollama 本地 API 怎么用于生成或 embedding？",
        "expected_categories": ["local model"],
    },
    {
        "query": "ingestion pipeline 为什么要包含 transformation 和 metadata？",
        "expected_categories": ["ingestion"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct vs planned retrieval.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--rerank-mode", choices=["none", "lexical"], default="lexical")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(name=args.collection)

    records: list[dict[str, Any]] = []
    for idx, case in enumerate(SMOKE_CASES, start=1):
        query = case["query"]
        expected = case["expected_categories"]
        direct = direct_retrieve(
            collection,
            query,
            args.embedding_model,
            args.ollama_host,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rerank_mode=args.rerank_mode,
        )
        plan, planned = planned_retrieve(
            collection,
            query,
            args.embedding_model,
            args.ollama_host,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rerank_mode=args.rerank_mode,
        )
        record = {
            "index": idx,
            "query": query,
            "expected_categories": expected,
            "plan": plan.as_dict(),
            "direct": summarize_hits(direct, expected),
            "planned": summarize_hits(planned, expected),
            "direct_results": chunks_to_rows(direct),
            "planned_results": chunks_to_rows(planned),
        }
        records.append(record)
        print(
            f"[{idx}/{len(SMOKE_CASES)}] "
            f"direct_hit@1={record['direct']['hit@1']} "
            f"planned_hit@1={record['planned']['hit@1']} "
            f"query={query}",
            flush=True,
        )

    summary = build_summary(records, args)
    (args.output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "summary.md").write_text(summary_markdown(summary, records), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {args.output_dir / 'results.jsonl'}")
    print(f"summary: {args.output_dir / 'summary.md'}")


def summarize_hits(chunks: list[RetrievedChunk], expected_categories: list[str]) -> dict[str, Any]:
    categories = [str(chunk.metadata.get("category")) for chunk in chunks]
    expected_set = set(expected_categories)
    return {
        "categories": categories,
        "hit@1": bool(categories and categories[0] in expected_set),
        "hit@3": any(category in expected_set for category in categories[:3]),
    }


def chunks_to_rows(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    rows = []
    for chunk in chunks:
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.metadata.get("title"),
                "category": chunk.metadata.get("category"),
                "heading_path": chunk.metadata.get("heading_path"),
                "distance": chunk.distance,
                "score": chunk.score,
                "source_query": chunk.source_query,
                "category_filter": chunk.category_filter,
            }
        )
    return rows


def build_summary(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    total = len(records)
    direct_hit1 = sum(1 for record in records if record["direct"]["hit@1"])
    planned_hit1 = sum(1 for record in records if record["planned"]["hit@1"])
    direct_hit3 = sum(1 for record in records if record["direct"]["hit@3"])
    planned_hit3 = sum(1 for record in records if record["planned"]["hit@3"])
    return {
        "total": total,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "rerank_mode": args.rerank_mode,
        "direct_hit@1": direct_hit1,
        "planned_hit@1": planned_hit1,
        "direct_hit@3": direct_hit3,
        "planned_hit@3": planned_hit3,
        "direct_hit@1_rate": round(direct_hit1 / total, 4),
        "planned_hit@1_rate": round(planned_hit1 / total, 4),
        "direct_hit@3_rate": round(direct_hit3 / total, 4),
        "planned_hit@3_rate": round(planned_hit3 / total, 4),
    }


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Planned Retrieval Smoke Evaluation",
        "",
        "## Summary",
        "",
        f"- total: {summary['total']}",
        f"- top_k: {summary['top_k']}",
        f"- candidate_k: {summary['candidate_k']}",
        f"- rerank_mode: {summary['rerank_mode']}",
        f"- direct hit@1: {summary['direct_hit@1']} ({summary['direct_hit@1_rate']:.2%})",
        f"- planned hit@1: {summary['planned_hit@1']} ({summary['planned_hit@1_rate']:.2%})",
        f"- direct hit@3: {summary['direct_hit@3']} ({summary['direct_hit@3_rate']:.2%})",
        f"- planned hit@3: {summary['planned_hit@3']} ({summary['planned_hit@3_rate']:.2%})",
        "",
        "## Cases",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"### {record['index']}. {record['query']}",
                "",
                f"- expected: {', '.join(record['expected_categories'])}",
                f"- plan categories: {', '.join(record['plan']['category_filters']) or 'none'}",
                f"- direct categories: {', '.join(record['direct']['categories'])}",
                f"- planned categories: {', '.join(record['planned']['categories'])}",
                f"- direct hit@1: {record['direct']['hit@1']}",
                f"- planned hit@1: {record['planned']['hit@1']}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
