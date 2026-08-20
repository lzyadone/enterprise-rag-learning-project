"""Build structure-aware chunks from normalized LLM/RAG documents."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import chunks_to_records, split_markdown_document  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "llm_rag_docs" / "documents.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "llm_rag_docs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split LLM/RAG docs into structure-aware chunks.")
    parser.add_argument("--documents", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--soft-max-chars", type=int, default=1800)
    parser.add_argument("--hard-max-chars", type=int, default=3500)
    parser.add_argument("--min-chars", type=int, default=280)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    docs = read_jsonl(args.documents)
    if args.limit:
        docs = docs[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output_dir / "chunks.jsonl"
    summary_path = args.output_dir / "chunk_summary.json"

    all_chunks: list[dict[str, object]] = []
    per_doc_counts: dict[str, int] = {}

    for doc in docs:
        required = ["doc_id", "source_id", "title", "category", "priority", "source_type", "url", "text"]
        missing = [field for field in required if field not in doc]
        if missing:
            raise ValueError(f"Document record missing fields {missing}: {doc.get('doc_id')}")

        chunks = split_markdown_document(
            str(doc["text"]),
            title=str(doc["title"]),
            soft_max_chars=args.soft_max_chars,
            hard_max_chars=args.hard_max_chars,
            min_chars=args.min_chars,
        )
        records = chunks_to_records(
            chunks,
            {
                "doc_id": str(doc["doc_id"]),
                "source_id": str(doc["source_id"]),
                "title": str(doc["title"]),
                "category": str(doc["category"]),
                "priority": str(doc["priority"]),
                "source_type": str(doc["source_type"]),
                "url": str(doc["url"]),
            },
        )
        per_doc_counts[str(doc["source_id"])] = len(records)
        all_chunks.extend(records)

    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    char_counts = [int(chunk["char_count"]) for chunk in all_chunks]
    token_counts = [int(chunk["token_estimate"]) for chunk in all_chunks]
    by_category = Counter(str(chunk["category"]) for chunk in all_chunks)
    by_doc_category: dict[str, Counter[str]] = defaultdict(Counter)
    for chunk in all_chunks:
        by_doc_category[str(chunk["category"])][str(chunk["source_id"])] += 1

    summary = {
        "documents": len(docs),
        "chunks": len(all_chunks),
        "soft_max_chars": args.soft_max_chars,
        "hard_max_chars": args.hard_max_chars,
        "min_chars": args.min_chars,
        "char_count": describe(char_counts),
        "token_estimate": describe(token_counts),
        "by_category": dict(sorted(by_category.items())),
        "per_doc_counts": dict(sorted(per_doc_counts.items())),
        "too_long_chunks": [
            chunk["chunk_id"]
            for chunk in all_chunks
            if int(chunk["char_count"]) > args.hard_max_chars + 250
        ],
        "tiny_chunks": [
            chunk["chunk_id"]
            for chunk in all_chunks
            if int(chunk["char_count"]) < 120
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"documents: {len(docs)}")
    print(f"chunks: {len(all_chunks)}")
    print(f"chunks_jsonl: {chunks_path}")
    print(f"summary: {summary_path}")
    print("char_count:", summary["char_count"])
    print("too_long_chunks:", len(summary["too_long_chunks"]))
    print("tiny_chunks:", len(summary["tiny_chunks"]))


def describe(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "avg": None}
    values_sorted = sorted(values)
    return {
        "min": values_sorted[0],
        "p50": values_sorted[len(values_sorted) // 2],
        "p90": values_sorted[int(len(values_sorted) * 0.9)],
        "max": values_sorted[-1],
        "avg": round(sum(values_sorted) / len(values_sorted), 2),
    }


if __name__ == "__main__":
    main()
