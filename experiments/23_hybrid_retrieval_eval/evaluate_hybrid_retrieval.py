"""Compare dense, BM25+dense hybrid, reranked, and planned retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.bm25_retrieval import get_bm25_index  # noqa: E402
from src.ollama_http import embed_query  # noqa: E402
from src.retrieval import DEFAULT_CHUNKS_PATH, RetrievedChunk, planned_retrieve, retrieve_with_strategy  # noqa: E402


DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "rag_system_smoke_eval.jsonl"
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "hybrid_retrieval_experiment"
DEFAULT_COLLECTION = "llm_rag_docs"


@dataclass(frozen=True)
class Variant:
    name: str
    retrieval_strategy: str
    rerank_mode: str
    planned: bool = False


VARIANTS = {
    variant.name: variant
    for variant in [
        Variant("dense", "dense", "none"),
        Variant("dense_lexical_rerank", "dense", "lexical"),
        Variant("hybrid", "hybrid", "none"),
        Variant("hybrid_lexical_rerank", "hybrid", "lexical"),
        Variant("planned_dense_lexical_rerank", "dense", "lexical", planned=True),
        Variant("planned_hybrid_lexical_rerank", "hybrid", "lexical", planned=True),
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid retrieval comparison matrix.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS), default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    configure_console_output()
    args = parse_args()
    cases = load_retrieval_cases(args.dataset, set(args.case_id))
    variants = [VARIANTS[name] for name in args.variant] if args.variant else list(VARIANTS.values())
    if not cases:
        raise SystemExit("No retrieval evaluation cases selected.")

    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(name=args.collection)
    warm_dependencies(args)

    records: list[dict[str, Any]] = []
    total_runs = len(cases) * len(variants)
    run_index = 0
    for case in cases:
        for variant in variants:
            run_index += 1
            started = time.perf_counter()
            plan, chunks = run_variant(collection, case["question"], variant, args)
            elapsed = time.perf_counter() - started
            evaluation = evaluate_chunks(case, chunks, args.top_k)
            record = {
                "case_id": case["id"],
                "question": case["question"],
                "variant": variant.name,
                "retrieval_strategy": variant.retrieval_strategy,
                "rerank_mode": variant.rerank_mode,
                "planned": variant.planned,
                "elapsed_seconds": round(elapsed, 4),
                "evaluation": evaluation,
                "plan": plan,
                "results": chunks_to_rows(chunks),
            }
            records.append(record)
            print(
                f"[{run_index}/{total_runs}] {case['id']} {variant.name} "
                f"category={evaluation['category_pass']} terms={evaluation['source_terms_pass']} "
                f"both={evaluation['both_pass']} time={elapsed:.2f}s",
                flush=True,
            )

    summary = build_summary(records, cases, variants, args, collection.count())
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    for row in summary["variants"]:
        print(
            f"{row['name']}: both={row['both_pass_rate']:.2%} "
            f"category={row['category_pass_rate']:.2%} terms={row['source_terms_pass_rate']:.2%} "
            f"mrr={row['category_mrr']:.3f} term_recall={row['avg_source_term_recall']:.3f} "
            f"avg={row['avg_seconds']:.2f}s"
        )
    print(f"summary: {args.output_dir / 'summary.md'}")


def load_retrieval_cases(path: Path, selected_ids: set[str]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        if selected_ids and str(case.get("id")) not in selected_ids:
            continue
        if int(case.get("min_category_hits", 0)) <= 0 and int(case.get("min_source_term_hits", 0)) <= 0:
            continue
        cases.append(case)
    return cases


def warm_dependencies(args: argparse.Namespace) -> None:
    # Exclude one-time model/index startup from the comparison timings.
    get_bm25_index(args.chunks)
    embed_query("RAG retrieval warmup", args.embedding_model, args.ollama_host)


def run_variant(
    collection: chromadb.Collection,
    query: str,
    variant: Variant,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, list[RetrievedChunk]]:
    if variant.planned:
        plan, chunks = planned_retrieve(
            collection,
            query,
            args.embedding_model,
            args.ollama_host,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rerank_mode=variant.rerank_mode,
            retrieval_strategy=variant.retrieval_strategy,
            chunks_path=args.chunks,
        )
        return plan.as_dict(), chunks

    chunks = retrieve_with_strategy(
        collection,
        query,
        args.embedding_model,
        args.ollama_host,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rerank_mode=variant.rerank_mode,
        retrieval_strategy=variant.retrieval_strategy,
        chunks_path=args.chunks,
    )
    return None, chunks


def evaluate_chunks(case: dict[str, Any], chunks: list[RetrievedChunk], top_k: int) -> dict[str, Any]:
    selected = chunks[:top_k]
    actual_categories = [str(chunk.metadata.get("category", "")) for chunk in selected]
    expected_categories = [str(value) for value in case.get("expected_categories") or []]
    category_hits = sorted(set(actual_categories).intersection(expected_categories))
    min_category_hits = int(case.get("min_category_hits", 0))
    category_pass = min_category_hits <= 0 or len(category_hits) >= min_category_hits

    source_blob = "\n".join(searchable_text(chunk) for chunk in selected).casefold()
    expected_terms = [str(value) for value in case.get("expected_source_terms") or []]
    source_term_hits = [term for term in expected_terms if term.casefold() in source_blob]
    min_source_term_hits = int(case.get("min_source_term_hits", 0))
    source_terms_pass = min_source_term_hits <= 0 or len(source_term_hits) >= min_source_term_hits
    source_term_recall = len(source_term_hits) / len(expected_terms) if expected_terms else 1.0

    first_category_rank = next(
        (rank for rank, category in enumerate(actual_categories, start=1) if category in set(expected_categories)),
        None,
    )
    return {
        "category_pass": category_pass,
        "category_hits": category_hits,
        "min_category_hits": min_category_hits,
        "actual_categories": actual_categories,
        "source_terms_pass": source_terms_pass,
        "source_term_hits": source_term_hits,
        "min_source_term_hits": min_source_term_hits,
        "source_term_recall": round(source_term_recall, 4),
        "first_category_rank": first_category_rank,
        "reciprocal_rank": round(1.0 / first_category_rank, 4) if first_category_rank else 0.0,
        "both_pass": category_pass and source_terms_pass,
    }


def searchable_text(chunk: RetrievedChunk) -> str:
    metadata = chunk.metadata
    return "\n".join(
        [
            str(metadata.get("title", "")),
            str(metadata.get("category", "")),
            str(metadata.get("heading_path", "")),
            chunk.document,
        ]
    )


def chunks_to_rows(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "chunk_id": chunk.chunk_id,
            "title": chunk.metadata.get("title"),
            "category": chunk.metadata.get("category"),
            "heading_path": chunk.metadata.get("heading_path"),
            "score": round(float(chunk.score), 6),
            "distance": round(float(chunk.distance), 6),
            "retrieval_channels": chunk.retrieval_channels,
            "rerank_reason": chunk.rerank_reason,
            "source_query": chunk.source_query,
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]


def build_summary(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    variants: list[Variant],
    args: argparse.Namespace,
    collection_count: int,
) -> dict[str, Any]:
    variant_rows = []
    for variant in variants:
        selected = [record for record in records if record["variant"] == variant.name]
        count = len(selected)
        category_passes = sum(1 for record in selected if record["evaluation"]["category_pass"])
        term_passes = sum(1 for record in selected if record["evaluation"]["source_terms_pass"])
        both_passes = sum(1 for record in selected if record["evaluation"]["both_pass"])
        variant_rows.append(
            {
                "name": variant.name,
                "retrieval_strategy": variant.retrieval_strategy,
                "rerank_mode": variant.rerank_mode,
                "planned": variant.planned,
                "cases": count,
                "category_pass_rate": round(category_passes / count, 4),
                "source_terms_pass_rate": round(term_passes / count, 4),
                "both_pass_rate": round(both_passes / count, 4),
                "category_mrr": round(
                    statistics.mean(record["evaluation"]["reciprocal_rank"] for record in selected), 4
                ),
                "avg_source_term_recall": round(
                    statistics.mean(record["evaluation"]["source_term_recall"] for record in selected), 4
                ),
                "avg_seconds": round(statistics.mean(record["elapsed_seconds"] for record in selected), 4),
            }
        )
    return {
        "dataset": str(args.dataset),
        "case_ids": [case["id"] for case in cases],
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "collection_count": collection_count,
        "variants": variant_rows,
    }


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Hybrid Retrieval Comparison",
        "",
        "This experiment isolates retrieval quality before answer generation. BM25 and Chroma reuse the same chunks.",
        "",
        "## Setup",
        "",
        f"- cases: {len(summary['case_ids'])}",
        f"- top_k: {summary['top_k']}",
        f"- candidate_k: {summary['candidate_k']}",
        "- dense channel: Chroma + bge-m3",
        "- sparse channel: BM25 with English tokens and Chinese bi/tri-grams",
        "- fusion: Reciprocal Rank Fusion (RRF)",
        "",
        "## Variant Results",
        "",
        "| variant | category pass | evidence-term pass | both pass | category MRR | avg term recall | avg seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["variants"]:
        lines.append(
            f"| {row['name']} | {row['category_pass_rate']:.2%} | "
            f"{row['source_terms_pass_rate']:.2%} | {row['both_pass_rate']:.2%} | "
            f"{row['category_mrr']:.3f} | {row['avg_source_term_recall']:.3f} | {row['avg_seconds']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per-case Both-pass",
            "",
            "| case | " + " | ".join(row["name"] for row in summary["variants"]) + " |",
            "|---|" + "---:|" * len(summary["variants"]),
        ]
    )
    by_key = {(record["case_id"], record["variant"]): record for record in records}
    for case_id in summary["case_ids"]:
        values = ["PASS" if by_key[(case_id, row["name"])]["evaluation"]["both_pass"] else "FAIL" for row in summary["variants"]]
        lines.append(f"| {case_id} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Reading the Metrics",
            "",
            "- category pass checks whether the retrieved set covers the required knowledge areas.",
            "- evidence-term pass checks whether the chunks contain enough expected technical evidence, not only the right label.",
            "- category MRR rewards placing the first relevant category near rank 1.",
            "- avg term recall measures how much of the expected evidence vocabulary appears in top-k chunks.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
