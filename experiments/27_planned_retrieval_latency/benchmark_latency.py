"""A/B benchmark repeated versus reused embeddings in planned retrieval."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ollama_http import (  # noqa: E402
    clear_embedding_cache,
    embed_query,
    embedding_cache_info,
    unload_embedding_model,
)
from src.retrieval import DEFAULT_CHUNKS_PATH, planned_retrieve  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "eval" / "retrieval_union_v1" / "pool_manifest.jsonl"
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "planned_retrieval_cache_benchmark"
DEFAULT_COLLECTION = "llm_rag_docs"
SYSTEMS = ["planned_dense", "planned_hybrid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark optimized planned retrieval latency.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    manifests = load_jsonl(args.manifest)
    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(name=args.collection)
    records: list[dict[str, Any]] = []
    try:
        embed_query("embedding cache benchmark warmup", args.embedding_model, args.ollama_host)
        for case_index, manifest in enumerate(manifests, start=1):
            for system in SYSTEMS:
                strategy = system.removeprefix("planned_")
                clear_embedding_cache()
                started = time.perf_counter()
                _, baseline_candidates = planned_retrieve(
                    collection,
                    str(manifest["question"]),
                    args.embedding_model,
                    args.ollama_host,
                    top_k=args.top_k,
                    candidate_k=args.candidate_k,
                    rerank_mode="none",
                    retrieval_strategy=strategy,
                    chunks_path=args.chunks,
                    reuse_query_embeddings=False,
                )
                baseline_seconds = time.perf_counter() - started
                baseline_cache = embedding_cache_info()

                clear_embedding_cache()
                started = time.perf_counter()
                plan, optimized_candidates = planned_retrieve(
                    collection,
                    str(manifest["question"]),
                    args.embedding_model,
                    args.ollama_host,
                    top_k=args.top_k,
                    candidate_k=args.candidate_k,
                    rerank_mode="none",
                    retrieval_strategy=strategy,
                    chunks_path=args.chunks,
                    reuse_query_embeddings=True,
                )
                optimized_seconds = time.perf_counter() - started
                optimized_cache = embedding_cache_info()
                baseline_ids = [candidate.chunk_id for candidate in baseline_candidates]
                optimized_ids = [candidate.chunk_id for candidate in optimized_candidates]
                record = {
                    "case_id": manifest["case_id"],
                    "system": system,
                    "baseline_seconds": round(baseline_seconds, 4),
                    "optimized_seconds": round(optimized_seconds, 4),
                    "speedup": round(baseline_seconds / optimized_seconds, 4),
                    "same_order": optimized_ids == baseline_ids,
                    "same_set": set(optimized_ids) == set(baseline_ids),
                    "unique_plan_queries": len(
                        set(
                            [aspect.search_query or aspect.question for aspect in plan.aspects]
                            + plan.sub_queries
                        )
                    ),
                    "baseline_embedding_calls": baseline_cache["api_requests"],
                    "optimized_embedding_calls": optimized_cache["api_requests"],
                    "baseline_chunk_ids": baseline_ids,
                    "optimized_chunk_ids": optimized_ids,
                }
                records.append(record)
                print(
                    f"[{case_index}/{len(manifests)}] {manifest['case_id']} {system}: "
                    f"{baseline_seconds:.2f}s -> {optimized_seconds:.2f}s "
                    f"speedup={record['speedup']:.2f}x same_order={record['same_order']} "
                    f"embed_requests={record['baseline_embedding_calls']}"
                    f"->{record['optimized_embedding_calls']}",
                    flush=True,
                )
    finally:
        del collection
        del client
        unload_embedding_model(args.embedding_model, args.ollama_host)

    summary = build_summary(args, records, manifests)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "results.jsonl", records)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(
        summary_markdown(summary, records),
        encoding="utf-8",
    )
    print("\n=== Summary ===")
    for row in summary["systems"]:
        print(
            f"{row['system']}: median {row['baseline_median_seconds']:.2f}s -> "
            f"{row['optimized_median_seconds']:.2f}s, speedup={row['median_speedup']:.2f}x, "
            f"same_order={row['same_order_cases']}/{row['cases']}"
        )


def build_summary(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    systems = []
    for system in SYSTEMS:
        selected = [row for row in records if row["system"] == system]
        systems.append(
            {
                "system": system,
                "cases": len(selected),
                "baseline_median_seconds": round(
                    statistics.median(row["baseline_seconds"] for row in selected), 4
                ),
                "optimized_median_seconds": round(
                    statistics.median(row["optimized_seconds"] for row in selected), 4
                ),
                "median_speedup": round(statistics.median(row["speedup"] for row in selected), 4),
                "same_order_cases": sum(1 for row in selected if row["same_order"]),
                "same_set_cases": sum(1 for row in selected if row["same_set"]),
                "avg_unique_embedding_queries": round(
                    statistics.mean(row["unique_plan_queries"] for row in selected), 2
                ),
                "avg_baseline_embedding_calls": round(
                    statistics.mean(row["baseline_embedding_calls"] for row in selected), 2
                ),
                "avg_optimized_embedding_calls": round(
                    statistics.mean(row["optimized_embedding_calls"] for row in selected), 2
                ),
            }
        )
    return {
        "cases": len(manifests),
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "optimization": "A/B: repeated embeddings versus one embedding per unique plan query",
        "systems": systems,
    }


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Planned Retrieval Embedding Cache Benchmark",
        "",
        "The baseline and optimized runs use the same current retrieval code and deterministic plan.",
        "The only changed variable is repeated embedding calls versus one embedding per unique query.",
        "",
        "| system | baseline median | optimized median | median speedup | same Top-10 order | same Top-10 set | avg embedding calls before/after |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["systems"]:
        lines.append(
            f"| {row['system']} | {row['baseline_median_seconds']:.2f}s | "
            f"{row['optimized_median_seconds']:.2f}s | {row['median_speedup']:.2f}x | "
            f"{row['same_order_cases']}/{row['cases']} | {row['same_set_cases']}/{row['cases']} | "
            f"{row['avg_baseline_embedding_calls']:.2f} / "
            f"{row['avg_optimized_embedding_calls']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Per-case latency",
            "",
            "| case | system | before | after | speedup | same order | same set | embed calls before/after |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in records:
        lines.append(
            f"| {row['case_id']} | {row['system']} | {row['baseline_seconds']:.2f}s | "
            f"{row['optimized_seconds']:.2f}s | {row['speedup']:.2f}x | "
            f"{'Y' if row['same_order'] else 'N'} | {'Y' if row['same_set'] else 'N'} | "
            f"{row['baseline_embedding_calls']} / {row['optimized_embedding_calls']} |"
        )
    lines.extend(
        [
            "",
            "Candidate equality is checked between paired runs in the same process.",
            "The embedding model is warmed once; the in-process query cache is cleared before each run.",
            "",
        ]
    )
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
