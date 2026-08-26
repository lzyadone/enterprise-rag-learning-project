"""Benchmark serial, parallel-cold and parallel-warm planned retrieval."""

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

from src.index_versioning import load_active_index, resolve_stored_path, write_json_atomic  # noqa: E402
from src.ollama_http import embed_query, unload_embedding_model  # noqa: E402
from src.query_planning import plan_query_v3  # noqa: E402
from src.retrieval import build_conservative_run_specs, planned_retrieve  # noqa: E402
from src.retrieval_cache import clear_retrieval_caches, retrieval_cache_info  # noqa: E402


DEFAULT_ACTIVE_INDEX = PROJECT_ROOT / "data" / "runtime" / "active_index.json"
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "rag_system_smoke_eval.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "runtime" / "planned_parallel_cache_benchmark"
DEFAULT_CASE_IDS = ("rag_compound_overview", "rag_evaluation_reliability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-index", type=Path, default=DEFAULT_ACTIVE_INDEX)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    selected_ids = set(args.case_id or DEFAULT_CASE_IDS)
    cases = [row for row in read_jsonl(args.dataset) if str(row.get("id")) in selected_ids]
    if {str(case["id"]) for case in cases} != selected_ids:
        raise ValueError("Benchmark case IDs do not match the selected dataset rows")

    _, _, manifest = load_active_index(args.active_index, PROJECT_ROOT)
    db_dir = resolve_stored_path(str(manifest["db_dir"]), PROJECT_ROOT)
    chunks_path = resolve_stored_path(str(manifest["chunks_path"]), PROJECT_ROOT)
    embedding_model = str(manifest["embedding_model"])
    collection = chromadb.PersistentClient(path=str(db_dir)).get_collection(
        str(manifest["collection"])
    )
    records: list[dict[str, Any]] = []
    try:
        for case in cases:
            question = str(case["question"])
            planned_runs = len(
                build_conservative_run_specs(question, plan_query_v3(question))
            )
            warm_plan_embeddings(question, embedding_model, args.ollama_host)
            for repeat in range(1, args.repeats + 1):
                serial_seconds, serial_ids = timed_retrieval(
                    collection,
                    question,
                    embedding_model,
                    args.ollama_host,
                    chunks_path,
                    str(manifest["version_id"]),
                    args,
                    parallel=False,
                    use_cache=False,
                )
                clear_retrieval_caches()
                parallel_seconds, parallel_ids = timed_retrieval(
                    collection,
                    question,
                    embedding_model,
                    args.ollama_host,
                    chunks_path,
                    str(manifest["version_id"]),
                    args,
                    parallel=True,
                    use_cache=True,
                )
                cold_cache = retrieval_cache_info()
                warm_seconds, warm_ids = timed_retrieval(
                    collection,
                    question,
                    embedding_model,
                    args.ollama_host,
                    chunks_path,
                    str(manifest["version_id"]),
                    args,
                    parallel=True,
                    use_cache=True,
                )
                warm_cache = retrieval_cache_info()
                same_parallel = serial_ids == parallel_ids
                same_warm = serial_ids == warm_ids
                records.append(
                    {
                        "case_id": case["id"],
                        "planned_runs": planned_runs,
                        "repeat": repeat,
                        "serial_seconds": round(serial_seconds, 6),
                        "parallel_seconds": round(parallel_seconds, 6),
                        "warm_seconds": round(warm_seconds, 6),
                        "parallel_speedup": round(serial_seconds / parallel_seconds, 4),
                        "warm_speedup": round(serial_seconds / warm_seconds, 4),
                        "same_parallel_order": same_parallel,
                        "same_warm_order": same_warm,
                        "candidate_cache_after_cold": cold_cache["candidates"],
                        "candidate_cache_after_warm": warm_cache["candidates"],
                        "rerank_cache_after_cold": cold_cache["rerank"],
                        "rerank_cache_after_warm": warm_cache["rerank"],
                        "chunk_ids": serial_ids,
                    }
                )
                print(
                    f"{case['id']} repeat={repeat}: serial={serial_seconds:.4f}s "
                    f"parallel={parallel_seconds:.4f}s warm={warm_seconds:.4f}s "
                    f"same={same_parallel and same_warm}",
                    flush=True,
                )
    finally:
        clear_retrieval_caches()
        unload_embedding_model(embedding_model, args.ollama_host)

    summary = build_summary(manifest, args, cases, records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "results.jsonl", records)
    (args.output_dir / "summary.md").write_text(
        summary_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["acceptance"]["passed"]:
        raise SystemExit(1)


def timed_retrieval(
    collection: chromadb.Collection,
    question: str,
    embedding_model: str,
    ollama_host: str,
    chunks_path: Path,
    version_id: str,
    args: argparse.Namespace,
    *,
    parallel: bool,
    use_cache: bool,
) -> tuple[float, list[str]]:
    started = time.perf_counter()
    _, candidates = planned_retrieve(
        collection,
        question,
        embedding_model,
        ollama_host,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rerank_mode="lexical",
        retrieval_strategy="hybrid",
        chunks_path=chunks_path,
        query_plan=plan_query_v3(question),
        fusion_mode="conservative",
        parallel_runs=parallel,
        max_workers=args.max_workers,
        use_candidate_cache=use_cache,
        use_rerank_cache=use_cache,
        cache_namespace=version_id,
    )
    return time.perf_counter() - started, [candidate.chunk_id for candidate in candidates]


def warm_plan_embeddings(question: str, model: str, host: str) -> None:
    plan = plan_query_v3(question)
    queries = dict.fromkeys(spec.query for spec in build_conservative_run_specs(question, plan))
    for query in queries:
        embed_query(query, model, host)


def build_summary(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    same_order = all(
        row["same_parallel_order"] and row["same_warm_order"] for row in records
    )
    serial_median = statistics.median(row["serial_seconds"] for row in records)
    parallel_median = statistics.median(row["parallel_seconds"] for row in records)
    warm_median = statistics.median(row["warm_seconds"] for row in records)
    case_summaries = []
    for case in cases:
        selected = [row for row in records if row["case_id"] == case["id"]]
        case_serial = statistics.median(row["serial_seconds"] for row in selected)
        case_parallel = statistics.median(row["parallel_seconds"] for row in selected)
        case_warm = statistics.median(row["warm_seconds"] for row in selected)
        case_summaries.append(
            {
                "case_id": case["id"],
                "planned_runs": selected[0]["planned_runs"],
                "serial_median_seconds": round(case_serial, 6),
                "parallel_median_seconds": round(case_parallel, 6),
                "warm_cache_median_seconds": round(case_warm, 6),
                "parallel_speedup": round(case_serial / case_parallel, 4),
                "warm_cache_speedup": round(case_serial / case_warm, 4),
                "same_order_runs": sum(
                    1
                    for row in selected
                    if row["same_parallel_order"] and row["same_warm_order"]
                ),
                "total_runs": len(selected),
            }
        )
    return {
        "index_version": manifest["version_id"],
        "case_ids": [case["id"] for case in cases],
        "repeats": args.repeats,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "max_workers": args.max_workers,
        "serial_median_seconds": round(serial_median, 6),
        "parallel_median_seconds": round(parallel_median, 6),
        "warm_cache_median_seconds": round(warm_median, 6),
        "parallel_median_speedup": round(serial_median / parallel_median, 4),
        "warm_cache_median_speedup": round(serial_median / warm_median, 4),
        "same_order_runs": sum(
            1 for row in records if row["same_parallel_order"] and row["same_warm_order"]
        ),
        "total_runs": len(records),
        "cases": case_summaries,
        "acceptance": {
            "same_top_k_order_required": True,
            "passed": same_order,
        },
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
            "# Planned Retrieval Parallel and Cache Benchmark",
            "",
            f"- index version: {summary['index_version']}",
            f"- cases: {len(summary['case_ids'])}",
            f"- repeats per case: {summary['repeats']}",
            f"- serial median: {summary['serial_median_seconds']:.4f}s",
            f"- parallel cold median: {summary['parallel_median_seconds']:.4f}s",
            f"- parallel warm median: {summary['warm_cache_median_seconds']:.4f}s",
            f"- parallel speedup: {summary['parallel_median_speedup']:.2f}x",
            f"- warm cache speedup: {summary['warm_cache_median_speedup']:.2f}x",
            f"- identical Top-K order: {summary['same_order_runs']}/{summary['total_runs']}",
            f"- acceptance: {'PASS' if summary['acceptance']['passed'] else 'FAIL'}",
            "",
            "| case | runs | serial | parallel cold | warm cache | parallel speedup | warm speedup | same order |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["cases"]:
        lines.append(
            f"| {row['case_id']} | {row['planned_runs']} | "
            f"{row['serial_median_seconds']:.4f}s | {row['parallel_median_seconds']:.4f}s | "
            f"{row['warm_cache_median_seconds']:.4f}s | {row['parallel_speedup']:.2f}x | "
            f"{row['warm_cache_speedup']:.2f}x | {row['same_order_runs']}/{row['total_runs']} |"
        )
    lines.append("")
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
