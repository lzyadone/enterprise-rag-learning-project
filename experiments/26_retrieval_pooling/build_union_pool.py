"""Build a pooled retrieval benchmark from multiple candidate generators."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ollama_http import unload_embedding_model  # noqa: E402
from src.retrieval import (  # noqa: E402
    DEFAULT_CHUNKS_PATH,
    bm25_retrieve,
    planned_retrieve,
    retrieve_with_strategy,
)
from src.retrieval_pooling import (  # noqa: E402
    chunk_record_to_candidate,
    inherit_qrels,
    pool_rankings,
)


DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "rag_system_smoke_eval.jsonl"
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "retrieval_union_v1"
DEFAULT_SOURCE_QRELS = PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_v1" / "qrels.jsonl"
DEFAULT_TARGET_QRELS = (
    PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_union_v1" / "qrels.jsonl"
)
DEFAULT_COLLECTION = "llm_rag_docs"
SYSTEMS = [
    "direct_dense",
    "direct_bm25",
    "direct_hybrid",
    "planned_dense",
    "planned_hybrid",
]
AVAILABLE_SYSTEMS = [*SYSTEMS, "planned_v2_hybrid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a judged union pool for retrieval evaluation.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--pool-depth", type=int, default=10)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=AVAILABLE_SYSTEMS,
        default=SYSTEMS,
        help="Candidate generators to include in the union pool.",
    )
    parser.add_argument(
        "--all-dataset-cases",
        action="store_true",
        help="Use every dataset row in file order instead of selecting cases from source qrels.",
    )
    parser.add_argument(
        "--no-inherit-qrels",
        action="store_true",
        help="Build a fresh pool without importing or writing any existing judgments.",
    )
    parser.add_argument("--source-qrels", type=Path, default=DEFAULT_SOURCE_QRELS)
    parser.add_argument("--target-qrels", type=Path, default=DEFAULT_TARGET_QRELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    if args.pool_depth <= 0:
        raise ValueError("pool-depth must be positive")
    systems = list(dict.fromkeys(args.systems))

    source_qrels = [] if args.no_inherit_qrels else load_jsonl(args.source_qrels)
    target_qrels = (
        load_jsonl(args.target_qrels)
        if not args.no_inherit_qrels and args.target_qrels.exists()
        else []
    )
    if args.all_dataset_cases:
        cases = load_all_cases(args.dataset)
    else:
        qrel_query_ids = list(dict.fromkeys(str(row["query_id"]) for row in source_qrels))
        if not qrel_query_ids:
            raise ValueError("source qrels are empty; pass --all-dataset-cases for a fresh dataset")
        cases = load_cases(args.dataset, qrel_query_ids)
    chunk_candidates = load_chunk_candidates(args.chunks)
    legacy_rows = merge_qrel_rows(source_qrels, target_qrels)
    legacy_by_query = legacy_candidates_by_query(legacy_rows, chunk_candidates)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "system_runs.jsonl"
    cache = {} if args.force else load_run_cache(cache_path)

    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(name=args.collection)
    try:
        for case_index, case in enumerate(cases, start=1):
            for system in systems:
                key = (str(case["id"]), system)
                if key in cache:
                    print(f"[{case_index}/{len(cases)}] {case['id']} {system}: reused", flush=True)
                    continue
                record = run_system(collection, case, system, args)
                cache[key] = record
                write_run_cache(cache_path, cache, cases, systems)
                print(
                    f"[{case_index}/{len(cases)}] {case['id']} {system}: "
                    f"{len(record['candidates'])} candidates in {record['seconds']:.2f}s",
                    flush=True,
                )
    finally:
        del collection
        del client
        unload_embedding_model(args.embedding_model, args.ollama_host)

    union_pools: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["id"])
        rankings = {
            system: cache[(case_id, system)]["candidates"]
            for system in systems
        }
        pooled, provenance = pool_rankings(
            case_id,
            rankings,
            legacy_by_query.get(case_id, []),
        )
        planned_system = next((system for system in systems if system.startswith("planned_")), None)
        plan = cache[(case_id, planned_system)].get("plan") if planned_system else None
        union_pools.append(
            {
                "case_id": case_id,
                "question": case["question"],
                "plan": plan,
                "candidates": pooled,
            }
        )
        manifests.append(
            {
                "case_id": case_id,
                "question": case["question"],
                "systems": {
                    system: {
                        "seconds": cache[(case_id, system)]["seconds"],
                        "chunk_ids": [
                            candidate["chunk_id"]
                            for candidate in cache[(case_id, system)]["candidates"]
                        ],
                    }
                    for system in systems
                },
                "pooled_candidate_count": len(pooled),
                "provenance": provenance,
            }
        )

    candidate_pool_path = args.output_dir / "candidate_pools.jsonl"
    manifest_path = args.output_dir / "pool_manifest.jsonl"
    write_jsonl(candidate_pool_path, union_pools)
    write_jsonl(manifest_path, manifests)

    valid_pair_order = [
        (str(pool["case_id"]), str(candidate["chunk_id"]))
        for pool in union_pools
        for candidate in pool["candidates"]
    ]
    inherited = inherit_qrels(legacy_rows, valid_pair_order)
    if not args.no_inherit_qrels:
        write_jsonl(args.target_qrels, inherited)
    summary = build_summary(args, manifests, inherited, systems)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")

    print("\n=== Pool Summary ===")
    print(json.dumps(summary["judgments"], ensure_ascii=False, indent=2))
    print(f"candidate pools: {portable_path(candidate_pool_path)}")
    if not args.no_inherit_qrels:
        print(f"target qrels: {portable_path(args.target_qrels)}")


def run_system(
    collection: chromadb.Collection,
    case: dict[str, Any],
    system: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    query = str(case["question"])
    started = time.perf_counter()
    plan = None
    if system == "direct_bm25":
        candidates = bm25_retrieve(query, top_k=args.pool_depth, chunks_path=args.chunks)
    elif system.startswith("direct_"):
        strategy = system.removeprefix("direct_")
        candidates = retrieve_with_strategy(
            collection,
            query,
            args.embedding_model,
            args.ollama_host,
            top_k=args.pool_depth,
            candidate_k=args.pool_depth,
            rerank_mode="none",
            retrieval_strategy=strategy,
            chunks_path=args.chunks,
        )
    elif system.startswith("planned_v2_"):
        strategy = system.removeprefix("planned_v2_")
        plan, candidates = planned_retrieve(
            collection,
            query,
            args.embedding_model,
            args.ollama_host,
            top_k=args.pool_depth,
            candidate_k=args.pool_depth,
            rerank_mode="none",
            retrieval_strategy=strategy,
            chunks_path=args.chunks,
            fusion_mode="anchored",
        )
    elif system.startswith("planned_"):
        strategy = system.removeprefix("planned_")
        plan, candidates = planned_retrieve(
            collection,
            query,
            args.embedding_model,
            args.ollama_host,
            top_k=args.pool_depth,
            candidate_k=args.pool_depth,
            rerank_mode="none",
            retrieval_strategy=strategy,
            chunks_path=args.chunks,
        )
    else:
        raise ValueError(f"unsupported retrieval system: {system}")
    return {
        "case_id": str(case["id"]),
        "system": system,
        "seconds": round(time.perf_counter() - started, 4),
        "plan": plan.as_dict() if plan else None,
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def load_cases(path: Path, ordered_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {
        str(row["id"]): row
        for row in load_jsonl(path)
        if isinstance(row, dict) and row.get("id")
    }
    missing = [query_id for query_id in ordered_ids if query_id not in by_id]
    if missing:
        raise ValueError(f"dataset is missing qrel queries: {missing}")
    return [by_id[query_id] for query_id in ordered_ids]


def load_all_cases(path: Path) -> list[dict[str, Any]]:
    cases = load_jsonl(path)
    ids = [str(row.get("id") or "") for row in cases]
    if any(not case_id for case_id in ids):
        raise ValueError("every dataset row must contain a non-empty id")
    if len(ids) != len(set(ids)):
        raise ValueError("dataset case ids must be unique")
    if any(not str(row.get("question") or "").strip() for row in cases):
        raise ValueError("every dataset row must contain a non-empty question")
    return cases


def load_chunk_candidates(path: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        candidate = chunk_record_to_candidate(record)
        candidates[candidate["chunk_id"]] = candidate
    return candidates


def legacy_candidates_by_query(
    qrels: list[dict[str, Any]],
    chunks: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in qrels:
        query_id = str(row["query_id"])
        chunk_id = str(row["chunk_id"])
        if chunk_id not in chunks:
            raise ValueError(f"qrels reference a missing processed chunk: {chunk_id}")
        by_query.setdefault(query_id, []).append(chunks[chunk_id])
    return by_query


def merge_qrel_rows(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for rows in groups:
        for row in rows:
            pair = (str(row["query_id"]), str(row["chunk_id"]))
            merged[pair] = row
    return list(merged.values())


def load_run_cache(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["case_id"]), str(row["system"])): row
        for row in load_jsonl(path)
    } if path.exists() else {}


def write_run_cache(
    path: Path,
    cache: dict[tuple[str, str], dict[str, Any]],
    cases: list[dict[str, Any]],
    systems: list[str],
) -> None:
    case_order = {str(case["id"]): index for index, case in enumerate(cases)}
    system_order = {system: index for index, system in enumerate(systems)}
    rows = sorted(
        cache.values(),
        key=lambda row: (case_order[str(row["case_id"])], system_order[str(row["system"])]),
    )
    write_jsonl(path, rows)


def build_summary(
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    inherited: list[dict[str, Any]],
    systems: list[str],
) -> dict[str, Any]:
    pool_sizes = [int(row["pooled_candidate_count"]) for row in manifests]
    total_pairs = sum(pool_sizes)
    system_unique = {system: 0 for system in systems}
    legacy_only = 0
    for manifest in manifests:
        for item in manifest["provenance"]:
            contributing_systems = list(item["system_ranks"])
            if len(contributing_systems) == 1:
                system_unique[contributing_systems[0]] += 1
            if item["legacy_judged"] and not contributing_systems:
                legacy_only += 1
    return {
        "dataset": portable_path(args.dataset),
        "collection": args.collection,
        "pool_depth_per_system": args.pool_depth,
        "systems": systems,
        "cases": len(manifests),
        "pool": {
            "total_query_chunk_pairs": total_pairs,
            "min_candidates_per_query": min(pool_sizes),
            "max_candidates_per_query": max(pool_sizes),
            "avg_candidates_per_query": round(statistics.mean(pool_sizes), 2),
            "system_unique_contributions": system_unique,
            "legacy_judged_not_in_current_top_depth": legacy_only,
        },
        "judgments": {
            "inherited": len(inherited),
            "total_required": total_pairs,
            "remaining": total_pairs - len(inherited),
            "completion_rate": round(len(inherited) / total_pairs, 4) if total_pairs else 0.0,
        },
        "candidate_pools": portable_path(args.output_dir / "candidate_pools.jsonl"),
        "manifest": portable_path(args.output_dir / "pool_manifest.jsonl"),
        "qrels": None if args.no_inherit_qrels else portable_path(args.target_qrels),
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    pool = summary["pool"]
    judgments = summary["judgments"]
    lines = [
        "# Retrieval Union Pool V1",
        "",
        f"This pool combines {len(summary['systems'])} retrieval systems and deduplicates candidates per query.",
        "Candidate order is deterministically shuffled before labeling so system rank is hidden.",
        "",
        "## Setup",
        "",
        f"- cases: {summary['cases']}",
        f"- pool depth per system: {summary['pool_depth_per_system']}",
        f"- systems: {', '.join(summary['systems'])}",
        f"- total query/chunk pairs: {pool['total_query_chunk_pairs']}",
        f"- candidates per query: {pool['min_candidates_per_query']} to {pool['max_candidates_per_query']} "
        f"(avg {pool['avg_candidates_per_query']})",
        "",
        "## Existing Judgments",
        "",
        f"- inherited: {judgments['inherited']}",
        f"- remaining: {judgments['remaining']}",
        f"- completion: {judgments['completion_rate']:.2%}",
        "",
        "## Unique Contributions",
        "",
        "| system | candidates found only by this system |",
        "|---|---:|",
    ]
    for system, count in pool["system_unique_contributions"].items():
        lines.append(f"| {system} | {count} |")
    lines.extend(
        [
            "",
            f"Legacy judged candidates outside every current top-depth run: "
            f"{pool['legacy_judged_not_in_current_top_depth']}.",
            "",
            "The union pool supports fair comparison between candidate generators after remaining pairs are judged.",
            "It still estimates relevance through depth pooling rather than exhaustively judging all collection chunks.",
            "",
        ]
    )
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
