"""Compare rerankers on identical hybrid-retrieval candidate pools."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_encoder_reranking import cross_encoder_rerank, get_cross_encoder, runtime_config  # noqa: E402
from src.ollama_http import unload_embedding_model  # noqa: E402
from src.query_planning import QueryPlan, plan_query  # noqa: E402
from src.retrieval import (  # noqa: E402
    DEFAULT_CHUNKS_PATH,
    RetrievedChunk,
    apply_plan_boosts,
    planned_retrieve,
    rerank,
    retrieve_with_strategy,
    select_with_plan_coverage,
)


DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "rag_system_smoke_eval.jsonl"
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "cross_encoder_reranker_experiment"
DEFAULT_COLLECTION = "llm_rag_docs"
MODEL_SPECS = {
    "cross_encoder_base": ("fastembed", "BAAI/bge-reranker-base"),
    "cross_encoder_multilingual": ("transformers", "BAAI/bge-reranker-v2-m3"),
    "cross_encoder_fused": ("transformers", "BAAI/bge-reranker-v2-m3"),
}
RERANK_MODES = ["none", "lexical", *MODEL_SPECS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rerankers over fixed hybrid candidate pools.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--retrieval-mode", choices=["direct", "planned"], default="direct")
    parser.add_argument("--mode", action="append", choices=RERANK_MODES, default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--candidate-pools", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    configure_console_output()
    args = parse_args()
    modes = args.mode or RERANK_MODES
    cases = load_cases(args.dataset, set(args.case_id))
    if not cases:
        raise SystemExit("No reranker evaluation cases selected.")

    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(name=args.collection)
    collection_count = collection.count()
    candidate_pool_path = args.candidate_pools or args.output_dir / "candidate_pools.jsonl"
    if args.candidate_pools:
        case_pools = load_candidate_pools(candidate_pool_path, cases)
    else:
        case_pools = retrieve_candidate_pools(collection, cases, args)
        save_candidate_pools(candidate_pool_path, case_pools)
        unload_embedding_model(args.embedding_model, args.ollama_host)

    del collection
    del client
    gc.collect()
    encoders, model_load_seconds = warm_cross_encoders(modes)

    records: list[dict[str, Any]] = []
    for case_index, (case, candidate_pool, retrieval_seconds, plan) in enumerate(case_pools, start=1):
        pool_grades = relevance_grades(case, candidate_pool)

        for mode in modes:
            candidates = clone_chunks(candidate_pool)
            rerank_started = time.perf_counter()
            ranked = rank_and_select(
                str(case["question"]),
                candidates,
                args.top_k,
                mode,
                encoders,
                plan,
            )
            rerank_seconds = time.perf_counter() - rerank_started
            evaluation = evaluate_ranked(case, ranked, pool_grades, args.top_k)
            records.append(
                {
                    "case_id": case["id"],
                    "question": case["question"],
                    "mode": mode,
                    "candidate_count": len(candidate_pool),
                    "retrieval_seconds": round(retrieval_seconds, 4),
                    "rerank_seconds": round(rerank_seconds, 4),
                    "evaluation": evaluation,
                    "results": chunks_to_rows(ranked, case),
                }
            )
            print(
                f"[{case_index}/{len(cases)}] {case['id']} {mode} "
                f"both={evaluation['both_pass']} ndcg={evaluation['ndcg_at_k']:.3f} "
                f"mrr={evaluation['category_reciprocal_rank']:.3f} rerank={rerank_seconds:.2f}s",
                flush=True,
            )

    summary = build_summary(records, cases, modes, args, collection_count, model_load_seconds)
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
    for row in summary["modes"]:
        print(
            f"{row['mode']}: both={row['both_pass_rate']:.2%} ndcg={row['avg_ndcg_at_k']:.3f} "
            f"mrr={row['category_mrr']:.3f} recall={row['avg_source_term_recall']:.3f} "
            f"rerank={row['avg_rerank_seconds']:.2f}s"
        )
    print(f"summary: {args.output_dir / 'summary.md'}")


def retrieve_candidate_pools(
    collection: chromadb.Collection,
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], list[RetrievedChunk], float, QueryPlan | None]]:
    pools: list[tuple[dict[str, Any], list[RetrievedChunk], float, QueryPlan | None]] = []
    for case_index, case in enumerate(cases, start=1):
        started = time.perf_counter()
        if args.retrieval_mode == "planned":
            plan, candidates = planned_retrieve(
                collection,
                str(case["question"]),
                args.embedding_model,
                args.ollama_host,
                top_k=args.candidate_k,
                candidate_k=args.candidate_k,
                rerank_mode="none",
                retrieval_strategy="hybrid",
                chunks_path=args.chunks,
            )
        else:
            plan = None
            candidates = retrieve_with_strategy(
                collection,
                str(case["question"]),
                args.embedding_model,
                args.ollama_host,
                top_k=args.candidate_k,
                candidate_k=args.candidate_k,
                rerank_mode="none",
                retrieval_strategy="hybrid",
                chunks_path=args.chunks,
            )
        seconds = time.perf_counter() - started
        pools.append((case, candidates, seconds, plan))
        print(f"[candidate {case_index}/{len(cases)}] {case['id']} retrieved={len(candidates)}", flush=True)
    return pools


def save_candidate_pools(
    path: Path,
    case_pools: list[tuple[dict[str, Any], list[RetrievedChunk], float, QueryPlan | None]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "case_id": case["id"],
            "retrieval_seconds": round(retrieval_seconds, 4),
            "plan": plan.as_dict() if plan else None,
            "candidates": [asdict(candidate) for candidate in candidates],
        }
        for case, candidates, retrieval_seconds, plan in case_pools
    ]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def load_candidate_pools(
    path: Path,
    cases: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[RetrievedChunk], float, QueryPlan | None]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_case_id = {str(row["case_id"]): row for row in rows}
    missing = [str(case["id"]) for case in cases if str(case["id"]) not in by_case_id]
    if missing:
        raise ValueError(f"candidate pool file is missing cases: {', '.join(missing)}")
    return [
        (
            case,
            [RetrievedChunk(**candidate) for candidate in by_case_id[str(case["id"])]["candidates"]],
            float(by_case_id[str(case["id"])]["retrieval_seconds"]),
            plan_query(str(case["question"])) if by_case_id[str(case["id"])].get("plan") else None,
        )
        for case in cases
    ]


def warm_cross_encoders(modes: list[str]) -> tuple[dict[str, Any], dict[str, float]]:
    encoders: dict[str, Any] = {}
    load_seconds: dict[str, float] = {}
    for mode, (backend, model_name) in MODEL_SPECS.items():
        if mode not in modes:
            continue
        started = time.perf_counter()
        encoder = get_cross_encoder(model_name, backend=backend)
        list(encoder.rerank("RAG reranker warmup", ["cross encoder relevance scoring"], batch_size=1))
        load_seconds[mode] = time.perf_counter() - started
        encoders[mode] = encoder
    return encoders, load_seconds


def rank_and_select(
    query: str,
    candidates: list[RetrievedChunk],
    top_k: int,
    mode: str,
    encoders: dict[str, Any],
    plan: QueryPlan | None,
) -> list[RetrievedChunk]:
    rerank_top_k = len(candidates) if plan else top_k
    if mode not in MODEL_SPECS:
        ranked = rerank(query, candidates, rerank_top_k, mode)
    else:
        backend, model_name = MODEL_SPECS[mode]
        ranked = cross_encoder_rerank(
            query,
            candidates,
            rerank_top_k,
            encoder=encoders[mode],
            model_name=model_name,
            backend=backend,
            retrieval_weight=0.65 if mode == "cross_encoder_fused" else 0.0,
        )
    if plan:
        ranked = apply_plan_boosts(ranked, plan)
        return select_with_plan_coverage(ranked, top_k, plan)
    return ranked


def load_cases(path: Path, selected_ids: set[str]) -> list[dict[str, Any]]:
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


def clone_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        replace(
            chunk,
            metadata=dict(chunk.metadata),
            retrieval_channels=list(chunk.retrieval_channels),
        )
        for chunk in chunks
    ]


def evaluate_ranked(
    case: dict[str, Any],
    ranked: list[RetrievedChunk],
    pool_grades: dict[str, float],
    top_k: int,
) -> dict[str, Any]:
    selected = ranked[:top_k]
    expected_categories = [str(value) for value in case.get("expected_categories") or []]
    actual_categories = [str(chunk.metadata.get("category", "")) for chunk in selected]
    category_hits = sorted(set(actual_categories).intersection(expected_categories))
    min_category_hits = int(case.get("min_category_hits", 0))
    category_pass = min_category_hits <= 0 or len(category_hits) >= min_category_hits

    expected_terms = [str(value) for value in case.get("expected_source_terms") or []]
    source_blob = "\n".join(searchable_text(chunk) for chunk in selected).casefold()
    source_term_hits = [term for term in expected_terms if term.casefold() in source_blob]
    min_source_term_hits = int(case.get("min_source_term_hits", 0))
    source_terms_pass = min_source_term_hits <= 0 or len(source_term_hits) >= min_source_term_hits
    source_term_recall = len(source_term_hits) / len(expected_terms) if expected_terms else 1.0

    expected_category_set = set(expected_categories)
    first_category_rank = next(
        (
            rank
            for rank, chunk in enumerate(selected, start=1)
            if str(chunk.metadata.get("category", "")) in expected_category_set
        ),
        None,
    )
    ranked_grades = [pool_grades.get(chunk.chunk_id, 0.0) for chunk in selected]
    ideal_grades = sorted(pool_grades.values(), reverse=True)[:top_k]
    return {
        "category_pass": category_pass,
        "category_hits": category_hits,
        "source_terms_pass": source_terms_pass,
        "source_term_hits": source_term_hits,
        "source_term_recall": round(source_term_recall, 4),
        "both_pass": category_pass and source_terms_pass,
        "first_category_rank": first_category_rank,
        "category_reciprocal_rank": round(1.0 / first_category_rank, 4) if first_category_rank else 0.0,
        "ndcg_at_k": round(ndcg(ranked_grades, ideal_grades), 4),
        "top1_relevance_grade": round(ranked_grades[0], 4) if ranked_grades else 0.0,
        "unique_categories": len(set(actual_categories)),
    }


def relevance_grades(case: dict[str, Any], chunks: list[RetrievedChunk]) -> dict[str, float]:
    expected_categories = {str(value) for value in case.get("expected_categories") or []}
    expected_terms = [str(value).casefold() for value in case.get("expected_source_terms") or []]
    grades: dict[str, float] = {}
    for chunk in chunks:
        text = searchable_text(chunk).casefold()
        category_grade = 2.0 if str(chunk.metadata.get("category", "")) in expected_categories else 0.0
        term_hits = sum(1 for term in expected_terms if term in text)
        term_grade = 2.0 * term_hits / len(expected_terms) if expected_terms else 0.0
        grades[chunk.chunk_id] = category_grade + term_grade
    return grades


def ndcg(ranked_grades: list[float], ideal_grades: list[float]) -> float:
    ideal = discounted_cumulative_gain(ideal_grades)
    return discounted_cumulative_gain(ranked_grades) / ideal if ideal > 0 else 1.0


def discounted_cumulative_gain(grades: list[float]) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def searchable_text(chunk: RetrievedChunk) -> str:
    return "\n".join(
        [
            str(chunk.metadata.get("title", "")),
            str(chunk.metadata.get("category", "")),
            str(chunk.metadata.get("heading_path", "")),
            chunk.document,
        ]
    )


def chunks_to_rows(chunks: list[RetrievedChunk], case: dict[str, Any]) -> list[dict[str, Any]]:
    grades = relevance_grades(case, chunks)
    return [
        {
            "rank": rank,
            "chunk_id": chunk.chunk_id,
            "title": chunk.metadata.get("title"),
            "category": chunk.metadata.get("category"),
            "heading_path": chunk.metadata.get("heading_path"),
            "score": round(float(chunk.score), 6),
            "relevance_grade": round(grades[chunk.chunk_id], 4),
            "retrieval_channels": chunk.retrieval_channels,
            "rerank_reason": chunk.rerank_reason,
        }
        for rank, chunk in enumerate(chunks, start=1)
    ]


def build_summary(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    modes: list[str],
    args: argparse.Namespace,
    collection_count: int,
    model_load_seconds: dict[str, float],
) -> dict[str, Any]:
    mode_rows = []
    for mode in modes:
        selected = [record for record in records if record["mode"] == mode]
        count = len(selected)
        mode_rows.append(
            {
                "mode": mode,
                "cases": count,
                "both_pass_rate": round(
                    sum(1 for record in selected if record["evaluation"]["both_pass"]) / count, 4
                ),
                "avg_ndcg_at_k": mean_metric(selected, "ndcg_at_k"),
                "category_mrr": mean_metric(selected, "category_reciprocal_rank"),
                "avg_source_term_recall": mean_metric(selected, "source_term_recall"),
                "avg_top1_relevance_grade": mean_metric(selected, "top1_relevance_grade"),
                "avg_unique_categories": mean_metric(selected, "unique_categories"),
                "avg_rerank_seconds": round(statistics.mean(record["rerank_seconds"] for record in selected), 4),
            }
        )
    return {
        "dataset": portable_path(args.dataset),
        "case_ids": [case["id"] for case in cases],
        "collection_count": collection_count,
        "retrieval_strategy": f"{args.retrieval_mode}-hybrid",
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "rerankers": {
            mode: portable_runtime_config(runtime_config(model_name, backend=backend))
            for mode, (backend, model_name) in MODEL_SPECS.items()
            if mode in modes
        },
        "model_warmup_seconds": {mode: round(seconds, 4) for mode, seconds in model_load_seconds.items()},
        "modes": mode_rows,
        "metric_note": "nDCG uses category and evidence-term relevance grades within each fixed candidate pool.",
    }


def mean_metric(records: list[dict[str, Any]], name: str) -> float:
    return round(statistics.mean(float(record["evaluation"][name]) for record in records), 4)


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def portable_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    portable = dict(config)
    portable["cache_dir"] = portable_path(Path(str(config["cache_dir"])))
    return portable


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    reranker_descriptions = [
        f"{mode}={config['model']} ({config['backend']}/{config['device']})"
        for mode, config in summary["rerankers"].items()
    ]
    lines = [
        "# Cross-encoder Reranker Comparison",
        "",
        "Each query uses one fixed hybrid candidate pool. Only the reranking method changes.",
        "",
        "## Setup",
        "",
        f"- cases: {len(summary['case_ids'])}",
        f"- collection chunks: {summary['collection_count']}",
        f"- retrieval strategy: {summary['retrieval_strategy']}",
        f"- top_k/candidate_k: {summary['top_k']}/{summary['candidate_k']}",
        f"- rerankers: {', '.join(reranker_descriptions)}",
        f"- warmup seconds: {json.dumps(summary['model_warmup_seconds'], ensure_ascii=False)}",
        "- nDCG relevance: category match plus expected evidence-term coverage within the fixed pool",
        "",
        "## Results",
        "",
        "| mode | both pass | nDCG@k | category MRR | term recall | top-1 grade | categories | rerank seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["modes"]:
        lines.append(
            f"| {row['mode']} | {row['both_pass_rate']:.2%} | {row['avg_ndcg_at_k']:.3f} | "
            f"{row['category_mrr']:.3f} | {row['avg_source_term_recall']:.3f} | "
            f"{row['avg_top1_relevance_grade']:.3f} | {row['avg_unique_categories']:.2f} | "
            f"{row['avg_rerank_seconds']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Per-case nDCG@k",
            "",
            "| case | " + " | ".join(row["mode"] for row in summary["modes"]) + " |",
            "|---|" + "---:|" * len(summary["modes"]),
        ]
    )
    by_key = {(record["case_id"], record["mode"]): record for record in records}
    for case_id in summary["case_ids"]:
        values = [f"{by_key[(case_id, row['mode'])]['evaluation']['ndcg_at_k']:.3f}" for row in summary["modes"]]
        lines.append(f"| {case_id} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "These are automatic pool-based relevance grades, not exhaustive human judgments over all 938 chunks.",
            "Use them to compare ordering on the same candidates; inspect changed rankings before deciding the default reranker.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
