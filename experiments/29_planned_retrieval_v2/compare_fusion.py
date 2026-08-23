"""Compare anchored planned retrieval with direct and legacy planned baselines."""

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

from src.ollama_http import embed_query, unload_embedding_model  # noqa: E402
from src.retrieval import DEFAULT_CHUNKS_PATH, planned_retrieve  # noqa: E402
from src.retrieval_judgments import load_candidate_pools, load_complete_qrels  # noqa: E402
from src.retrieval_metrics import evaluate_retrieval_ranking  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "eval" / "retrieval_union_v1" / "pool_manifest.jsonl"
DEFAULT_POOLS = PROJECT_ROOT / "eval" / "retrieval_union_v1" / "candidate_pools.jsonl"
DEFAULT_QRELS = PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_union_v1" / "qrels_llm.jsonl"
DEFAULT_ADDITIONAL_QRELS = (
    PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_union_v1" / "qrels_planned_v2_incremental.jsonl"
)
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "planned_retrieval_v2_dev"
DEFAULT_COLLECTION = "llm_rag_docs"
SYSTEMS = ("direct_hybrid", "planned_hybrid", "planned_v2_hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare planned retrieval fusion modes.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-pools", type=Path, default=DEFAULT_POOLS)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument(
        "--additional-qrels",
        type=Path,
        nargs="*",
        default=[DEFAULT_ADDITIONAL_QRELS],
        help="Zero or more incremental qrels files layered over the immutable base qrels.",
    )
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument("--relevant-threshold", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    manifests = load_jsonl(args.manifest)
    pools = load_candidate_pools(args.candidate_pools)
    grades_by_query = load_complete_qrels(args.qrels, pools)
    for additional_qrels in args.additional_qrels:
        if additional_qrels.exists():
            merge_additional_qrels(grades_by_query, load_jsonl(additional_qrels))
    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(name=args.collection)
    records: list[dict[str, Any]] = []
    unjudged: list[dict[str, Any]] = []
    try:
        embed_query("planned v2 benchmark warmup", args.embedding_model, args.ollama_host)
        for index, manifest in enumerate(manifests, start=1):
            case_id = str(manifest["case_id"])
            question = str(manifest["question"])
            started = time.perf_counter()
            plan, candidates = planned_retrieve(
                collection,
                question,
                args.embedding_model,
                args.ollama_host,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                rerank_mode="none",
                retrieval_strategy="hybrid",
                chunks_path=args.chunks,
                fusion_mode="anchored",
            )
            seconds = time.perf_counter() - started
            candidate_ids = [candidate.chunk_id for candidate in candidates]
            missing = [chunk_id for chunk_id in candidate_ids if chunk_id not in grades_by_query[case_id]]
            if missing:
                candidate_by_id = {candidate.chunk_id: candidate for candidate in candidates}
                for chunk_id in missing:
                    candidate = candidate_by_id[chunk_id]
                    unjudged.append(
                        {
                            "case_id": case_id,
                            "question": question,
                            "chunk_id": chunk_id,
                            "document": candidate.document,
                            "metadata": candidate.metadata,
                        }
                    )
            records.append(
                {
                    "case_id": case_id,
                    "question": question,
                    "planned_v2_seconds": round(seconds, 4),
                    "planned_v2_chunk_ids": candidate_ids,
                    "unjudged_chunk_ids": missing,
                    "plan": plan.as_dict(),
                }
            )
            print(
                f"[{index}/{len(manifests)}] {case_id}: {seconds:.2f}s, "
                f"unjudged={len(missing)}",
                flush=True,
            )
    finally:
        del collection
        del client
        unload_embedding_model(args.embedding_model, args.ollama_host)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "results.jsonl", records)
    write_jsonl(args.output_dir / "unjudged_candidates.jsonl", unjudged)
    write_jsonl(
        args.output_dir / "unjudged_candidate_pools.jsonl",
        build_unjudged_candidate_pools(unjudged),
    )
    if unjudged:
        summary = {
            "status": "needs_additional_judging",
            "cases": len(records),
            "unjudged_pairs": len(unjudged),
            "unjudged_cases": len({row["case_id"] for row in unjudged}),
        }
        write_json(args.output_dir / "summary.json", summary)
        (args.output_dir / "summary.md").write_text(incomplete_markdown(summary), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    complete_records = add_metrics(args, manifests, records, grades_by_query)
    summary = build_summary(args, complete_records)
    write_jsonl(args.output_dir / "results.jsonl", complete_records)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(
        summary_markdown(summary, complete_records),
        encoding="utf-8",
    )
    print("\n=== Planned V2 Development Summary ===")
    for row in summary["systems"]:
        print(
            f"{row['system']}: R@10={row['avg_recall_at_10']:.3f} "
            f"MRR={row['mrr_at_10']:.3f} nDCG={row['avg_ndcg_at_10']:.3f}"
        )


def add_metrics(
    args: argparse.Namespace,
    manifests: list[dict[str, Any]],
    records: list[dict[str, Any]],
    grades_by_query: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    manifest_by_id = {str(row["case_id"]): row for row in manifests}
    complete = []
    for record in records:
        case_id = str(record["case_id"])
        manifest = manifest_by_id[case_id]
        rankings = {
            "direct_hybrid": manifest["systems"]["direct_hybrid"]["chunk_ids"],
            "planned_hybrid": manifest["systems"]["planned_hybrid"]["chunk_ids"],
            "planned_v2_hybrid": record["planned_v2_chunk_ids"],
        }
        item = dict(record)
        item["metrics"] = {
            system: ranking_metrics(
                rankings[system],
                grades_by_query[case_id],
                args.relevant_threshold,
            )
            for system in SYSTEMS
        }
        complete.append(item)
    return complete


def merge_additional_qrels(
    grades_by_query: dict[str, dict[str, int]],
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        case_id = str(row["query_id"])
        if case_id not in grades_by_query:
            raise ValueError(f"additional qrels contain an unknown query: {case_id}")
        relevance = int(row["relevance"])
        if relevance not in {0, 1, 2, 3}:
            raise ValueError(f"invalid additional relevance grade: {relevance}")
        grades_by_query[case_id][str(row["chunk_id"])] = relevance


def build_unjudged_candidate_pools(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pools: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        pool = pools.setdefault(
            case_id,
            {
                "case_id": case_id,
                "question": str(row["question"]),
                "plan": None,
                "candidates": [],
            },
        )
        pool["candidates"].append(
            {
                "chunk_id": str(row["chunk_id"]),
                "document": str(row["document"]),
                "metadata": dict(row.get("metadata") or {}),
            }
        )
    return list(pools.values())


def ranking_metrics(
    chunk_ids: list[str],
    grades: dict[str, int],
    relevant_threshold: int,
) -> dict[str, float]:
    values = evaluate_retrieval_ranking(
        [str(chunk_id) for chunk_id in chunk_ids],
        grades,
        k=10,
        relevant_threshold=relevant_threshold,
    )
    return {
        "recall_at_10": float(values["recall_at_k"]),
        "precision_at_10": float(values["precision_at_k"]),
        "mrr_at_10": float(values["reciprocal_rank"]),
        "ndcg_at_10": float(values["ndcg_at_k"]),
    }


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    systems = []
    for system in SYSTEMS:
        rows = [record["metrics"][system] for record in records]
        systems.append(
            {
                "system": system,
                "avg_recall_at_10": mean(rows, "recall_at_10"),
                "avg_precision_at_10": mean(rows, "precision_at_10"),
                "mrr_at_10": mean(rows, "mrr_at_10"),
                "avg_ndcg_at_10": mean(rows, "ndcg_at_10"),
            }
        )
    return {
        "status": "complete",
        "evaluation_role": "development/calibration; not an independent holdout",
        "cases": len(records),
        "qrels": portable_path(args.qrels),
        "relevant_threshold": args.relevant_threshold,
        "planned_v2_median_seconds": round(
            statistics.median(float(row["planned_v2_seconds"]) for row in records),
            4,
        ),
        "systems": systems,
    }


def mean(rows: list[dict[str, float]], key: str) -> float:
    return round(statistics.mean(row[key] for row in rows), 4)


def incomplete_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Planned Retrieval V2 Development A/B",
            "",
            "Scoring stopped because the anchored system retrieved candidates outside the judged pool.",
            "",
            f"- cases: {summary['cases']}",
            f"- unjudged query/chunk pairs: {summary['unjudged_pairs']}",
            f"- affected cases: {summary['unjudged_cases']}",
            "",
            "Judge these candidates before comparing metrics; unjudged evidence must not be treated as irrelevant.",
            "",
        ]
    )


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Planned Retrieval V2 Development A/B",
        "",
        "This is a development/calibration result, not an independent holdout.",
        "Anchored v2 keeps the original query dominant, caps total expansion weight, deduplicates runs,",
        "and limits forced plan coverage to at most two slots.",
        "",
        "| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["systems"]:
        lines.append(
            f"| {row['system']} | {row['avg_recall_at_10']:.3f} | "
            f"{row['avg_precision_at_10']:.3f} | {row['mrr_at_10']:.3f} | "
            f"{row['avg_ndcg_at_10']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Anchored v2 median latency: {summary['planned_v2_median_seconds']:.2f}s.",
            "",
            "## Per-case nDCG@10",
            "",
            "| case | direct | legacy planned | anchored v2 | v2 seconds |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for record in records:
        metrics = record["metrics"]
        lines.append(
            f"| {record['case_id']} | {metrics['direct_hybrid']['ndcg_at_10']:.3f} | "
            f"{metrics['planned_hybrid']['ndcg_at_10']:.3f} | "
            f"{metrics['planned_v2_hybrid']['ndcg_at_10']:.3f} | "
            f"{record['planned_v2_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            "All ranked chunks were required to have complete qrels before metrics were computed.",
            "A separate untouched holdout is required after development decisions are finalized.",
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
