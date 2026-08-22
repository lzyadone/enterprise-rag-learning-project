"""Evaluate candidate generators against complete pooled qrels."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_judgments import load_candidate_pools, load_complete_qrels  # noqa: E402
from src.retrieval_metrics import evaluate_retrieval_ranking  # noqa: E402


DEFAULT_CANDIDATE_POOLS = PROJECT_ROOT / "eval" / "retrieval_union_v1" / "candidate_pools.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "eval" / "retrieval_union_v1" / "pool_manifest.jsonl"
DEFAULT_QRELS = (
    PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_union_v1" / "qrels_llm.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "retrieval_union_v1" / "llm_judged_system_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval systems on a union pool.")
    parser.add_argument("--candidate-pools", type=Path, default=DEFAULT_CANDIDATE_POOLS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--relevant-threshold", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    pools = load_candidate_pools(args.candidate_pools)
    grades_by_query = load_complete_qrels(args.qrels, pools)
    manifests = load_jsonl(args.manifest)
    systems = list(manifests[0]["systems"])

    records: list[dict[str, Any]] = []
    for manifest in manifests:
        query_id = str(manifest["case_id"])
        grades = grades_by_query[query_id]
        for system in systems:
            run = manifest["systems"][system]
            ranked_ids = [str(chunk_id) for chunk_id in run["chunk_ids"]]
            at_5 = evaluate_retrieval_ranking(
                ranked_ids,
                grades,
                k=5,
                relevant_threshold=args.relevant_threshold,
            )
            at_10 = evaluate_retrieval_ranking(
                ranked_ids,
                grades,
                k=10,
                relevant_threshold=args.relevant_threshold,
            )
            records.append(
                {
                    "case_id": query_id,
                    "question": manifest["question"],
                    "system": system,
                    "seconds": float(run["seconds"]),
                    "metrics": {
                        "recall_at_5": at_5["recall_at_k"],
                        "precision_at_5": at_5["precision_at_k"],
                        "recall_at_10": at_10["recall_at_k"],
                        "precision_at_10": at_10["precision_at_k"],
                        "mrr_at_10": at_10["reciprocal_rank"],
                        "ndcg_at_10": at_10["ndcg_at_k"],
                        "relevant_in_pool": at_10["relevant_in_pool"],
                        "relevant_retrieved_at_10": at_10["relevant_retrieved"],
                    },
                }
            )

    summary = build_summary(args, records, manifests, systems)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "results.jsonl", records)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(
        summary_markdown(summary, records),
        encoding="utf-8",
    )

    print("=== Candidate Generator Summary ===")
    for row in summary["systems"]:
        print(
            f"{row['system']}: R@5={row['avg_recall_at_5']:.3f} "
            f"R@10={row['avg_recall_at_10']:.3f} MRR={row['mrr_at_10']:.3f} "
            f"nDCG={row['avg_ndcg_at_10']:.3f} median={row['median_seconds']:.2f}s"
        )
    print(f"summary: {portable_path(args.output_dir / 'summary.md')}")


def build_summary(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    systems: list[str],
) -> dict[str, Any]:
    system_rows: list[dict[str, Any]] = []
    for system in systems:
        selected = [row for row in records if row["system"] == system]
        seconds = [float(row["seconds"]) for row in selected]
        system_rows.append(
            {
                "system": system,
                "cases": len(selected),
                "avg_recall_at_5": mean_metric(selected, "recall_at_5"),
                "avg_precision_at_5": mean_metric(selected, "precision_at_5"),
                "avg_recall_at_10": mean_metric(selected, "recall_at_10"),
                "avg_precision_at_10": mean_metric(selected, "precision_at_10"),
                "mrr_at_10": mean_metric(selected, "mrr_at_10"),
                "avg_ndcg_at_10": mean_metric(selected, "ndcg_at_10"),
                "avg_seconds": round(statistics.mean(seconds), 4),
                "median_seconds": round(statistics.median(seconds), 4),
                "max_seconds": round(max(seconds), 4),
            }
        )
    return {
        "qrels": portable_path(args.qrels),
        "qrels_type": "complete_blind_llm_judgments",
        "cases": len(manifests),
        "pooled_query_chunk_pairs": sum(row["pooled_candidate_count"] for row in manifests),
        "relevant_threshold": args.relevant_threshold,
        "systems": system_rows,
        "metric_boundary": (
            "Metrics are computed against a depth-10 union pool, not exhaustive judgments over "
            "all collection chunks. LLM labels are consistent across all 224 pairs but are not "
            "a substitute for independent human judgments."
        ),
    }


def mean_metric(records: list[dict[str, Any]], metric: str) -> float:
    return round(statistics.mean(float(row["metrics"][metric]) for row in records), 4)


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Candidate Generator Evaluation (LLM-Judged Union Pool)",
        "",
        "## Setup",
        "",
        f"- cases: {summary['cases']}",
        f"- pooled query/chunk pairs: {summary['pooled_query_chunk_pairs']}",
        f"- relevant threshold: >= {summary['relevant_threshold']}",
        f"- qrels: {summary['qrels']}",
        f"- qrels type: {summary['qrels_type']}",
        "",
        "## Results",
        "",
        "| system | Recall@5 | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds | max seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["systems"]:
        lines.append(
            f"| {row['system']} | {row['avg_recall_at_5']:.3f} | "
            f"{row['avg_recall_at_10']:.3f} | {row['avg_precision_at_10']:.3f} | "
            f"{row['mrr_at_10']:.3f} | {row['avg_ndcg_at_10']:.3f} | "
            f"{row['median_seconds']:.2f} | {row['max_seconds']:.2f} |"
        )

    systems = [row["system"] for row in summary["systems"]]
    by_key = {(row["case_id"], row["system"]): row for row in records}
    case_ids = list(dict.fromkeys(row["case_id"] for row in records))
    lines.extend(
        [
            "",
            "## Per-case Recall@10 / nDCG@10",
            "",
            "| case | " + " | ".join(systems) + " |",
            "|---|" + "---:|" * len(systems),
        ]
    )
    for case_id in case_ids:
        values = []
        for system in systems:
            metrics = by_key[(case_id, system)]["metrics"]
            values.append(f"{metrics['recall_at_10']:.3f} / {metrics['ndcg_at_10']:.3f}")
        lines.append(f"| {case_id} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            summary["metric_boundary"],
            "Use the result to choose which candidate generators deserve human validation and online testing.",
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
