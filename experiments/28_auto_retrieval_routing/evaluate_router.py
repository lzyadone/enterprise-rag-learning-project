"""Evaluate automatic direct/planned routing on the pooled retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.query_planning import plan_query  # noqa: E402
from src.retrieval_judgments import load_candidate_pools, load_complete_qrels  # noqa: E402
from src.retrieval_metrics import evaluate_retrieval_ranking  # noqa: E402
from src.retrieval_routing import route_retrieval  # noqa: E402


DEFAULT_MANIFEST = PROJECT_ROOT / "eval" / "retrieval_union_v1" / "pool_manifest.jsonl"
DEFAULT_CANDIDATE_POOLS = PROJECT_ROOT / "eval" / "retrieval_union_v1" / "candidate_pools.jsonl"
DEFAULT_QRELS = PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_union_v1" / "qrels_llm.jsonl"
DEFAULT_LATENCY_RESULTS = PROJECT_ROOT / "eval" / "planned_retrieval_cache_benchmark" / "results.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "auto_retrieval_routing"
SYSTEMS = ("direct_hybrid", "planned_hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate automatic retrieval routing.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidate-pools", type=Path, default=DEFAULT_CANDIDATE_POOLS)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--latency-results", type=Path, default=DEFAULT_LATENCY_RESULTS)
    parser.add_argument("--latency-budget-ms", type=int, default=12000)
    parser.add_argument("--relevant-threshold", type=int, default=2)
    parser.add_argument(
        "--evaluation-role",
        choices=("calibration", "holdout"),
        default="calibration",
        help="Declare whether routing rules have already seen this question set.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    manifests = load_jsonl(args.manifest)
    pools = load_candidate_pools(args.candidate_pools)
    grades_by_query = load_complete_qrels(args.qrels, pools)
    optimized_latency = load_optimized_latency(args.latency_results)

    records: list[dict[str, Any]] = []
    for manifest in manifests:
        case_id = str(manifest["case_id"])
        question = str(manifest["question"])
        plan = plan_query(question)
        decision = route_retrieval(
            plan,
            requested_mode="auto",
            latency_budget_ms=args.latency_budget_ms,
        )
        metrics_by_system = {
            system: ranking_metrics(
                manifest["systems"][system]["chunk_ids"],
                grades_by_query[case_id],
                args.relevant_threshold,
            )
            for system in SYSTEMS
        }
        selected_system = f"{decision.selected_mode}_hybrid"
        oracle_system = max(
            SYSTEMS,
            key=lambda system: (
                metrics_by_system[system]["ndcg_at_10"],
                metrics_by_system[system]["recall_at_10"],
                system == "direct_hybrid",
            ),
        )
        latency_by_system = {
            system: system_latency_seconds(manifest, system, optimized_latency)
            for system in SYSTEMS
        }
        records.append(
            {
                "case_id": case_id,
                "question": question,
                "decision": decision.as_dict(),
                "selected_system": selected_system,
                "oracle_system": oracle_system,
                "oracle_match": selected_system == oracle_system,
                "selected_seconds": latency_by_system[selected_system],
                "metrics": metrics_by_system,
                "latency_seconds": latency_by_system,
            }
        )
        print(
            f"{case_id}: {selected_system} oracle={oracle_system} "
            f"nDCG={metrics_by_system[selected_system]['ndcg_at_10']:.3f} "
            f"seconds={latency_by_system[selected_system]:.2f}",
            flush=True,
        )

    summary = build_summary(args, records, bool(optimized_latency))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "results.jsonl", records)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(
        summary_markdown(summary, records),
        encoding="utf-8",
    )
    print("\n=== Auto Routing Summary ===")
    for row in summary["systems"]:
        print(
            f"{row['system']}: R@10={row['avg_recall_at_10']:.3f} "
            f"MRR={row['mrr_at_10']:.3f} nDCG={row['avg_ndcg_at_10']:.3f} "
            f"median={row['median_seconds']:.2f}s"
        )
    print(
        f"oracle agreement: {summary['oracle_agreement_cases']}/{summary['cases']} "
        f"({summary['oracle_agreement_rate']:.1%})"
    )


def ranking_metrics(
    chunk_ids: list[str],
    grades: dict[str, int],
    relevant_threshold: int,
) -> dict[str, float]:
    at_10 = evaluate_retrieval_ranking(
        [str(chunk_id) for chunk_id in chunk_ids],
        grades,
        k=10,
        relevant_threshold=relevant_threshold,
    )
    return {
        "recall_at_10": float(at_10["recall_at_k"]),
        "precision_at_10": float(at_10["precision_at_k"]),
        "mrr_at_10": float(at_10["reciprocal_rank"]),
        "ndcg_at_10": float(at_10["ndcg_at_k"]),
    }


def load_optimized_latency(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    return {
        (str(row["case_id"]), str(row["system"])): float(row["optimized_seconds"])
        for row in load_jsonl(path)
    }


def system_latency_seconds(
    manifest: dict[str, Any],
    system: str,
    optimized_latency: dict[tuple[str, str], float],
) -> float:
    key = (str(manifest["case_id"]), system)
    if system.startswith("planned_") and key in optimized_latency:
        return optimized_latency[key]
    return float(manifest["systems"][system]["seconds"])


def build_summary(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    used_optimized_planned_latency: bool,
) -> dict[str, Any]:
    rows = []
    for system in ("direct_hybrid", "planned_hybrid", "auto"):
        metrics = []
        seconds = []
        for record in records:
            selected = record["selected_system"] if system == "auto" else system
            metrics.append(record["metrics"][selected])
            seconds.append(float(record["latency_seconds"][selected]))
        rows.append(
            {
                "system": system,
                "avg_recall_at_10": mean(metrics, "recall_at_10"),
                "avg_precision_at_10": mean(metrics, "precision_at_10"),
                "mrr_at_10": mean(metrics, "mrr_at_10"),
                "avg_ndcg_at_10": mean(metrics, "ndcg_at_10"),
                "avg_seconds": round(statistics.mean(seconds), 4),
                "median_seconds": round(statistics.median(seconds), 4),
            }
        )
    oracle_agreement = sum(1 for record in records if record["oracle_match"])
    route_counts = {
        mode: sum(1 for record in records if record["decision"]["selected_mode"] == mode)
        for mode in ("direct", "planned")
    }
    is_holdout = args.evaluation_role == "holdout"
    return {
        "evaluation_role": args.evaluation_role,
        "cases": len(records),
        "qrels": portable_path(args.qrels),
        "relevant_threshold": args.relevant_threshold,
        "latency_budget_ms": args.latency_budget_ms,
        "used_optimized_planned_latency": used_optimized_planned_latency,
        "route_counts": route_counts,
        "oracle_agreement_cases": oracle_agreement,
        "oracle_agreement_rate": round(oracle_agreement / len(records), 4),
        "systems": rows,
        "boundary": (
            "Questions were frozen before candidate generation and were not used to tune the current "
            "routing threshold. LLM relevance labels are an independent holdout signal, but a future "
            "human audit is still required."
            if is_holdout
            else "These queries informed the initial routing threshold, so oracle agreement is a "
            "calibration result rather than evidence of generalization."
        ),
    }


def mean(rows: list[dict[str, float]], key: str) -> float:
    return round(statistics.mean(row[key] for row in rows), 4)


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        (
            "# Automatic Retrieval Routing Holdout"
            if summary["evaluation_role"] == "holdout"
            else "# Automatic Retrieval Routing Calibration"
        ),
        "",
        "## Setup",
        "",
        f"- cases: {summary['cases']}",
        f"- evaluation role: {summary['evaluation_role']}",
        f"- qrels: {summary['qrels']}",
        f"- latency budget: {summary['latency_budget_ms']} ms",
        f"- route counts: {summary['route_counts']}",
        f"- optimized planned latency used: {summary['used_optimized_planned_latency']}",
        "",
        "## Results",
        "",
        "| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["systems"]:
        lines.append(
            f"| {row['system']} | {row['avg_recall_at_10']:.3f} | "
            f"{row['avg_precision_at_10']:.3f} | {row['mrr_at_10']:.3f} | "
            f"{row['avg_ndcg_at_10']:.3f} | {row['median_seconds']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Oracle agreement: {summary['oracle_agreement_cases']}/{summary['cases']} "
            f"({summary['oracle_agreement_rate']:.1%}).",
            "",
            "## Per-case Decisions",
            "",
            "| case | selected | oracle | score | nDCG direct/planned | seconds | reasons |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in records:
        direct = row["metrics"]["direct_hybrid"]["ndcg_at_10"]
        planned = row["metrics"]["planned_hybrid"]["ndcg_at_10"]
        decision = row["decision"]
        lines.append(
            f"| {row['case_id']} | {row['selected_system']} | {row['oracle_system']} | "
            f"{decision['complexity_score']} | {direct:.3f} / {planned:.3f} | "
            f"{row['selected_seconds']:.2f} | {', '.join(decision['reasons'])} |"
        )
    lines.extend(["", "## Interpretation Boundary", "", summary["boundary"], ""])
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
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
