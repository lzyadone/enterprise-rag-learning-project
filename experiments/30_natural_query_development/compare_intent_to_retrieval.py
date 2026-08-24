"""Compare generated route intent with observed direct/planned retrieval quality."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "rag_natural_query_dev_v1.jsonl"
DEFAULT_RETRIEVAL_RESULTS = (
    PROJECT_ROOT
    / "eval"
    / "natural_query_retrieval_dev_v1"
    / "routing_evaluation"
    / "results.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "eval" / "natural_query_dev_v1" / "retrieval_intent_analysis"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare intended and observed retrieval routes.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--retrieval-results", type=Path, default=DEFAULT_RETRIEVAL_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--direct-system", default="direct_hybrid")
    parser.add_argument("--planned-system", default="planned_v2_hybrid")
    parser.add_argument("--meaningful-margin", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    dataset = load_jsonl(args.dataset)
    retrieval_results = load_jsonl(args.retrieval_results)
    records = build_records(
        dataset,
        retrieval_results,
        direct_system=args.direct_system,
        planned_system=args.planned_system,
        meaningful_margin=args.meaningful_margin,
    )
    summary = build_summary(args, records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "results.jsonl", records)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(
        summary_markdown(summary, records),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_records(
    dataset: list[dict[str, Any]],
    retrieval_results: list[dict[str, Any]],
    *,
    direct_system: str,
    planned_system: str,
    meaningful_margin: float,
) -> list[dict[str, Any]]:
    if meaningful_margin < 0:
        raise ValueError("meaningful-margin must be non-negative")
    dataset_by_id = {str(row["id"]): row for row in dataset}
    results_by_id = {str(row["case_id"]): row for row in retrieval_results}
    if set(dataset_by_id) != set(results_by_id):
        missing_results = sorted(set(dataset_by_id) - set(results_by_id))
        unknown_results = sorted(set(results_by_id) - set(dataset_by_id))
        raise ValueError(
            f"dataset/results ids differ; missing_results={missing_results}, "
            f"unknown_results={unknown_results}"
        )

    records = []
    for dataset_row in dataset:
        case_id = str(dataset_row["id"])
        result = results_by_id[case_id]
        evaluable = bool(result.get("evaluable", True))
        direct_ndcg = float(result["metrics"][direct_system]["ndcg_at_10"])
        planned_ndcg = float(result["metrics"][planned_system]["ndcg_at_10"])
        delta = planned_ndcg - direct_ndcg
        if not evaluable:
            benefit_class = "coverage_gap"
        elif delta > meaningful_margin:
            benefit_class = "planned_better"
        elif delta < -meaningful_margin:
            benefit_class = "direct_better"
        else:
            benefit_class = "comparable"
        intended_route = str(dataset_row["intended_route"])
        intended_system = planned_system if intended_route == "planned" else direct_system
        records.append(
            {
                "case_id": case_id,
                "question": dataset_row["question"],
                "persona": dataset_row["persona"],
                "intended_route": intended_route,
                "intended_system": intended_system,
                "auto_system": result["selected_system"],
                "oracle_system": result["oracle_system"],
                "evaluable": evaluable,
                "intended_oracle_match": (
                    intended_system == result["oracle_system"] if evaluable else None
                ),
                "auto_oracle_match": bool(result["oracle_match"]) if evaluable else None,
                "direct_ndcg_at_10": direct_ndcg,
                "planned_ndcg_at_10": planned_ndcg,
                "planned_ndcg_delta": round(delta, 4),
                "benefit_class": benefit_class,
            }
        )
    return records


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable_records = [row for row in records if row.get("evaluable", True)]
    if not evaluable_records:
        raise ValueError("no evaluable retrieval records")
    intended_matches = sum(1 for row in evaluable_records if row["intended_oracle_match"])
    auto_matches = sum(1 for row in evaluable_records if row["auto_oracle_match"])
    by_intended = {}
    for route in ("direct", "planned"):
        all_route_rows = [row for row in records if row["intended_route"] == route]
        rows = [row for row in all_route_rows if row.get("evaluable", True)]
        by_intended[route] = {
            "cases": len(all_route_rows),
            "evaluable_cases": len(rows),
            "coverage_gap_cases": len(all_route_rows) - len(rows),
            "avg_direct_ndcg_at_10": mean(rows, "direct_ndcg_at_10"),
            "avg_planned_ndcg_at_10": mean(rows, "planned_ndcg_at_10"),
            "avg_planned_ndcg_delta": mean(rows, "planned_ndcg_delta"),
            "benefit_classes": dict(Counter(row["benefit_class"] for row in rows)),
        }
    confusion = {
        intended: {
            oracle: sum(
                1
                for row in evaluable_records
                if row["intended_route"] == intended
                and system_route(row["oracle_system"], args.planned_system) == oracle
            )
            for oracle in ("direct", "planned")
        }
        for intended in ("direct", "planned")
    }
    return {
        "dataset": portable_path(args.dataset),
        "retrieval_results": portable_path(args.retrieval_results),
        "cases": len(records),
        "evaluable_cases": len(evaluable_records),
        "coverage_gap_cases": len(records) - len(evaluable_records),
        "direct_system": args.direct_system,
        "planned_system": args.planned_system,
        "meaningful_margin": args.meaningful_margin,
        "benefit_classes": dict(Counter(row["benefit_class"] for row in records)),
        "intended_oracle_matches": intended_matches,
        "intended_oracle_match_rate": round(intended_matches / len(evaluable_records), 4),
        "auto_oracle_matches": auto_matches,
        "auto_oracle_match_rate": round(auto_matches / len(evaluable_records), 4),
        "intended_oracle_confusion": confusion,
        "by_intended_route": by_intended,
        "boundary": (
            "This is development analysis. LLM-generated intent labels and LLM relevance qrels may "
            "share model biases. Observed retrieval deltas can guide development but require a new "
            "independent holdout before any release claim."
        ),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(statistics.mean(float(row[key]) for row in rows), 4) if rows else 0.0


def system_route(system: str, planned_system: str) -> str:
    return "planned" if system == planned_system else "direct"


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    confusion = summary["intended_oracle_confusion"]
    lines = [
        "# Natural Query Intent vs Retrieval Quality",
        "",
        f"- cases: {summary['cases']}",
        f"- evaluable cases: {summary['evaluable_cases']}",
        f"- candidate-pool coverage gaps: {summary['coverage_gap_cases']}",
        f"- meaningful nDCG margin: {summary['meaningful_margin']}",
        f"- benefit classes: {summary['benefit_classes']}",
        f"- intended route / oracle agreement: {summary['intended_oracle_matches']}/"
        f"{summary['evaluable_cases']} ({summary['intended_oracle_match_rate']:.1%})",
        f"- current auto / oracle agreement: {summary['auto_oracle_matches']}/"
        f"{summary['evaluable_cases']} ({summary['auto_oracle_match_rate']:.1%})",
        "",
        "## Intended vs Oracle",
        "",
        "| intended / oracle | direct | planned |",
        "|---|---:|---:|",
        f"| direct | {confusion['direct']['direct']} | {confusion['direct']['planned']} |",
        f"| planned | {confusion['planned']['direct']} | {confusion['planned']['planned']} |",
        "",
        "## Quality by Intended Route",
        "",
        "| intended | cases | evaluable | gaps | direct nDCG | planned nDCG | planned delta | benefit classes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for route, row in summary["by_intended_route"].items():
        lines.append(
            f"| {route} | {row['cases']} | {row['evaluable_cases']} | "
            f"{row['coverage_gap_cases']} | {row['avg_direct_ndcg_at_10']:.3f} | "
            f"{row['avg_planned_ndcg_at_10']:.3f} | {row['avg_planned_ndcg_delta']:+.3f} | "
            f"{row['benefit_classes']} |"
        )
    lines.extend(
        [
            "",
            "## Largest Planned Deltas",
            "",
            "| case | intended | direct | planned | delta | class | question |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    ordered = sorted(
        (row for row in records if row.get("evaluable", True)),
        key=lambda row: float(row["planned_ndcg_delta"]),
        reverse=True,
    )
    for row in ordered[:8] + ordered[-8:]:
        lines.append(
            f"| {row['case_id']} | {row['intended_route']} | "
            f"{row['direct_ndcg_at_10']:.3f} | {row['planned_ndcg_at_10']:.3f} | "
            f"{row['planned_ndcg_delta']:+.3f} | {row['benefit_class']} | "
            f"{str(row['question']).replace('|', '/')} |"
        )
    lines.extend(["", "## Boundary", "", summary["boundary"], ""])
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
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
