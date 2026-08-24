"""Analyze current direct/planned routing on a natural-query development set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.query_planning import plan_query  # noqa: E402
from src.retrieval_routing import route_retrieval  # noqa: E402


DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "rag_natural_query_dev_v1.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "natural_query_dev_v1" / "routing_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze routing on natural user questions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latency-budget-ms", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    rows = load_jsonl(args.dataset)
    records = []
    for row in rows:
        question = str(row["question"])
        plan = plan_query(question)
        decision = route_retrieval(
            plan,
            requested_mode="auto",
            latency_budget_ms=args.latency_budget_ms,
        )
        intended = str(row["intended_route"])
        records.append(
            {
                "case_id": str(row["id"]),
                "question": question,
                "persona": row["persona"],
                "stratum": row["stratum"],
                "intended_route": intended,
                "selected_route": decision.selected_mode,
                "route_match": intended == decision.selected_mode,
                "decision": decision.as_dict(),
                "plan": plan.as_dict(),
            }
        )
        print(
            f"{row['id']}: intended={intended} selected={decision.selected_mode} "
            f"score={decision.complexity_score}",
            flush=True,
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


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {
        intended: {
            selected: sum(
                1
                for row in records
                if row["intended_route"] == intended and row["selected_route"] == selected
            )
            for selected in ("direct", "planned")
        }
        for intended in ("direct", "planned")
    }
    matched = sum(1 for row in records if row["route_match"])
    over_routed = [row["case_id"] for row in records if row["intended_route"] == "direct" and row["selected_route"] == "planned"]
    under_routed = [row["case_id"] for row in records if row["intended_route"] == "planned" and row["selected_route"] == "direct"]
    reason_counts = Counter(
        reason
        for row in records
        for reason in row["decision"]["reasons"]
        if reason not in {"complexity_threshold_reached", "simple_or_specific_query"}
    )
    return {
        "dataset": portable_path(args.dataset),
        "cases": len(records),
        "latency_budget_ms": args.latency_budget_ms,
        "intended_route_counts": dict(Counter(row["intended_route"] for row in records)),
        "selected_route_counts": dict(Counter(row["selected_route"] for row in records)),
        "confusion": confusion,
        "route_matches": matched,
        "route_match_rate": round(matched / len(records), 4),
        "direct_retention_rate": round(confusion["direct"]["direct"] / max(1, sum(confusion["direct"].values())), 4),
        "planned_detection_rate": round(confusion["planned"]["planned"] / max(1, sum(confusion["planned"].values())), 4),
        "over_routed_case_ids": over_routed,
        "under_routed_case_ids": under_routed,
        "reason_counts": dict(reason_counts),
        "boundary": (
            "Agreement measures whether the rule router recognizes the generator's intended complexity. "
            "It does not show which retrieval path has better evidence quality; that requires pooled qrels."
        ),
    }


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    confusion = summary["confusion"]
    lines = [
        "# Natural Query Routing Development Analysis",
        "",
        f"- cases: {summary['cases']}",
        f"- intended routes: {summary['intended_route_counts']}",
        f"- selected routes: {summary['selected_route_counts']}",
        f"- route intent agreement: {summary['route_matches']}/{summary['cases']} "
        f"({summary['route_match_rate']:.1%})",
        f"- direct retention: {summary['direct_retention_rate']:.1%}",
        f"- planned detection: {summary['planned_detection_rate']:.1%}",
        "",
        "## Confusion Matrix",
        "",
        "| intended / selected | direct | planned |",
        "|---|---:|---:|",
        f"| direct | {confusion['direct']['direct']} | {confusion['direct']['planned']} |",
        f"| planned | {confusion['planned']['direct']} | {confusion['planned']['planned']} |",
        "",
        "## Mismatches",
        "",
        "| case | intended | selected | score | reasons | question |",
        "|---|---|---|---:|---|---|",
    ]
    for row in records:
        if row["route_match"]:
            continue
        decision = row["decision"]
        lines.append(
            f"| {row['case_id']} | {row['intended_route']} | {row['selected_route']} | "
            f"{decision['complexity_score']} | {', '.join(decision['reasons'])} | "
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
