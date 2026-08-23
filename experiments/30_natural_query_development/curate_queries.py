"""Prune same-route semantic duplicates from generated natural user questions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ollama_http import embed_texts, unload_embedding_model  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "eval" / "datasets" / "rag_natural_query_dev_v1_raw.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "datasets" / "rag_natural_query_dev_v1.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "eval" / "natural_query_dev_v1" / "curation_summary.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "eval" / "natural_query_dev_v1" / "curation_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate generated natural user questions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--same-route-threshold", type=float, default=0.86)
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    rows = load_jsonl(args.input)
    vectors = embed_texts(
        [str(row["question"]) for row in rows],
        args.embedding_model,
        args.ollama_host,
    )
    try:
        kept, removed = curate(rows, vectors, args.same_route_threshold)
    finally:
        unload_embedding_model(args.embedding_model, args.ollama_host)

    write_jsonl(args.output, kept)
    summary = {
        "input": portable_path(args.input),
        "output": portable_path(args.output),
        "embedding_model": args.embedding_model,
        "same_route_threshold": args.same_route_threshold,
        "raw_count": len(rows),
        "curated_count": len(kept),
        "removed_count": len(removed),
        "curated_route_counts": dict(Counter(str(row["intended_route"]) for row in kept)),
        "removed": removed,
        "boundary": (
            "Only later questions with a same-route cosine similarity at or above the threshold "
            "were removed. Similar cross-route pairs are retained as useful routing contrasts."
        ),
    }
    write_json(args.report, summary)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)
    args.report_md.write_text(summary_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def curate(
    rows: list[dict[str, Any]],
    vectors: list[list[float]],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) != len(vectors):
        raise ValueError("row and embedding counts must match")
    if not 0 < threshold < 1:
        raise ValueError("same-route-threshold must be between 0 and 1")
    kept: list[dict[str, Any]] = []
    kept_vectors: list[list[float]] = []
    removed = []
    for row, vector in zip(rows, vectors, strict=True):
        same_route = [
            (kept_row, kept_vector)
            for kept_row, kept_vector in zip(kept, kept_vectors, strict=True)
            if kept_row["intended_route"] == row["intended_route"]
        ]
        nearest = max(
            (
                (cosine_similarity(vector, kept_vector), kept_row)
                for kept_row, kept_vector in same_route
            ),
            key=lambda pair: pair[0],
            default=None,
        )
        if nearest and nearest[0] >= threshold:
            removed.append(
                {
                    "removed_id": row["id"],
                    "kept_id": nearest[1]["id"],
                    "intended_route": row["intended_route"],
                    "cosine_similarity": round(nearest[0], 4),
                }
            )
            continue
        kept.append(row)
        kept_vectors.append(vector)
    return kept, removed


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must be non-empty and have equal length")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Natural Query Curation",
        "",
        f"- raw questions: {summary['raw_count']}",
        f"- curated questions: {summary['curated_count']}",
        f"- removed same-route duplicates: {summary['removed_count']}",
        f"- route counts: {summary['curated_route_counts']}",
        f"- bge-m3 cosine threshold: {summary['same_route_threshold']}",
        "",
        "| removed | kept | route | cosine |",
        "|---|---|---|---:|",
    ]
    for row in summary["removed"]:
        lines.append(
            f"| {row['removed_id']} | {row['kept_id']} | {row['intended_route']} | "
            f"{row['cosine_similarity']:.4f} |"
        )
    lines.extend(["", "## Boundary", "", summary["boundary"], ""])
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
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
