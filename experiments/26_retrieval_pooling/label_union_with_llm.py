"""Create complete, consistently model-labeled qrels for the retrieval union pool."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deepseek_client import DEFAULT_MODEL  # noqa: E402
from src.relevance_audit import agreement_summary  # noqa: E402
from src.relevance_judge import judge_pool  # noqa: E402
from src.retrieval_judgments import JudgmentStore, load_candidate_pools  # noqa: E402


DEFAULT_CANDIDATE_POOLS = PROJECT_ROOT / "eval" / "retrieval_union_v1" / "candidate_pools.jsonl"
DEFAULT_SEED_AUDIT = (
    PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_v1" / "llm_audit.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_union_v1" / "qrels_llm.jsonl"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_union_v1" / "qrels_llm_summary.json"
)
DEFAULT_HUMAN_QRELS = (
    PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_v1" / "qrels.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label the retrieval union pool with DeepSeek.")
    parser.add_argument("--candidate-pools", type=Path, default=DEFAULT_CANDIDATE_POOLS)
    parser.add_argument("--seed-audit", type=Path, default=DEFAULT_SEED_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--human-qrels", type=Path, default=DEFAULT_HUMAN_QRELS)
    parser.add_argument(
        "--skip-human-audit",
        action="store_true",
        help="Skip agreement calculation when the fresh benchmark has no human labels yet.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-document-chars", type=int, default=3200)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--rejudge-all",
        action="store_true",
        help="Do not seed old audit labels; judge every candidate in the shuffled union pool.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    pools = load_candidate_pools(args.candidate_pools)
    pair_order = [
        (str(pool["case_id"]), str(candidate["chunk_id"]))
        for pool in pools
        for candidate in pool["candidates"]
    ]
    valid_pairs = set(pair_order)
    judgments = {} if args.force else load_qrels(args.output, valid_pairs)
    if not args.rejudge_all:
        seeded = seed_from_audit(args.seed_audit, args.model, valid_pairs)
        for pair, row in seeded.items():
            judgments.setdefault(pair, row)

    for index, pool in enumerate(pools, start=1):
        query_id = str(pool["case_id"])
        missing_candidates = [
            candidate
            for candidate in pool["candidates"]
            if (query_id, str(candidate["chunk_id"])) not in judgments
        ]
        if not missing_candidates:
            print(f"[{index}/{len(pools)}] {query_id}: reused", flush=True)
            continue
        labeled_for_query = 0
        for offset in range(0, len(missing_candidates), args.batch_size):
            batch = missing_candidates[offset : offset + args.batch_size]
            model_rows = judge_pool(
                {"question": pool["question"], "candidates": batch},
                model=args.model,
                max_document_chars=args.max_document_chars,
            )
            timestamp = datetime.now(UTC).isoformat(timespec="seconds")
            for model_row in model_rows:
                pair = (query_id, str(model_row["chunk_id"]))
                judgments[pair] = {
                    "query_id": query_id,
                    "chunk_id": model_row["chunk_id"],
                    "relevance": int(model_row["relevance"]),
                    "annotator": f"llm:{args.model}",
                    "note": f"LLM盲审：{model_row['reason']}",
                    "updated_at": timestamp,
                }
            labeled_for_query += len(model_rows)
            write_ordered_qrels(args.output, judgments, pair_order)
            print(
                f"[{index}/{len(pools)}] {query_id}: batch "
                f"{offset // args.batch_size + 1} labeled {len(model_rows)}",
                flush=True,
            )
        print(f"[{index}/{len(pools)}] {query_id}: total {labeled_for_query}", flush=True)

    write_ordered_qrels(args.output, judgments, pair_order)
    store = JudgmentStore(args.output, pools)
    progress = store.progress()
    if progress["labeled"] != progress["total"]:
        raise RuntimeError(f"LLM qrels are incomplete: {progress}")
    rows = store.rows()
    human_overlap = (
        {"available": False, **agreement_summary([])}
        if args.skip_human_audit
        else {"available": True, **compare_with_human(rows, args.human_qrels)}
    )
    total = progress["total"]
    summary = {
        "model": args.model,
        "total": progress["total"],
        "judging_mode": (
            "full_shuffled_union" if args.rejudge_all else "seeded_prior_audit"
        ),
        "label_distribution": {
            str(label): sum(1 for row in rows if int(row["relevance"]) == label)
            for label in range(4)
        },
        "human_overlap_audit": human_overlap,
        "provenance": (
            f"All {total} labels were judged in the deterministically shuffled union pool "
            "with retrieval ranks, scores, channels, and plans hidden. Human qrels remain separate."
            if args.rejudge_all
            else f"All {total} labels use the same blind DeepSeek judge. Eligible labels may be "
            "reused from the earlier LLM audit; human qrels remain separate."
        ),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"qrels: {portable_path(args.output)}")


def compare_with_human(
    model_rows: list[dict[str, Any]],
    human_qrels_path: Path,
) -> dict[str, Any]:
    human = {
        (str(row["query_id"]), str(row["chunk_id"])): row
        for row in load_jsonl(human_qrels_path)
    }
    comparisons = []
    for row in model_rows:
        pair = (str(row["query_id"]), str(row["chunk_id"]))
        if pair not in human:
            continue
        comparisons.append(
            {
                "human_relevance": int(human[pair]["relevance"]),
                "llm_relevance": int(row["relevance"]),
            }
        )
    return agreement_summary(comparisons)


def seed_from_audit(
    path: Path,
    model: str,
    valid_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    seeded: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(path):
        if str(row.get("model") or "") != model:
            raise ValueError(f"seed audit model does not match requested model: {row.get('model')}")
        pair = (str(row["query_id"]), str(row["chunk_id"]))
        if pair not in valid_pairs:
            continue
        seeded[pair] = {
            "query_id": pair[0],
            "chunk_id": pair[1],
            "relevance": int(row["llm_relevance"]),
            "annotator": f"llm:{model}",
            "note": f"LLM盲审：{row.get('llm_reason') or ''}",
            "updated_at": "",
        }
    return seeded


def load_qrels(
    path: Path,
    valid_pairs: set[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    judgments: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(path):
        pair = (str(row["query_id"]), str(row["chunk_id"]))
        if pair not in valid_pairs:
            raise ValueError(f"existing LLM qrels contain an unknown pair: {pair}")
        judgments[pair] = row
    return judgments


def write_ordered_qrels(
    path: Path,
    judgments: dict[tuple[str, str], dict[str, Any]],
    pair_order: list[tuple[str, str]],
) -> None:
    write_jsonl(path, [judgments[pair] for pair in pair_order if pair in judgments])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
