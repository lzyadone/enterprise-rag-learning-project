"""Human relevance judgments for retrieval evaluation benchmarks."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


MIN_RELEVANCE = 0
MAX_RELEVANCE = 3


def load_candidate_pools(path: Path) -> list[dict[str, Any]]:
    """Load and validate retrieval candidate pools from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"candidate pool not found: {path}")

    pools: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            pool = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on candidate-pool line {line_number}: {exc}") from exc
        if not isinstance(pool, dict):
            raise ValueError(f"candidate-pool line {line_number} must be an object")

        case_id = str(pool.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"candidate-pool line {line_number} is missing case_id")
        if case_id in case_ids:
            raise ValueError(f"duplicate case_id in candidate pool: {case_id}")
        case_ids.add(case_id)

        plan = pool.get("plan") or {}
        question = str(plan.get("original_query") or pool.get("question") or "").strip()
        if not question:
            raise ValueError(f"candidate pool {case_id} is missing its question")

        candidates = pool.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"candidate pool {case_id} must contain candidates")
        chunk_ids: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError(f"candidate pool {case_id} contains a non-object candidate")
            chunk_id = str(candidate.get("chunk_id") or "").strip()
            if not chunk_id:
                raise ValueError(f"candidate pool {case_id} contains a candidate without chunk_id")
            if chunk_id in chunk_ids:
                raise ValueError(f"candidate pool {case_id} contains duplicate chunk_id {chunk_id}")
            chunk_ids.add(chunk_id)

        normalized = dict(pool)
        normalized["question"] = question
        pools.append(normalized)

    if not pools:
        raise ValueError(f"candidate pool is empty: {path}")
    return pools


class JudgmentStore:
    """Thread-safe JSONL store constrained to a fixed set of query/chunk pairs."""

    def __init__(self, path: Path, candidate_pools: list[dict[str, Any]]) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._pair_order: dict[tuple[str, str], int] = {}
        self._valid_pairs: set[tuple[str, str]] = set()
        order = 0
        for pool in candidate_pools:
            query_id = str(pool["case_id"])
            for candidate in pool["candidates"]:
                pair = (query_id, str(candidate["chunk_id"]))
                self._pair_order[pair] = order
                self._valid_pairs.add(pair)
                order += 1
        self._judgments = self._load()

    def _load(self) -> dict[tuple[str, str], dict[str, Any]]:
        if not self.path.exists():
            return {}
        judgments: dict[tuple[str, str], dict[str, Any]] = {}
        for line_number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on qrels line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"qrels line {line_number} must be an object")
            query_id = str(row.get("query_id") or "").strip()
            chunk_id = str(row.get("chunk_id") or "").strip()
            relevance = row.get("relevance")
            self._validate_pair(query_id, chunk_id)
            self._validate_relevance(relevance)
            pair = (query_id, chunk_id)
            if pair in judgments:
                raise ValueError(f"duplicate qrels pair on line {line_number}: {query_id}/{chunk_id}")
            judgments[pair] = self._normalize_row(row)
        return judgments

    def upsert(
        self,
        query_id: str,
        chunk_id: str,
        relevance: int,
        note: str = "",
        annotator: str = "human",
    ) -> dict[str, Any]:
        query_id = query_id.strip()
        chunk_id = chunk_id.strip()
        self._validate_pair(query_id, chunk_id)
        self._validate_relevance(relevance)
        row = {
            "query_id": query_id,
            "chunk_id": chunk_id,
            "relevance": relevance,
            "annotator": annotator.strip() or "human",
            "note": note.strip(),
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with self._lock:
            self._judgments[(query_id, chunk_id)] = row
            self._write()
        return dict(row)

    def delete(self, query_id: str, chunk_id: str) -> bool:
        pair = (query_id.strip(), chunk_id.strip())
        self._validate_pair(*pair)
        with self._lock:
            removed = self._judgments.pop(pair, None) is not None
            if removed:
                self._write()
            return removed

    def get(self, query_id: str, chunk_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._judgments.get((query_id, chunk_id))
            return dict(row) if row else None

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(row)
                for pair, row in sorted(
                    self._judgments.items(), key=lambda item: self._pair_order[item[0]]
                )
            ]

    def progress(self) -> dict[str, int]:
        with self._lock:
            return {
                "labeled": len(self._judgments),
                "total": len(self._valid_pairs),
            }

    def progress_for(self, query_id: str) -> dict[str, int]:
        total = sum(1 for pair in self._valid_pairs if pair[0] == query_id)
        with self._lock:
            labeled = sum(1 for pair in self._judgments if pair[0] == query_id)
        return {"labeled": labeled, "total": total}

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in self.rows()
        )
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)

    def _validate_pair(self, query_id: str, chunk_id: str) -> None:
        if (query_id, chunk_id) not in self._valid_pairs:
            raise ValueError(f"unknown query/chunk pair: {query_id}/{chunk_id}")

    @staticmethod
    def _validate_relevance(relevance: Any) -> None:
        if isinstance(relevance, bool) or not isinstance(relevance, int):
            raise ValueError("relevance must be an integer from 0 to 3")
        if not MIN_RELEVANCE <= relevance <= MAX_RELEVANCE:
            raise ValueError("relevance must be an integer from 0 to 3")

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "query_id": str(row["query_id"]),
            "chunk_id": str(row["chunk_id"]),
            "relevance": int(row["relevance"]),
            "annotator": str(row.get("annotator") or "human"),
            "note": str(row.get("note") or ""),
            "updated_at": str(row.get("updated_at") or ""),
        }
