"""Candidate pooling helpers for retrieval-system evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


def pool_rankings(
    query_id: str,
    system_rankings: Mapping[str, Sequence[dict[str, Any]]],
    legacy_candidates: Sequence[dict[str, Any]] = (),
    *,
    seed: str = "rag-retrieval-union-v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate system rankings and hide rank with deterministic shuffling."""
    candidates_by_id: dict[str, dict[str, Any]] = {}
    provenance: dict[str, dict[str, Any]] = {}

    for system, candidates in system_rankings.items():
        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = candidate_id(candidate)
            candidates_by_id.setdefault(chunk_id, dict(candidate))
            item = provenance.setdefault(
                chunk_id,
                {"chunk_id": chunk_id, "system_ranks": {}, "legacy_judged": False},
            )
            item["system_ranks"][str(system)] = rank

    for candidate in legacy_candidates:
        chunk_id = candidate_id(candidate)
        candidates_by_id.setdefault(chunk_id, dict(candidate))
        item = provenance.setdefault(
            chunk_id,
            {"chunk_id": chunk_id, "system_ranks": {}, "legacy_judged": False},
        )
        item["legacy_judged"] = True

    ordered_ids = sorted(
        candidates_by_id,
        key=lambda chunk_id: hashlib.sha256(
            f"{seed}:{query_id}:{chunk_id}".encode("utf-8")
        ).hexdigest(),
    )
    return (
        [blind_candidate(candidates_by_id[chunk_id]) for chunk_id in ordered_ids],
        [provenance[chunk_id] for chunk_id in ordered_ids],
    )


def inherit_qrels(
    rows: Sequence[dict[str, Any]],
    valid_pair_order: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Keep existing judgments that still belong to the new pooled benchmark."""
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    valid_pairs = set(valid_pair_order)
    for row in rows:
        pair = (str(row.get("query_id") or ""), str(row.get("chunk_id") or ""))
        if pair not in valid_pairs:
            continue
        by_pair[pair] = dict(row)
    return [by_pair[pair] for pair in valid_pair_order if pair in by_pair]


def chunk_record_to_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """Convert a processed chunk row to the candidate schema used by the labeler."""
    chunk_id = candidate_id(record)
    document = str(record.get("text") or record.get("document") or "")
    if not document:
        raise ValueError(f"processed chunk has no text: {chunk_id}")
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"chunk_id", "text", "document"}
    }
    return {
        "chunk_id": chunk_id,
        "document": document,
        "metadata": metadata,
        "distance": 1.0,
        "score": 0.0,
        "source_query": "legacy human judgment",
        "category_filter": None,
        "aspect": None,
        "rank": 0,
        "rerank_score": 0.0,
        "rerank_reason": "",
        "retrieval_channels": [],
    }


def blind_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Remove retrieval-system signals before a candidate reaches an annotator."""
    blinded = dict(candidate)
    blinded.update(
        {
            "distance": 0.0,
            "score": None,
            "source_query": "",
            "category_filter": None,
            "aspect": None,
            "rank": None,
            "rerank_score": 0.0,
            "rerank_reason": "",
            "retrieval_channels": [],
        }
    )
    return blinded


def candidate_id(candidate: Mapping[str, Any]) -> str:
    chunk_id = str(candidate.get("chunk_id") or "").strip()
    if not chunk_id:
        raise ValueError("candidate is missing chunk_id")
    return chunk_id
