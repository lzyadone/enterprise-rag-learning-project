"""Standard retrieval metrics over graded relevance judgments."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def evaluate_retrieval_ranking(
    ranked_chunk_ids: Sequence[str],
    relevance_grades: Mapping[str, int | float],
    *,
    k: int,
    relevant_threshold: int = 2,
) -> dict[str, int | float]:
    """Evaluate one ranking against a fully judged fixed candidate pool."""
    if k <= 0:
        raise ValueError("k must be positive")
    if len(set(ranked_chunk_ids)) != len(ranked_chunk_ids):
        raise ValueError("ranked_chunk_ids contains duplicates")
    missing = [chunk_id for chunk_id in ranked_chunk_ids if chunk_id not in relevance_grades]
    if missing:
        raise ValueError(f"ranking contains chunks without qrels: {missing}")

    relevant_ids = {
        chunk_id
        for chunk_id, grade in relevance_grades.items()
        if float(grade) >= relevant_threshold
    }
    if not relevant_ids:
        raise ValueError(
            f"qrels contain no chunks with relevance >= {relevant_threshold}"
        )

    selected = list(ranked_chunk_ids[:k])
    retrieved_relevant = sum(1 for chunk_id in selected if chunk_id in relevant_ids)
    first_relevant_rank = next(
        (rank for rank, chunk_id in enumerate(selected, start=1) if chunk_id in relevant_ids),
        None,
    )
    ranked_grades = [float(relevance_grades[chunk_id]) for chunk_id in selected]
    ideal_grades = sorted(
        (float(grade) for grade in relevance_grades.values()),
        reverse=True,
    )[:k]

    return {
        "relevant_in_pool": len(relevant_ids),
        "relevant_retrieved": retrieved_relevant,
        "recall_at_k": round(retrieved_relevant / len(relevant_ids), 4),
        "precision_at_k": round(retrieved_relevant / k, 4),
        "hit_at_k": int(retrieved_relevant > 0),
        "first_relevant_rank": first_relevant_rank or 0,
        "reciprocal_rank": round(1.0 / first_relevant_rank, 4) if first_relevant_rank else 0.0,
        "ndcg_at_k": round(normalized_discounted_cumulative_gain(ranked_grades, ideal_grades), 4),
        "top1_relevance_grade": round(ranked_grades[0], 4) if ranked_grades else 0.0,
    }


def normalized_discounted_cumulative_gain(
    ranked_grades: Sequence[float],
    ideal_grades: Sequence[float],
) -> float:
    ideal = discounted_cumulative_gain(ideal_grades)
    return discounted_cumulative_gain(ranked_grades) / ideal if ideal > 0 else 0.0


def discounted_cumulative_gain(grades: Sequence[float]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades, start=1)
    )
