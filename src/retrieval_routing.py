"""Explainable routing between direct and planned retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.query_planning import QueryPlan


AUTO_PLANNED_SCORE_THRESHOLD = 3
AUTO_PLANNED_BASE_LATENCY_MS = 1500
AUTO_PLANNED_PER_QUERY_MS = 1000


@dataclass(frozen=True)
class RetrievalRouteDecision:
    requested_mode: str
    selected_mode: str
    latency_budget_ms: int
    estimated_planned_latency_ms: int
    complexity_score: int
    threshold: int
    reasons: tuple[str, ...]
    features: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "selected_mode": self.selected_mode,
            "latency_budget_ms": self.latency_budget_ms,
            "estimated_planned_latency_ms": self.estimated_planned_latency_ms,
            "complexity_score": self.complexity_score,
            "threshold": self.threshold,
            "reasons": list(self.reasons),
            "features": self.features,
        }


def route_retrieval(
    plan: QueryPlan,
    *,
    requested_mode: str = "auto",
    latency_budget_ms: int = 12000,
) -> RetrievalRouteDecision:
    """Choose a retrieval path from plan complexity and an explicit latency budget."""
    if requested_mode not in {"auto", "direct", "planned"}:
        raise ValueError("requested_mode must be auto, direct, or planned")
    if latency_budget_ms <= 0:
        raise ValueError("latency_budget_ms must be positive")

    features = plan_features(plan)
    score, complexity_reasons = score_plan_complexity(features)
    estimated_planned_latency_ms = estimate_planned_latency_ms(features)

    if requested_mode in {"direct", "planned"}:
        return RetrievalRouteDecision(
            requested_mode=requested_mode,
            selected_mode=requested_mode,
            latency_budget_ms=latency_budget_ms,
            estimated_planned_latency_ms=estimated_planned_latency_ms,
            complexity_score=score,
            threshold=AUTO_PLANNED_SCORE_THRESHOLD,
            reasons=("forced_by_user", *complexity_reasons),
            features=features,
        )

    reasons = list(complexity_reasons)
    if score >= AUTO_PLANNED_SCORE_THRESHOLD and estimated_planned_latency_ms <= latency_budget_ms:
        selected_mode = "planned"
        reasons.insert(0, "complexity_threshold_reached")
    elif score >= AUTO_PLANNED_SCORE_THRESHOLD:
        selected_mode = "direct"
        reasons.insert(0, "estimated_planned_latency_exceeds_budget")
    else:
        selected_mode = "direct"
        reasons.insert(0, "simple_or_specific_query")

    return RetrievalRouteDecision(
        requested_mode=requested_mode,
        selected_mode=selected_mode,
        latency_budget_ms=latency_budget_ms,
        estimated_planned_latency_ms=estimated_planned_latency_ms,
        complexity_score=score,
        threshold=AUTO_PLANNED_SCORE_THRESHOLD,
        reasons=tuple(reasons),
        features=features,
    )


def plan_features(plan: QueryPlan) -> dict[str, Any]:
    unique_queries = {
        query.strip().casefold()
        for query in (
            [aspect.search_query or aspect.question for aspect in plan.aspects]
            + plan.sub_queries
        )
        if query.strip()
    }
    return {
        "aspect_count": len(plan.aspects),
        "sub_query_count": len(plan.sub_queries),
        "category_count": len(plan.category_filters),
        "unique_retrieval_query_count": len(unique_queries),
        "confidence": float(plan.confidence),
        "intent": plan.intent,
    }


def score_plan_complexity(features: dict[str, Any]) -> tuple[int, tuple[str, ...]]:
    score = 0
    reasons: list[str] = []
    aspect_count = int(features["aspect_count"])
    sub_query_count = int(features["sub_query_count"])
    category_count = int(features["category_count"])
    confidence = float(features["confidence"])
    intent = str(features["intent"])

    if aspect_count >= 2:
        score += 4
        reasons.append("multiple_answer_aspects")
    elif aspect_count == 0 and sub_query_count >= 3 and category_count >= 2:
        score += 3
        reasons.append("multi_category_expansion_without_aspects")
    elif aspect_count == 1 and confidence < 0.25:
        score += 3
        reasons.append("aspect_detected_with_low_category_confidence")

    if intent == "compare" and category_count >= 2:
        score += 2
        reasons.append("cross_category_comparison")

    return score, tuple(reasons)


def estimate_planned_latency_ms(features: dict[str, Any]) -> int:
    """Estimate local planned latency from the calibrated embedding-query fanout."""
    query_count = max(1, int(features["unique_retrieval_query_count"]))
    return AUTO_PLANNED_BASE_LATENCY_MS + AUTO_PLANNED_PER_QUERY_MS * query_count
