from __future__ import annotations

import unittest

from src.query_planning import QueryAspect, QueryPlan
from src.retrieval_routing import route_retrieval


class RetrievalRoutingTest(unittest.TestCase):
    def test_multiple_aspects_use_planned_retrieval(self) -> None:
        decision = route_retrieval(make_plan(aspect_count=2, sub_query_count=5))

        self.assertEqual("planned", decision.selected_mode)
        self.assertIn("multiple_answer_aspects", decision.reasons)

    def test_narrow_specific_query_uses_direct_retrieval(self) -> None:
        decision = route_retrieval(
            make_plan(aspect_count=1, sub_query_count=3, category_count=1, confidence=0.9)
        )

        self.assertEqual("direct", decision.selected_mode)
        self.assertIn("simple_or_specific_query", decision.reasons)

    def test_multi_category_expansion_without_aspects_uses_planned(self) -> None:
        decision = route_retrieval(
            make_plan(aspect_count=0, sub_query_count=3, category_count=2, confidence=0.8)
        )

        self.assertEqual("planned", decision.selected_mode)

    def test_low_latency_budget_forces_auto_route_to_direct(self) -> None:
        decision = route_retrieval(
            make_plan(aspect_count=3, sub_query_count=9),
            latency_budget_ms=3000,
        )

        self.assertEqual("direct", decision.selected_mode)
        self.assertIn("estimated_planned_latency_exceeds_budget", decision.reasons)

    def test_explicit_mode_is_preserved(self) -> None:
        decision = route_retrieval(
            make_plan(aspect_count=3, sub_query_count=9),
            requested_mode="direct",
        )

        self.assertEqual("direct", decision.selected_mode)
        self.assertEqual("forced_by_user", decision.reasons[0])


def make_plan(
    *,
    aspect_count: int,
    sub_query_count: int,
    category_count: int = 2,
    confidence: float = 0.5,
) -> QueryPlan:
    return QueryPlan(
        original_query="question",
        rewritten_query="question",
        intent="answer",
        category_filters=[f"category-{index}" for index in range(category_count)],
        sub_queries=[f"query-{index}" for index in range(sub_query_count)],
        aspects=[
            QueryAspect(name=f"aspect-{index}", question=f"aspect question {index}")
            for index in range(aspect_count)
        ],
        confidence=confidence,
    )


if __name__ == "__main__":
    unittest.main()
