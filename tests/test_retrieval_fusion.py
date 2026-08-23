import unittest
from unittest.mock import patch

from src.query_planning import QueryAspect, QueryPlan
from src.retrieval import (
    PLANNED_ANCHOR_WEIGHT,
    PLANNED_FILTERED_EXPANSION_WEIGHT,
    PLANNED_GLOBAL_EXPANSION_WEIGHT,
    RetrievedChunk,
    apply_plan_boosts,
    anchored_planned_retrieve,
    anchored_run_weights,
    build_anchored_run_specs,
    reciprocal_rank_fusion,
    select_with_bounded_plan_coverage,
)


def chunk(
    chunk_id: str,
    *,
    category: str = "",
    aspect: str | None = None,
    distance: float = 0.5,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document=chunk_id,
        metadata={"category": category, "source_id": chunk_id},
        distance=distance,
        score=0.0,
        source_query="query",
        category_filter=None,
        aspect=aspect,
        rank=1,
    )


class RetrievalFusionTest(unittest.TestCase):
    def test_plan_boost_respects_rrf_scale_cap(self) -> None:
        plan = QueryPlan(
            original_query="q",
            rewritten_query="q",
            intent="answer",
            aspects=[QueryAspect("evaluation", "q", categories=["evaluation"])],
        )
        candidate = chunk("evaluation", category="evaluation", aspect="evaluation")
        candidate.document = "faithfulness groundedness correctness relevance badcase evaluation"
        candidate.score = 0.02

        boosted = apply_plan_boosts([candidate], plan, max_boost=0.006)

        self.assertLessEqual(boosted[0].score, 0.026)

    def test_anchor_keeps_direct_candidate_window_when_expansions_grow(self) -> None:
        plan = QueryPlan(
            original_query="original",
            rewritten_query="original",
            intent="answer",
            category_filters=["retrieval"],
            sub_queries=["original", "expanded"],
            aspects=[QueryAspect("techniques", "expanded", categories=["retrieval"])],
        )
        call_index = 0

        def fake_retrieve(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_index
            call_index += 1
            return [chunk(f"chunk-{call_index}")]

        with (
            patch("src.retrieval.embed_query", return_value=[0.1]),
            patch("src.retrieval.retrieve_with_strategy", side_effect=fake_retrieve) as retrieve,
        ):
            anchored_planned_retrieve(
                None,  # type: ignore[arg-type]
                "original",
                "bge-m3",
                "http://127.0.0.1:11434",
                top_k=10,
                candidate_k=10,
                retrieval_strategy="hybrid",
                query_plan=plan,
            )

        first_call = retrieve.call_args_list[0].kwargs
        self.assertEqual(10, first_call["top_k"])
        self.assertEqual(10, first_call["candidate_k"])
        self.assertTrue(
            any(call.kwargs["top_k"] == 20 for call in retrieve.call_args_list[1:])
        )

    def test_weighted_rrf_preserves_anchor_priority(self) -> None:
        ranked = reciprocal_rank_fusion(
            [[chunk("anchor", distance=0.8)], [chunk("expansion", distance=0.1)]],
            weights=[2.0, 0.25],
        )

        self.assertEqual(["anchor", "expansion"], [item.chunk_id for item in ranked])

    def test_weighted_rrf_rejects_invalid_weights(self) -> None:
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[chunk("a")]], weights=[])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[chunk("a")]], weights=[-1.0])

    def test_anchored_specs_deduplicate_query_category_pairs(self) -> None:
        plan = QueryPlan(
            original_query="original",
            rewritten_query="original",
            intent="answer",
            category_filters=["retrieval"],
            sub_queries=["original", "expanded", "expanded"],
        )

        specs = build_anchored_run_specs("original", plan)
        keys = [(spec.query.casefold(), spec.category) for spec in specs]

        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual("anchor", specs[0].group)

    def test_expansion_weight_is_bounded_independent_of_fanout(self) -> None:
        plan = QueryPlan(
            original_query="original",
            rewritten_query="original",
            intent="answer",
            category_filters=["retrieval", "evaluation"],
            sub_queries=["expanded one", "expanded two", "expanded three"],
        )
        specs = build_anchored_run_specs("original", plan)
        weights = anchored_run_weights(specs)
        totals = {
            group: sum(weight for spec, weight in zip(specs, weights) if spec.group == group)
            for group in {spec.group for spec in specs}
        }

        self.assertAlmostEqual(PLANNED_ANCHOR_WEIGHT, totals["anchor"])
        self.assertAlmostEqual(PLANNED_GLOBAL_EXPANSION_WEIGHT, totals["global_expansion"])
        self.assertAlmostEqual(PLANNED_FILTERED_EXPANSION_WEIGHT, totals["filtered_expansion"])

    def test_bounded_coverage_only_promotes_configured_slots(self) -> None:
        plan = QueryPlan(
            original_query="q",
            rewritten_query="q",
            intent="answer",
            aspects=[
                QueryAspect("first", "q1", categories=["chunking"]),
                QueryAspect("second", "q2", categories=["evaluation"]),
            ],
        )
        ranked = [
            chunk("best", category="RAG overview"),
            chunk("plain", category="retrieval"),
            chunk("first", category="chunking", aspect="first"),
            chunk("second", category="evaluation", aspect="second"),
        ]

        selected = select_with_bounded_plan_coverage(
            ranked,
            4,
            plan,
            max_coverage_slots=1,
        )

        self.assertEqual(["best", "first", "plain", "second"], [item.chunk_id for item in selected])


if __name__ == "__main__":
    unittest.main()
