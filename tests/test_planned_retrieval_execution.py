from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.query_planning import QueryAspect, QueryPlan
from src.retrieval import (
    PlannedRunRequest,
    PlannedRunSpec,
    RetrievedChunk,
    anchored_planned_retrieve,
    execute_planned_run_requests,
    rerank_planned_candidates,
)
from src.retrieval_cache import clear_retrieval_caches, retrieval_cache_info


class PlannedRetrievalExecutionTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_retrieval_caches()

    def tearDown(self) -> None:
        clear_retrieval_caches()

    def test_parallel_runs_overlap_and_preserve_request_order(self) -> None:
        requests = [request(f"query-{index}") for index in range(4)]
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_retrieve(_collection, query, *_args, **_kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.04)
            with lock:
                active -= 1
            return [chunk(query)]

        with patch("src.retrieval.retrieve_with_strategy", side_effect=fake_retrieve):
            runs = execute_requests(requests, parallel=True, namespace="version-a")

        self.assertGreaterEqual(max_active, 2)
        self.assertEqual(
            [f"query-{index}" for index in range(4)],
            [run[0].chunk_id for run in runs],
        )

    def test_candidate_cache_reuses_deep_copies(self) -> None:
        calls = 0

        def fake_retrieve(_collection, query, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            return [chunk(query, document="original")]

        with patch("src.retrieval.retrieve_with_strategy", side_effect=fake_retrieve):
            first = execute_requests([request("same")], parallel=False, namespace="version-a")
            first[0][0].document = "mutated"
            second = execute_requests([request("same")], parallel=False, namespace="version-a")
            execute_requests([request("same")], parallel=False, namespace="version-b")

        self.assertEqual(2, calls)
        self.assertEqual("original", second[0][0].document)
        self.assertEqual(1, retrieval_cache_info()["candidates"]["hits"])

    def test_rerank_cache_reuses_result_without_sharing_mutation(self) -> None:
        calls = 0

        def fake_rerank(_query, candidates, _top_k, _mode):
            nonlocal calls
            calls += 1
            candidates[0].score = 9.0
            return candidates

        candidates = [chunk("one")]
        with patch("src.retrieval.rerank", side_effect=fake_rerank):
            first = rerank_planned_candidates(
                "query", candidates, "lexical", "bge-m3", "http://host", "version-a", True
            )
            first[0].score = -1.0
            second = rerank_planned_candidates(
                "query",
                [chunk("one")],
                "lexical",
                "bge-m3",
                "http://host",
                "version-a",
                True,
            )

        self.assertEqual(1, calls)
        self.assertEqual(9.0, second[0].score)
        self.assertEqual(1, retrieval_cache_info()["rerank"]["hits"])

    def test_parallel_and_serial_anchored_results_are_identical(self) -> None:
        def make_plan() -> QueryPlan:
            return QueryPlan(
                original_query="original",
                rewritten_query="original",
                intent="answer",
                category_filters=["retrieval", "evaluation"],
                aspects=[
                    QueryAspect("first", "first", "first query", ["retrieval"]),
                    QueryAspect("second", "second", "second query", ["evaluation"]),
                ],
            )

        def fake_retrieve(_collection, query, *_args, **kwargs):
            category = kwargs.get("category") or "all"
            return [chunk(f"{query}:{category}"), chunk("shared")]

        with (
            patch("src.retrieval.embed_query", return_value=[0.1]),
            patch("src.retrieval.retrieve_with_strategy", side_effect=fake_retrieve),
        ):
            _, serial = anchored_planned_retrieve(
                None,  # type: ignore[arg-type]
                "original",
                "bge-m3",
                "http://host",
                top_k=7,
                candidate_k=8,
                retrieval_strategy="hybrid",
                query_plan=make_plan(),
                planning_mode="conservative",
                parallel_runs=False,
                use_candidate_cache=False,
                use_rerank_cache=False,
            )
            _, parallel = anchored_planned_retrieve(
                None,  # type: ignore[arg-type]
                "original",
                "bge-m3",
                "http://host",
                top_k=7,
                candidate_k=8,
                retrieval_strategy="hybrid",
                query_plan=make_plan(),
                planning_mode="conservative",
                parallel_runs=True,
                max_workers=4,
                use_candidate_cache=False,
                use_rerank_cache=False,
            )

        self.assertEqual(
            [item.chunk_id for item in serial],
            [item.chunk_id for item in parallel],
        )
        self.assertEqual(
            [item.score for item in serial],
            [item.score for item in parallel],
        )

    def test_embedding_cache_diagnostic_bypass_also_bypasses_retrieval_caches(self) -> None:
        calls = 0

        def fake_retrieve(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return [chunk(f"call-{calls}")]

        def make_plan() -> QueryPlan:
            return QueryPlan("query", "query", "answer")

        with patch("src.retrieval.retrieve_with_strategy", side_effect=fake_retrieve):
            for _ in range(2):
                anchored_planned_retrieve(
                    None,  # type: ignore[arg-type]
                    "query",
                    "bge-m3",
                    "http://host",
                    query_plan=make_plan(),
                    planning_mode="conservative",
                    reuse_query_embeddings=False,
                    cache_namespace="version-a",
                )

        self.assertEqual(2, calls)
        self.assertEqual(0, retrieval_cache_info()["candidates"]["size"])

    def test_invalid_worker_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_workers"):
            execute_requests([request("query")], parallel=True, namespace="version", max_workers=0)


def request(query: str) -> PlannedRunRequest:
    return PlannedRunRequest(
        spec=PlannedRunSpec(query, None, None, "global_expansion"),
        top_k=3,
        candidate_k=3,
    )


def chunk(chunk_id: str, document: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document=document or chunk_id,
        metadata={"source_id": chunk_id, "text_hash": f"hash-{chunk_id}"},
        distance=0.5,
        score=0.1,
        source_query="query",
        category_filter=None,
        aspect=None,
        rank=1,
        retrieval_channels=["dense"],
    )


def execute_requests(
    requests: list[PlannedRunRequest],
    *,
    parallel: bool,
    namespace: str,
    max_workers: int = 4,
) -> list[list[RetrievedChunk]]:
    return execute_planned_run_requests(
        requests,
        collection=None,  # type: ignore[arg-type]
        embedding_model="bge-m3",
        ollama_host="http://host",
        retrieval_strategy="hybrid",
        chunks_path=Path("chunks.jsonl"),
        embedding_by_query={request.spec.query: [0.1] for request in requests},
        reuse_query_embeddings=True,
        parallel_runs=parallel,
        max_workers=max_workers,
        use_candidate_cache=True,
        cache_namespace=namespace,
    )


if __name__ == "__main__":
    unittest.main()
