import unittest

from src.retrieval_pooling import chunk_record_to_candidate, inherit_qrels, pool_rankings


class RetrievalPoolingTest(unittest.TestCase):
    def test_pool_deduplicates_and_preserves_provenance(self) -> None:
        pooled, provenance = pool_rankings(
            "q1",
            {
                "dense": [{"chunk_id": "a"}, {"chunk_id": "shared"}],
                "bm25": [{"chunk_id": "shared"}, {"chunk_id": "b"}],
            },
            [{"chunk_id": "legacy"}, {"chunk_id": "a"}],
        )

        self.assertEqual({"a", "shared", "b", "legacy"}, {row["chunk_id"] for row in pooled})
        by_id = {row["chunk_id"]: row for row in provenance}
        self.assertEqual({"dense": 2, "bm25": 1}, by_id["shared"]["system_ranks"])
        self.assertTrue(by_id["a"]["legacy_judged"])
        self.assertTrue(by_id["legacy"]["legacy_judged"])

    def test_pool_order_is_deterministic_and_query_specific(self) -> None:
        ranking = {"dense": [{"chunk_id": str(index)} for index in range(10)]}
        first, _ = pool_rankings("q1", ranking)
        repeated, _ = pool_rankings("q1", ranking)
        another_query, _ = pool_rankings("q2", ranking)

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, another_query)

    def test_pool_hides_retrieval_signals_from_annotator(self) -> None:
        pooled, _ = pool_rankings(
            "q1",
            {
                "dense": [
                    {
                        "chunk_id": "a",
                        "score": 0.99,
                        "rank": 1,
                        "retrieval_channels": ["dense"],
                        "source_query": "expanded query",
                        "aspect": "techniques",
                    }
                ]
            },
        )

        self.assertIsNone(pooled[0]["score"])
        self.assertIsNone(pooled[0]["rank"])
        self.assertEqual([], pooled[0]["retrieval_channels"])
        self.assertEqual("", pooled[0]["source_query"])
        self.assertIsNone(pooled[0]["aspect"])

    def test_inherit_qrels_uses_new_pool_order_and_latest_duplicate(self) -> None:
        rows = [
            {"query_id": "q1", "chunk_id": "a", "relevance": 2},
            {"query_id": "q1", "chunk_id": "a", "relevance": 3},
            {"query_id": "q1", "chunk_id": "removed", "relevance": 1},
        ]

        inherited = inherit_qrels(rows, [("q1", "b"), ("q1", "a")])

        self.assertEqual(["a"], [row["chunk_id"] for row in inherited])
        self.assertEqual(3, inherited[0]["relevance"])

    def test_processed_chunk_conversion_keeps_metadata(self) -> None:
        candidate = chunk_record_to_candidate(
            {"chunk_id": "doc::1", "text": "evidence", "title": "Doc", "category": "retrieval"}
        )

        self.assertEqual("evidence", candidate["document"])
        self.assertEqual("Doc", candidate["metadata"]["title"])
        self.assertEqual("retrieval", candidate["metadata"]["category"])


if __name__ == "__main__":
    unittest.main()
