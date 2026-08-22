import unittest

from src.retrieval_metrics import evaluate_retrieval_ranking


class RetrievalMetricsTest(unittest.TestCase):
    def test_perfect_graded_ranking(self) -> None:
        metrics = evaluate_retrieval_ranking(
            ["direct", "supporting", "related", "noise"],
            {"direct": 3, "supporting": 2, "related": 1, "noise": 0},
            k=2,
        )

        self.assertEqual(1.0, metrics["recall_at_k"])
        self.assertEqual(1.0, metrics["precision_at_k"])
        self.assertEqual(1.0, metrics["reciprocal_rank"])
        self.assertEqual(1.0, metrics["ndcg_at_k"])

    def test_metrics_penalize_late_relevant_evidence(self) -> None:
        metrics = evaluate_retrieval_ranking(
            ["related", "noise", "supporting", "direct"],
            {"direct": 3, "supporting": 2, "related": 1, "noise": 0},
            k=3,
        )

        self.assertEqual(0.5, metrics["recall_at_k"])
        self.assertEqual(0.3333, metrics["precision_at_k"])
        self.assertEqual(0.3333, metrics["reciprocal_rank"])
        self.assertLess(metrics["ndcg_at_k"], 0.5)

    def test_unjudged_ranked_chunk_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "without qrels"):
            evaluate_retrieval_ranking(["missing"], {"known": 3}, k=1)

    def test_query_without_relevant_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no chunks"):
            evaluate_retrieval_ranking(["noise"], {"noise": 1}, k=1)


if __name__ == "__main__":
    unittest.main()
