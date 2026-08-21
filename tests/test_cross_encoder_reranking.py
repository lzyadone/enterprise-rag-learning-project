from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any

from src.cross_encoder_reranking import build_reranker_document, cross_encoder_rerank


@dataclass
class FakeChunk:
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    distance: float
    score: float
    rerank_score: float = 0.0
    rerank_reason: str = ""


@dataclass
class FakeEncoder:
    scores: list[float]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def rerank(self, query: str, documents: list[str], batch_size: int = 64, **kwargs: Any) -> list[float]:
        self.calls.append({"query": query, "documents": documents, "batch_size": batch_size})
        return self.scores


class CrossEncoderRerankingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            FakeChunk("a", "dense retrieval text", {"title": "Dense", "category": "retrieval"}, 0.1, 0.8),
            FakeChunk("b", "cross encoder reranking text", {"title": "Reranker", "category": "reranking"}, 0.2, 0.5),
            FakeChunk("c", "unrelated text", {"title": "Other", "category": "overview"}, 0.3, 0.4),
        ]

    def test_model_scores_replace_base_order(self) -> None:
        encoder = FakeEncoder([0.1, 0.95, -0.2])
        ranked = cross_encoder_rerank("重排有什么用", self.chunks, top_k=2, encoder=encoder, batch_size=2)

        self.assertEqual(["b", "a"], [item.chunk_id for item in ranked])
        self.assertEqual(0.95, ranked[0].rerank_score)
        self.assertIn("base=0.500000", ranked[0].rerank_reason)
        self.assertEqual(2, encoder.calls[0]["batch_size"])

    def test_reranker_document_contains_structure_and_text(self) -> None:
        text = build_reranker_document(self.chunks[1])
        self.assertIn("Title: Reranker", text)
        self.assertIn("Category: reranking", text)
        self.assertIn("cross encoder reranking text", text)

    def test_score_count_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "returned 1 scores for 3 candidates"):
            cross_encoder_rerank("query", self.chunks, top_k=2, encoder=FakeEncoder([0.5]))

    def test_rank_fusion_preserves_strong_retrieval_signal(self) -> None:
        encoder = FakeEncoder([0.1, 0.2, 0.95])
        ranked = cross_encoder_rerank(
            "query",
            self.chunks,
            top_k=3,
            encoder=encoder,
            retrieval_weight=0.65,
        )

        self.assertEqual(["a", "b", "c"], [item.chunk_id for item in ranked])
        self.assertIn("retrieval_weight=0.65", ranked[0].rerank_reason)

    def test_invalid_rank_fusion_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "retrieval_weight"):
            cross_encoder_rerank("query", self.chunks, top_k=2, encoder=FakeEncoder([]), retrieval_weight=1.1)


if __name__ == "__main__":
    unittest.main()
