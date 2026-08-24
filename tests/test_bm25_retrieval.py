from __future__ import annotations

import unittest

from src.bm25_retrieval import BM25ChunkIndex, SimpleBM25Okapi, tokenize


class BM25ChunkIndexTest(unittest.TestCase):
    def setUp(self) -> None:
        self.index = BM25ChunkIndex(
            [
                {
                    "chunk_id": "dense-doc",
                    "title": "Dense retrieval",
                    "category": "retrieval",
                    "heading_path": "Dense embeddings",
                    "text": "Dense vector search uses embedding similarity for semantic recall.",
                },
                {
                    "chunk_id": "filter-doc",
                    "title": "Chroma metadata filter",
                    "category": "vector db",
                    "heading_path": "Filter records with where",
                    "text": "Use a metadata filter and a where clause to restrict Chroma results.",
                },
                {
                    "chunk_id": "rerank-doc",
                    "title": "重排",
                    "category": "reranking",
                    "heading_path": "候选集重排",
                    "text": "重排模型对第一阶段召回的候选文档重新打分。",
                },
            ]
        )

    def test_exact_technical_term_is_retrieved(self) -> None:
        hits = self.index.search("Chroma 的 metadata filter 和 where 有什么用？", top_k=2)
        self.assertEqual("filter-doc", hits[0].chunk_id)

    def test_chinese_query_uses_ngram_tokens(self) -> None:
        self.assertIn("重排", tokenize("为什么需要重排模型"))
        hits = self.index.search("候选文档为什么需要重排？", top_k=1)
        self.assertEqual("rerank-doc", hits[0].chunk_id)

    def test_category_filter_is_applied_before_top_k_cutoff(self) -> None:
        hits = self.index.search("retrieval filter", top_k=1, category="retrieval")
        self.assertEqual("dense-doc", hits[0].chunk_id)

    def test_builtin_fallback_scores_matching_documents(self) -> None:
        model = SimpleBM25Okapi([["dense", "retrieval"], ["metadata", "filter"]])

        scores = model.get_scores(["metadata", "filter"])

        self.assertGreater(scores[1], scores[0])


if __name__ == "__main__":
    unittest.main()
