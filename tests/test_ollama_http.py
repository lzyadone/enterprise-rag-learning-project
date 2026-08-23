from __future__ import annotations

import unittest
from unittest.mock import patch

from src.ollama_http import clear_embedding_cache, embed_queries, embed_query, embedding_cache_info


class OllamaEmbeddingCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_embedding_cache()

    def tearDown(self) -> None:
        clear_embedding_cache()

    def test_batches_unique_misses_and_reuses_normalized_queries(self) -> None:
        def fake_embed_texts(texts: list[str], model: str, host: str) -> list[list[float]]:
            self.assertEqual("bge-m3", model)
            self.assertEqual("http://127.0.0.1:11434", host)
            return [[float(len(text))] for text in texts]

        with patch("src.ollama_http.embed_texts", side_effect=fake_embed_texts) as mocked:
            vectors = embed_queries(
                ["RAG   retrieval", "RAG retrieval", "metadata filter"],
                model="bge-m3",
                host="http://127.0.0.1:11434/",
            )
            repeated = embed_query(
                " RAG retrieval ",
                model="bge-m3",
                host="http://127.0.0.1:11434",
            )

        self.assertEqual([[13.0], [13.0], [15.0]], vectors)
        self.assertEqual([13.0], repeated)
        mocked.assert_called_once_with(
            ["RAG retrieval", "metadata filter"],
            model="bge-m3",
            host="http://127.0.0.1:11434",
        )
        self.assertEqual(
            {
                "size": 2,
                "max_size": 256,
                "hits": 2,
                "misses": 2,
                "api_requests": 1,
                "bypasses": 0,
            },
            embedding_cache_info(),
        )

    def test_cache_can_be_bypassed_for_controlled_benchmarks(self) -> None:
        with patch("src.ollama_http.embed_texts", return_value=[[0.1, 0.2]]) as mocked:
            first = embed_query("same", "model", "http://host", use_cache=False)
            second = embed_query("same", "model", "http://host", use_cache=False)

        self.assertEqual(first, second)
        self.assertEqual(2, mocked.call_count)
        self.assertEqual(2, embedding_cache_info()["api_requests"])
        self.assertEqual(2, embedding_cache_info()["bypasses"])


if __name__ == "__main__":
    unittest.main()
