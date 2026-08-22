from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.retrieval import prepare_model_residency


class ModelResidencyTest(unittest.TestCase):
    @patch("src.retrieval.unload_embedding_model")
    def test_cross_encoder_unloads_embedding_model_by_default(self, unload: object) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAG_EXCLUSIVE_MODEL_RESIDENCY", None)
            prepare_model_residency("cross_encoder", "bge-m3", "http://127.0.0.1:11434")

        unload.assert_called_once_with("bge-m3", "http://127.0.0.1:11434")

    @patch("src.retrieval.unload_embedding_model")
    def test_non_model_reranker_does_not_unload(self, unload: object) -> None:
        prepare_model_residency("lexical", "bge-m3", "http://127.0.0.1:11434")
        unload.assert_not_called()

    @patch("src.retrieval.unload_embedding_model")
    def test_exclusive_residency_can_be_disabled(self, unload: object) -> None:
        with patch.dict(os.environ, {"RAG_EXCLUSIVE_MODEL_RESIDENCY": "false"}):
            prepare_model_residency("cross_encoder", "bge-m3", "http://127.0.0.1:11434")

        unload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
