import os
import unittest
from unittest.mock import patch

from webapp.server import default_retrieval_mode


class WebDefaultsTest(unittest.TestCase):
    def test_retrieval_defaults_to_direct(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_retrieval_mode(), "direct")

    def test_valid_explicit_mode_is_preserved(self) -> None:
        with patch.dict(os.environ, {"RAG_DEFAULT_RETRIEVAL_MODE": "AUTO"}, clear=True):
            self.assertEqual(default_retrieval_mode(), "auto")

    def test_invalid_explicit_mode_falls_back_to_direct(self) -> None:
        with patch.dict(os.environ, {"RAG_DEFAULT_RETRIEVAL_MODE": "unknown"}, clear=True):
            self.assertEqual(default_retrieval_mode(), "direct")


if __name__ == "__main__":
    unittest.main()
