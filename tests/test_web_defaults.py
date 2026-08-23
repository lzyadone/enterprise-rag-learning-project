import os
import unittest
from unittest.mock import patch

from webapp.server import default_planned_fusion_mode, default_retrieval_mode


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

    def test_planned_fusion_defaults_to_anchored(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_planned_fusion_mode(), "anchored")

    def test_legacy_planned_fusion_can_be_selected_for_ab(self) -> None:
        with patch.dict(os.environ, {"RAG_PLANNED_FUSION_MODE": "legacy"}, clear=True):
            self.assertEqual(default_planned_fusion_mode(), "legacy")


if __name__ == "__main__":
    unittest.main()
