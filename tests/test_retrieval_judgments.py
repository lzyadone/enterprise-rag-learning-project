import json
import tempfile
import unittest
from pathlib import Path

from src.retrieval_judgments import JudgmentStore, load_candidate_pools, load_complete_qrels


class RetrievalJudgmentsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.pool_path = self.root / "candidate_pools.jsonl"
        self.pool_path.write_text(
            json.dumps(
                {
                    "case_id": "q1",
                    "plan": {"original_query": "What is RAG?"},
                    "candidates": [
                        {"chunk_id": "doc::1", "document": "answer"},
                        {"chunk_id": "doc::2", "document": "noise"},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.pools = load_candidate_pools(self.pool_path)
        self.qrels_path = self.root / "qrels.jsonl"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_upsert_persists_one_row_per_pair(self) -> None:
        store = JudgmentStore(self.qrels_path, self.pools)
        store.upsert("q1", "doc::1", 2, note="supporting")
        store.upsert("q1", "doc::1", 3, note="direct evidence")

        reloaded = JudgmentStore(self.qrels_path, self.pools)
        self.assertEqual(1, reloaded.progress()["labeled"])
        self.assertEqual(3, reloaded.get("q1", "doc::1")["relevance"])
        self.assertEqual("direct evidence", reloaded.get("q1", "doc::1")["note"])

    def test_invalid_grade_is_rejected(self) -> None:
        store = JudgmentStore(self.qrels_path, self.pools)
        with self.assertRaisesRegex(ValueError, "0 to 3"):
            store.upsert("q1", "doc::1", 4)

    def test_unknown_candidate_is_rejected(self) -> None:
        store = JudgmentStore(self.qrels_path, self.pools)
        with self.assertRaisesRegex(ValueError, "unknown query/chunk pair"):
            store.upsert("q1", "doc::missing", 1)

    def test_delete_removes_persisted_judgment(self) -> None:
        store = JudgmentStore(self.qrels_path, self.pools)
        store.upsert("q1", "doc::2", 0)
        self.assertTrue(store.delete("q1", "doc::2"))
        self.assertFalse(store.delete("q1", "doc::2"))
        self.assertEqual([], JudgmentStore(self.qrels_path, self.pools).rows())

    def test_complete_qrels_returns_grades_by_query(self) -> None:
        store = JudgmentStore(self.qrels_path, self.pools)
        store.upsert("q1", "doc::1", 3)
        store.upsert("q1", "doc::2", 0)

        grades = load_complete_qrels(self.qrels_path, self.pools)

        self.assertEqual({"doc::1": 3, "doc::2": 0}, grades["q1"])

    def test_incomplete_qrels_is_rejected(self) -> None:
        JudgmentStore(self.qrels_path, self.pools).upsert("q1", "doc::1", 3)

        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_complete_qrels(self.qrels_path, self.pools)


if __name__ == "__main__":
    unittest.main()
