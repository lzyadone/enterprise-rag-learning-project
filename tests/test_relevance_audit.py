import json
import unittest

from src.relevance_audit import (
    agreement_summary,
    parse_llm_judgments,
    refresh_human_judgments,
)


class RelevanceAuditTest(unittest.TestCase):
    def test_parse_orders_and_validates_judgments(self) -> None:
        content = json.dumps(
            {
                "judgments": [
                    {"chunk_id": "b", "relevance": 1, "reason": "related"},
                    {"chunk_id": "a", "relevance": 3, "reason": "direct"},
                ]
            }
        )
        rows = parse_llm_judgments(content, ["a", "b"])
        self.assertEqual(["a", "b"], [row["chunk_id"] for row in rows])
        self.assertEqual([3, 1], [row["relevance"] for row in rows])

    def test_parse_rejects_missing_candidate(self) -> None:
        content = json.dumps({"judgments": [{"chunk_id": "a", "relevance": 2}]})
        with self.assertRaisesRegex(ValueError, "omitted chunk_ids"):
            parse_llm_judgments(content, ["a", "b"])

    def test_agreement_summary_counts_severe_disagreement(self) -> None:
        summary = agreement_summary(
            [
                {"human_relevance": 3, "llm_relevance": 3},
                {"human_relevance": 2, "llm_relevance": 1},
                {"human_relevance": 3, "llm_relevance": 0},
            ]
        )
        self.assertEqual(1, summary["exact_agreement"])
        self.assertEqual(2, summary["within_one"])
        self.assertEqual(1, summary["severe_disagreements"])

    def test_refresh_human_judgments_recomputes_cached_difference(self) -> None:
        audit_rows = [
            {
                "query_id": "q1",
                "chunk_id": "c1",
                "human_relevance": 3,
                "llm_relevance": 1,
                "absolute_difference": 2,
            }
        ]
        human_lookup = {
            ("q1", "c1"): {
                "relevance": 2,
                "note": "复核：改为辅助证据。",
                "updated_at": "2026-08-22T14:00:00+00:00",
            }
        }

        refreshed = refresh_human_judgments(audit_rows, human_lookup)

        self.assertEqual(2, refreshed[0]["human_relevance"])
        self.assertEqual(1, refreshed[0]["absolute_difference"])
        self.assertEqual("复核：改为辅助证据。", refreshed[0]["human_note"])


if __name__ == "__main__":
    unittest.main()
