from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "30_natural_query_development"
    / "compare_intent_to_retrieval.py"
)
SPEC = importlib.util.spec_from_file_location("compare_intent_to_retrieval", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NaturalQueryRetrievalAnalysisTest(unittest.TestCase):
    def test_coverage_gap_has_no_oracle_match(self) -> None:
        dataset = [
            {
                "id": "q1",
                "question": "question",
                "persona": "user",
                "intended_route": "planned",
            }
        ]
        result = {
            "case_id": "q1",
            "selected_system": "direct_hybrid",
            "oracle_system": None,
            "oracle_match": None,
            "evaluable": False,
            "metrics": {
                "direct_hybrid": {"ndcg_at_10": 0.0},
                "planned_v2_hybrid": {"ndcg_at_10": 0.0},
            },
        }

        rows = MODULE.build_records(
            dataset,
            [result],
            direct_system="direct_hybrid",
            planned_system="planned_v2_hybrid",
            meaningful_margin=0.05,
        )

        self.assertEqual("coverage_gap", rows[0]["benefit_class"])
        self.assertIsNone(rows[0]["intended_oracle_match"])
        self.assertIsNone(rows[0]["auto_oracle_match"])

    def test_build_records_classifies_meaningful_planned_gain(self) -> None:
        dataset = [{"id": "q1", "question": "question", "persona": "user", "intended_route": "planned"}]
        retrieval = [
            {
                "case_id": "q1",
                "selected_system": "planned_v2_hybrid",
                "oracle_system": "planned_v2_hybrid",
                "oracle_match": True,
                "metrics": {
                    "direct_hybrid": {"ndcg_at_10": 0.4},
                    "planned_v2_hybrid": {"ndcg_at_10": 0.7},
                },
            }
        ]

        rows = MODULE.build_records(
            dataset,
            retrieval,
            direct_system="direct_hybrid",
            planned_system="planned_v2_hybrid",
            meaningful_margin=0.05,
        )

        self.assertEqual("planned_better", rows[0]["benefit_class"])
        self.assertTrue(rows[0]["intended_oracle_match"])

    def test_build_records_rejects_mismatched_case_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "dataset/results ids differ"):
            MODULE.build_records(
                [{"id": "q1"}],
                [],
                direct_system="direct_hybrid",
                planned_system="planned_v2_hybrid",
                meaningful_margin=0.05,
            )


if __name__ == "__main__":
    unittest.main()
