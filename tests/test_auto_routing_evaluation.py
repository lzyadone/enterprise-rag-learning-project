from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "experiments" / "28_auto_retrieval_routing" / "evaluate_router.py"
SPEC = importlib.util.spec_from_file_location("evaluate_router", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AutoRoutingAcceptanceTest(unittest.TestCase):
    def test_acceptance_uses_configured_system_names(self) -> None:
        args = make_args()
        records = [
            make_record("q1", "planned_v2_hybrid", 0.8, 0.75, oracle_match=True),
            make_record("q2", "direct_hybrid", 0.7, 0.6, oracle_match=True),
        ]
        rows = [
            metric_row("direct_hybrid", ndcg=0.75, recall=0.8, mrr=0.8, median=1.0),
            metric_row("planned_v2_hybrid", ndcg=0.68, recall=0.79, mrr=0.76, median=4.0),
            metric_row("auto", ndcg=0.75, recall=0.8, mrr=0.8, median=3.0),
        ]

        result = MODULE.evaluate_acceptance(
            args,
            records,
            rows,
            ("direct_hybrid", "planned_v2_hybrid"),
        )

        self.assertTrue(result["passed"])
        self.assertEqual([], result["harmful_planned_cases"])

    def test_large_planned_regret_fails_gate(self) -> None:
        args = make_args()
        records = [make_record("q1", "planned_v2_hybrid", 0.9, 0.5, oracle_match=False)]
        rows = [
            metric_row("direct_hybrid", ndcg=0.9, recall=1.0, mrr=1.0, median=1.0),
            metric_row("planned_v2_hybrid", ndcg=0.5, recall=0.8, mrr=0.7, median=4.0),
            metric_row("auto", ndcg=0.5, recall=0.8, mrr=0.7, median=4.0),
        ]

        result = MODULE.evaluate_acceptance(
            args,
            records,
            rows,
            ("direct_hybrid", "planned_v2_hybrid"),
        )

        self.assertFalse(result["passed"])
        self.assertEqual("q1", result["harmful_planned_cases"][0]["case_id"])


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        quality_margin=0.02,
        mrr_margin=0.05,
        oracle_agreement_min=0.5,
        max_planned_regret=0.25,
        latency_budget_ms=12000,
    )


def make_record(
    case_id: str,
    selected_system: str,
    direct_ndcg: float,
    planned_ndcg: float,
    *,
    oracle_match: bool,
) -> dict:
    return {
        "case_id": case_id,
        "selected_system": selected_system,
        "oracle_match": oracle_match,
        "metrics": {
            "direct_hybrid": {"ndcg_at_10": direct_ndcg},
            "planned_v2_hybrid": {"ndcg_at_10": planned_ndcg},
        },
    }


def metric_row(system: str, *, ndcg: float, recall: float, mrr: float, median: float) -> dict:
    return {
        "system": system,
        "avg_ndcg_at_10": ndcg,
        "avg_recall_at_10": recall,
        "mrr_at_10": mrr,
        "median_seconds": median,
    }


if __name__ == "__main__":
    unittest.main()
