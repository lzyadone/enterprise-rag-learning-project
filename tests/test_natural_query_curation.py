from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "experiments" / "30_natural_query_development" / "curate_queries.py"
SPEC = importlib.util.spec_from_file_location("curate_queries", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NaturalQueryCurationTest(unittest.TestCase):
    def test_same_route_duplicate_is_removed(self) -> None:
        rows = [make_row("q1", "direct"), make_row("q2", "direct")]
        kept, removed = MODULE.curate(rows, [[1.0, 0.0], [0.99, 0.01]], 0.95)

        self.assertEqual(["q1"], [row["id"] for row in kept])
        self.assertEqual("q2", removed[0]["removed_id"])

    def test_cross_route_pair_is_retained(self) -> None:
        rows = [make_row("q1", "direct"), make_row("q2", "planned")]
        kept, removed = MODULE.curate(rows, [[1.0, 0.0], [1.0, 0.0]], 0.95)

        self.assertEqual(["q1", "q2"], [row["id"] for row in kept])
        self.assertEqual([], removed)


def make_row(case_id: str, route: str) -> dict:
    return {"id": case_id, "question": case_id, "intended_route": route}


if __name__ == "__main__":
    unittest.main()
