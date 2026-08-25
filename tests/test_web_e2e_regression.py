import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "experiments" / "32_web_e2e_regression" / "run_web_regression.py"
SPEC = importlib.util.spec_from_file_location("web_e2e_regression", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WebE2ERegressionTest(unittest.TestCase):
    def test_payload_uses_local_generation_and_conservative_planner(self) -> None:
        payload = MODULE.build_payload("question", "planned")

        self.assertEqual("ollama", payload["llm_provider"])
        self.assertEqual("planned", payload["retrieval_mode"])
        self.assertEqual("conservative", payload["planned_fusion_mode"])
        self.assertFalse(payload["use_memory"])
        self.assertFalse(payload["audit_answer"])

    def test_direct_response_contract_passes(self) -> None:
        checks = MODULE.evaluate_response(
            response_contract("direct", plan=None),
            {"expected_category": "chunking"},
            "direct",
            wall_seconds=5.0,
            max_seconds=60.0,
        )

        self.assertTrue(all(checks.values()), checks)

    def test_planned_response_requires_v3_plan(self) -> None:
        response = response_contract("planned", plan={"planner_version": "legacy"})
        checks = MODULE.evaluate_response(
            response,
            {"expected_category": "chunking"},
            "planned",
            wall_seconds=5.0,
            max_seconds=60.0,
        )

        self.assertFalse(checks["planner_shape"])

    def test_response_requires_a_valid_source_citation(self) -> None:
        response = response_contract("direct", plan=None)
        response["answer"] = "answer without citations"
        checks = MODULE.evaluate_response(
            response,
            {"expected_category": "chunking"},
            "direct",
            wall_seconds=5.0,
            max_seconds=60.0,
        )

        self.assertFalse(checks["citation_present"])


def response_contract(mode: str, plan):
    return {
        "ok": True,
        "answer": "grounded answer [1]",
        "settings": {
            "requested_retrieval_mode": mode,
            "retrieval_mode": mode,
            "planned_fusion_mode": "conservative",
        },
        "routing": {"selected_mode": mode},
        "generation": {"provider": "ollama", "provider_path": ["ollama:test"]},
        "sources": [{"index": 1, "category": "chunking", "url": "https://example.com/source"}],
        "plan": plan,
        "timings": {"total_seconds": 4.0},
    }


if __name__ == "__main__":
    unittest.main()
