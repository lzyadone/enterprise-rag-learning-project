import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "37_unified_quality_gate"
    / "run_quality_gate.py"
)
SPEC = importlib.util.spec_from_file_location("run_quality_gate", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeRunner:
    def __init__(self, failing_stage_text: str = "") -> None:
        self.commands: list[list[str]] = []
        self.failing_stage_text = failing_stage_text

    def __call__(self, command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        joined = " ".join(command)
        if self.failing_stage_text and self.failing_stage_text in joined:
            return subprocess.CompletedProcess(command, 1, "", "failed")
        if "unittest" in command:
            return subprocess.CompletedProcess(command, 0, "", "Ran 131 tests\n\nOK")
        if "run_security_regression.py" in joined:
            return subprocess.CompletedProcess(
                command,
                0,
                "rag_security_v1: 8/8 passed; gate_pass=true",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")


class UnifiedQualityGateTest(unittest.TestCase):
    def run_gate(self, runner: FakeRunner, **kwargs):
        with tempfile.TemporaryDirectory() as temporary:
            return MODULE.run_quality_gate(
                root=Path(temporary),
                profile="ci",
                security_output=Path(temporary) / "security.json",
                runner=runner,
                node_path="node",
                **kwargs,
            )

    def test_core_gate_passes_and_records_counts(self) -> None:
        report = self.run_gate(FakeRunner())
        stages = {stage["id"]: stage for stage in report["stages"]}
        self.assertEqual("passed", report["status"])
        self.assertEqual(131, stages["unit_tests"]["test_count"])
        self.assertEqual(8, stages["security_gate"]["passed_count"])
        self.assertEqual("skipped", stages["index_release_gate"]["status"])

    def test_failed_stage_fails_the_overall_gate(self) -> None:
        report = self.run_gate(FakeRunner("compileall"))
        self.assertEqual("failed", report["status"])
        self.assertEqual(["compile_python"], report["failed_stages"])

    def test_manifest_adds_index_release_gate(self) -> None:
        runner = FakeRunner()
        report = self.run_gate(runner, manifest=Path("candidate/manifest.json"))
        stages = {stage["id"]: stage for stage in report["stages"]}
        self.assertEqual("passed", stages["index_release_gate"]["status"])
        self.assertTrue(any("run_gate.py" in " ".join(command) for command in runner.commands))

    def test_ci_profile_requires_node(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = MODULE.run_quality_gate(
                root=Path(temporary),
                profile="ci",
                security_output=Path(temporary) / "security.json",
                runner=FakeRunner(),
                node_path=None,
                node_resolver=lambda _: None,
            )
        self.assertEqual("failed", report["status"])
        self.assertIn("javascript_syntax", report["failed_stages"])


if __name__ == "__main__":
    unittest.main()
