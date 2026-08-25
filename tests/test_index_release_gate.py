from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.index_release_gate import (
    evaluate_retrieval_case,
    execute_release_gate,
    load_gate_spec,
)
from src.index_versioning import write_json_atomic


class IndexReleaseGateTest(unittest.TestCase):
    def test_retrieval_case_checks_categories_terms_and_anchor_rank(self) -> None:
        case = {
            "id": "metadata",
            "question": "question",
            "expected_categories": ["document loading"],
            "min_category_hits": 1,
            "expected_source_terms": ["page_label", "metadata"],
            "min_source_term_hits": 2,
            "expected_chunk_ids": ["source::chunk_0000"],
            "min_chunk_hits": 1,
            "max_chunk_rank": 1,
            "required": True,
        }
        chunks = [
            make_chunk(
                "source::chunk_0000",
                "page_label is preserved in metadata",
                "document loading",
            )
        ]

        result = evaluate_retrieval_case(case, chunks, top_k=7)

        self.assertTrue(result["passed"])
        self.assertTrue(result["required"])
        self.assertEqual(["source::chunk_0000"], result["chunk_hits"])

    def test_gate_spec_rejects_duplicate_case_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "gate.json"
            write_json_atomic(
                path,
                {
                    "schema_version": 1,
                    "retrieval": {"min_pass_rate": 1.0},
                    "cases": [
                        {"id": "same", "question": "first"},
                        {"id": "same", "question": "second"},
                    ],
                },
            )

            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_gate_spec(path)

    def test_passing_gate_writes_reports_and_activates_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path, spec_path = create_inputs(root)
            activations: list[tuple[Path, Path, Path]] = []

            def fake_activate(active_path: Path, candidate: Path, project_root: Path):
                saved = json.loads(
                    (root / "report" / "report.json").read_text(encoding="utf-8")
                )
                self.assertEqual("approved", saved["activation"]["status"])
                activations.append((active_path, candidate, project_root))
                return {"version_id": "candidate"}

            report = execute_release_gate(
                root=root,
                manifest_path=manifest_path,
                gate_spec_path=spec_path,
                report_dir=root / "report",
                active_index_path=root / "active.json",
                activate=True,
                retrieval_runner=lambda _manifest, _spec, _root: passing_retrieval(),
                test_runner=lambda _root: passing_tests(),
                activator=fake_activate,
                structural_validator=lambda _path, _root: candidate_manifest(),
            )

            self.assertEqual("passed", report["status"])
            self.assertEqual("activated", report["activation"]["status"])
            self.assertEqual(1, len(activations))
            self.assertTrue((root / "report" / "report.json").is_file())
            self.assertTrue((root / "report" / "report.md").is_file())

    def test_failed_retrieval_blocks_activation_but_keeps_test_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path, spec_path = create_inputs(root)
            activated = False

            def fake_activate(_active: Path, _candidate: Path, _root: Path):
                nonlocal activated
                activated = True
                return {"version_id": "candidate"}

            retrieval = passing_retrieval()
            retrieval["status"] = "failed"
            retrieval["required_failures"] = ["required-case"]
            report = execute_release_gate(
                root=root,
                manifest_path=manifest_path,
                gate_spec_path=spec_path,
                report_dir=root / "report",
                active_index_path=root / "active.json",
                activate=True,
                retrieval_runner=lambda _manifest, _spec, _root: retrieval,
                test_runner=lambda _root: passing_tests(),
                activator=fake_activate,
                structural_validator=lambda _path, _root: candidate_manifest(),
            )

            self.assertEqual("failed", report["status"])
            self.assertEqual("blocked", report["activation"]["status"])
            self.assertEqual("passed", report["stages"]["tests"]["status"])
            self.assertFalse(activated)

    def test_test_runner_exception_is_reported_and_blocks_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path, spec_path = create_inputs(root)

            def broken_tests(_root: Path) -> dict[str, object]:
                raise RuntimeError("test process unavailable")

            report = execute_release_gate(
                root=root,
                manifest_path=manifest_path,
                gate_spec_path=spec_path,
                report_dir=root / "report",
                active_index_path=root / "active.json",
                activate=True,
                retrieval_runner=lambda _manifest, _spec, _root: passing_retrieval(),
                test_runner=broken_tests,
                structural_validator=lambda _path, _root: candidate_manifest(),
            )

            self.assertEqual("failed", report["status"])
            self.assertEqual("blocked", report["activation"]["status"])
            self.assertEqual("RuntimeError", report["stages"]["tests"]["error_type"])


def make_chunk(chunk_id: str, document: str, category: str):
    return SimpleNamespace(
        chunk_id=chunk_id,
        document=document,
        metadata={"title": "Title", "category": category, "heading_path": "Section"},
    )


def create_inputs(root: Path) -> tuple[Path, Path]:
    manifest_path = root / "manifest.json"
    spec_path = root / "gate.json"
    write_json_atomic(manifest_path, {"placeholder": True})
    write_json_atomic(
        spec_path,
        {
            "schema_version": 1,
            "gate_id": "test-gate",
            "retrieval": {"min_pass_rate": 1.0},
            "cases": [{"id": "case", "question": "question"}],
        },
    )
    return manifest_path, spec_path


def candidate_manifest() -> dict[str, object]:
    return {
        "version_id": "candidate",
        "parent_version": "base",
        "document_count": 2,
        "chunk_count": 3,
        "embedding_model": "fake",
        "documents_hash": "documents",
        "chunks_hash": "chunks",
        "delta": {"added": [], "changed": [], "deleted": [], "unchanged": ["a", "b"]},
        "build": {"reused_embeddings": 3, "new_embeddings": 0},
    }


def passing_retrieval() -> dict[str, object]:
    return {
        "status": "passed",
        "case_count": 1,
        "passed_count": 1,
        "pass_rate": 1.0,
        "min_pass_rate": 1.0,
        "required_failures": [],
        "cases": [],
    }


def passing_tests() -> dict[str, object]:
    return {"status": "passed", "test_count": 10, "return_code": 0}


if __name__ == "__main__":
    unittest.main()
