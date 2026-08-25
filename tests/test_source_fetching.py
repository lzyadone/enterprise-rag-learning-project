from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "experiments" / "16_llm_rag_sources" / "fetch_sources.py"
SPEC = importlib.util.spec_from_file_location("fetch_sources", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SourceFetchingTest(unittest.TestCase):
    def test_targeted_fetch_keeps_all_eligible_document_sources(self) -> None:
        sources = [make_source("one"), make_source("two"), make_source("later", priority="P1")]
        args = make_args(source_id=["two"])

        selected = MODULE.select_sources(sources, args)
        document_sources = MODULE.select_eligible_sources(sources, args)

        self.assertEqual(["two"], [source.source_id for source in selected])
        self.assertEqual(["one", "two"], [source.source_id for source in document_sources])

    def test_incomplete_targeted_aggregation_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            write_normalized_source(raw_dir, "one", "first body")
            processed_dir.mkdir(parents=True)
            output = processed_dir / "documents.jsonl"
            output.write_text("existing corpus\n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                MODULE.build_documents_jsonl(
                    raw_dir,
                    processed_dir,
                    [make_source("one"), make_source("two")],
                    require_complete=True,
                )

            self.assertEqual("existing corpus\n", output.read_text(encoding="utf-8"))
            self.assertFalse((processed_dir / "documents.jsonl.tmp").exists())

    def test_complete_targeted_aggregation_replaces_full_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            write_normalized_source(raw_dir, "one", "first body")
            write_normalized_source(raw_dir, "two", "second body")

            count = MODULE.build_documents_jsonl(
                raw_dir,
                processed_dir,
                [make_source("one"), make_source("two")],
                require_complete=True,
            )

            rows = [
                json.loads(line)
                for line in (processed_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(2, count)
            self.assertEqual(["one", "two"], [row["source_id"] for row in rows])


def make_args(*, source_id: list[str] | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        priority="P0",
        include_all=False,
        ingest_first_only=True,
        source_id=source_id,
        limit=None,
    )


def make_source(source_id: str, *, priority: str = "P0"):
    return MODULE.Source(
        source_id=source_id,
        priority=priority,
        category="document loading",
        title=f"Source {source_id}",
        source_type="official_doc",
        url=f"https://example.com/{source_id}",
        ingest_first="yes",
        why="test source",
        notes="",
    )


def write_normalized_source(raw_dir: Path, source_id: str, body: str) -> None:
    source_dir = raw_dir / source_id
    source_dir.mkdir(parents=True)
    (source_dir / "document.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
