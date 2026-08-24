from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT_ROOT
    / "experiments"
    / "31_planner_v3_holdout"
    / "generate_grounded_holdout.py"
)
SPEC = importlib.util.spec_from_file_location("generate_grounded_holdout", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GroundedHoldoutGenerationTest(unittest.TestCase):
    def test_evidence_selection_is_deterministic_and_respects_categories(self) -> None:
        specs = (
            MODULE.TargetSpec("one", "focused", ("retrieval",)),
            MODULE.TargetSpec("two", "compound", ("retrieval", "evaluation")),
        )
        chunks = [
            make_chunk("r1", "retrieval", "source-r1"),
            make_chunk("r2", "retrieval", "source-r2"),
            make_chunk("e1", "evaluation", "source-e1"),
        ]

        first = MODULE.select_evidence_bundles(chunks, specs, seed=7)
        second = MODULE.select_evidence_bundles(chunks, specs, seed=7)

        self.assertEqual(
            [[row["chunk_id"] for row in bundle.chunks] for bundle in first],
            [[row["chunk_id"] for row in bundle.chunks] for bundle in second],
        )
        self.assertEqual(["retrieval"], [row["category"] for row in first[0].chunks])
        self.assertEqual(
            ["retrieval", "evaluation"],
            [row["category"] for row in first[1].chunks],
        )

    def test_generation_validation_rejects_exam_style(self) -> None:
        target = MODULE.TargetSpec("one", "focused", ("retrieval",))
        item = {
            "question": "请根据资料分别说明混合检索的内容",
            "persona": MODULE.PERSONAS[0],
            "scenario": "学习检索",
            "information_needs": ["第一点", "第二点"],
        }

        issue = MODULE.validate_generated_item(item, target, [], 0.82)

        self.assertEqual("exam_style", issue)

    def test_generation_validation_allows_multiple_needs_before_review(self) -> None:
        target = MODULE.TargetSpec("one", "focused", ("retrieval",))
        item = {
            "question": "线上检索总漏掉关键词时，我该先排查哪些检索环节？",
            "persona": MODULE.PERSONAS[2],
            "scenario": "排查检索问题",
            "information_needs": ["确认召回方式", "确认重排影响"],
        }

        issue = MODULE.validate_generated_item(item, target, [], 0.82)

        self.assertIsNone(issue)

    def test_generation_validation_rejects_empty_or_excessive_information_needs(self) -> None:
        target = MODULE.TargetSpec("one", "compound", ("retrieval", "evaluation"))
        item = {
            "question": "我想验收检索效果时，应该看哪些召回和相关性信号？",
            "persona": MODULE.PERSONAS[5],
            "scenario": "验收检索效果",
            "information_needs": [],
        }

        self.assertEqual(
            "invalid_information_needs",
            MODULE.validate_generated_item(item, target, [], 0.82),
        )

        item["information_needs"] = [
            f"need {index}" for index in range(MODULE.MAX_GENERATED_INFORMATION_NEEDS + 1)
        ]

        self.assertEqual(
            "invalid_information_needs",
            MODULE.validate_generated_item(item, target, [], 0.82),
        )

    def test_review_requires_full_evidence_coverage(self) -> None:
        target = MODULE.TargetSpec("one", "compound", ("retrieval", "evaluation"))
        review = {
            "naturalness": 5,
            "standalone": True,
            "observed_stratum": "compound",
            "information_need_count": 2,
            "answerable": True,
            "evidence_coverage": 0.75,
            "unsupported_assumptions": False,
        }

        self.assertEqual("critic_not_fully_answerable", MODULE.review_issue(review, target))

    def test_review_enforces_need_count_by_stratum(self) -> None:
        focused = MODULE.TargetSpec("one", "focused", ("retrieval",))
        focused_review = {
            "naturalness": 5,
            "standalone": True,
            "observed_stratum": "focused",
            "information_need_count": 2,
            "answerable": True,
            "evidence_coverage": 1.0,
            "unsupported_assumptions": False,
        }

        self.assertEqual(
            "critic_focused_need_count",
            MODULE.review_issue(focused_review, focused),
        )

        compound = MODULE.TargetSpec("two", "compound", ("retrieval", "evaluation"))
        compound_review = {
            "naturalness": 5,
            "standalone": True,
            "observed_stratum": "compound",
            "information_need_count": 1,
            "answerable": True,
            "evidence_coverage": 1.0,
            "unsupported_assumptions": False,
        }

        self.assertEqual(
            "critic_compound_need_count",
            MODULE.review_issue(compound_review, compound),
        )

    def test_dataset_rows_do_not_expose_source_anchors(self) -> None:
        row = {
            "target_id": "one",
            "question": "线上检索结果为什么总是漏掉关键词？",
            "stratum": "focused",
        }

        result = MODULE.dataset_row(row, 1)

        self.assertEqual("natural_holdout3_001", result["id"])
        self.assertNotIn("target_id", result)
        self.assertNotIn("evidence", result)


def make_chunk(chunk_id: str, category: str, source_id: str) -> dict[str, object]:
    text = "This evidence explains retrieval behavior in sufficient detail. " * 12
    return {
        "chunk_id": chunk_id,
        "source_id": source_id,
        "title": f"Title {chunk_id}",
        "category": category,
        "heading_path": f"Section {chunk_id}",
        "char_count": len(text),
        "text": text,
        "url": "https://example.com",
    }


if __name__ == "__main__":
    unittest.main()
