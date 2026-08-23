from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "experiments" / "30_natural_query_development" / "generate_queries.py"
SPEC = importlib.util.spec_from_file_location("generate_queries", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NaturalQueryGenerationTest(unittest.TestCase):
    def test_accepts_natural_direct_question(self) -> None:
        item = make_item(
            "我把 PDF 放进知识库后，为什么搜索结果里看不到文件名？",
            topics=["document loading"],
            needs=["确认来源 metadata 在加载阶段如何保留"],
        )

        issue = MODULE.validate_generated_item(
            item,
            intended_route="direct",
            seen_questions=[],
            max_similarity=0.84,
        )

        self.assertIsNone(issue)

    def test_rejects_exam_style_question(self) -> None:
        item = make_item(
            "请根据资料列举 RAG 的关键组件。",
            topics=["RAG basics"],
            needs=["了解关键组件"],
        )

        issue = MODULE.validate_generated_item(
            item,
            intended_route="direct",
            seen_questions=[],
            max_similarity=0.84,
        )

        self.assertEqual("exam_style", issue)

    def test_planned_question_requires_multiple_information_needs(self) -> None:
        item = make_item(
            "线上答案不稳定，我应该先排查哪里？",
            topics=["RAG failures"],
            needs=["定位问题"],
        )

        issue = MODULE.validate_generated_item(
            item,
            intended_route="planned",
            seen_questions=[],
            max_similarity=0.84,
        )

        self.assertEqual("planned_need_count", issue)

    def test_rejects_near_duplicate(self) -> None:
        item = make_item(
            "Chroma 里的 metadata filter 到底有什么用？",
            topics=["vector database"],
            needs=["理解 metadata filter 的作用"],
        )

        issue = MODULE.validate_generated_item(
            item,
            intended_route="direct",
            seen_questions=["Chroma 里的 metadata filter 有什么用？"],
            max_similarity=0.84,
        )

        self.assertEqual("near_duplicate", issue)

    def test_critic_route_mismatch_is_rejected(self) -> None:
        issue = MODULE.review_issue(
            {
                "naturalness": 5,
                "standalone": True,
                "route": "planned",
                "information_need_count": 3,
            },
            intended_route="direct",
        )

        self.assertEqual("critic_route_mismatch", issue)

    def test_cosine_similarity_detects_identical_vectors(self) -> None:
        self.assertAlmostEqual(1.0, MODULE.cosine_similarity([1.0, 2.0], [1.0, 2.0]))


def make_item(question: str, *, topics: list[str], needs: list[str]) -> dict:
    return {
        "question": question,
        "persona": "正在接入知识库的开发者",
        "scenario": "正在调试一个内部 RAG 项目",
        "topics": topics,
        "information_needs": needs,
    }


if __name__ == "__main__":
    unittest.main()
