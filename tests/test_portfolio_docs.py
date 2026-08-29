import json
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO_DIR = PROJECT_ROOT / "docs" / "portfolio"
BEGINNER_GUIDE = PROJECT_ROOT / "docs" / "BEGINNER_PROJECT_GUIDE.md"
PORTFOLIO_FILES = [
    PORTFOLIO_DIR / "README.md",
    PORTFOLIO_DIR / "architecture.md",
    PORTFOLIO_DIR / "metrics.md",
    PORTFOLIO_DIR / "technical_decisions.md",
    PORTFOLIO_DIR / "demo_script.md",
    PORTFOLIO_DIR / "resume_project.md",
]


class PortfolioDocsTest(unittest.TestCase):
    def test_required_portfolio_documents_exist(self) -> None:
        for path in PORTFOLIO_FILES + [BEGINNER_GUIDE]:
            self.assertTrue(path.is_file(), path)
            text = path.read_text(encoding="utf-8")
            self.assertGreater(len(text), 500, path)
            self.assertEqual(0, text.count("```") % 2, path)

    def test_relative_markdown_links_resolve(self) -> None:
        documents = PORTFOLIO_FILES + [
            BEGINNER_GUIDE,
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs" / "demo.md",
        ]
        failures = []
        for document in documents:
            text = document.read_text(encoding="utf-8")
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                relative_target = target.split("#", 1)[0]
                if not relative_target:
                    continue
                resolved = (document.parent / relative_target).resolve()
                if not resolved.exists():
                    failures.append(f"{document.relative_to(PROJECT_ROOT)} -> {target}")
        self.assertEqual([], failures)

    def test_headline_metrics_match_structured_evidence(self) -> None:
        metrics_text = (PORTFOLIO_DIR / "metrics.md").read_text(encoding="utf-8")
        security = json.loads(
            (PROJECT_ROOT / "eval" / "rag_security_v1" / "summary.json").read_text(encoding="utf-8")
        )
        system = json.loads(
            (PROJECT_ROOT / "eval" / "rag_system_full_hybrid" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        cache = json.loads(
            (
                PROJECT_ROOT
                / "eval"
                / "planned_retrieval_cache_benchmark"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )

        self.assertTrue(security["summary"]["gate_pass"])
        self.assertEqual(8, security["summary"]["passed"])
        self.assertEqual(10, system["passed"])
        self.assertEqual(10, system["quality_pass"])
        self.assertTrue(all(row["same_order_cases"] == 8 for row in cache["systems"]))
        for expected in ["8/8", "10/10", "Recall@10 1.000", "nDCG@10 0.843", "139/139"]:
            self.assertIn(expected, metrics_text)

    def test_architecture_and_demo_cover_release_boundaries(self) -> None:
        architecture = (PORTFOLIO_DIR / "architecture.md").read_text(encoding="utf-8")
        demo = (PORTFOLIO_DIR / "demo_script.md").read_text(encoding="utf-8")
        guide = BEGINNER_GUIDE.read_text(encoding="utf-8")
        resume = (PORTFOLIO_DIR / "resume_project.md").read_text(encoding="utf-8")

        self.assertGreaterEqual(architecture.count("```mermaid"), 3)
        for marker in ["查询安全与知识边界", "Chroma dense index", "BM25 sparse index", "rollback"]:
            self.assertIn(marker, architecture)
        for marker in [
            "模式**：direct",
            "planned v3",
            "memory answer mode",
            "security_policy",
            "明天杭州天气如何",
        ]:
            self.assertIn(marker, demo)
        for marker in [
            "什么是 RAG",
            "Embedding 是什么",
            "为什么要做混合检索",
            "版本化增量索引",
            "换成其他数据能不能迁移",
            "术语表",
            "推荐学习路线",
        ]:
            self.assertIn(marker, guide)
        for marker in [
            "企业级 RAG 知识库系统",
            "60 秒面试自我介绍",
            "三个重点面试案例",
            "Recall@10 1.000",
            "不要虚构公司",
            "投递前检查清单",
        ]:
            self.assertIn(marker, resume)


if __name__ == "__main__":
    unittest.main()
