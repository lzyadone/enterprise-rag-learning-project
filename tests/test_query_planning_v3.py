from __future__ import annotations

import unittest

from src.query_planning import plan_query, plan_query_v3


class ConservativeQueryPlanningTest(unittest.TestCase):
    def test_specific_api_question_does_not_expand_to_generic_rag_aspects(self) -> None:
        query = "我在用 Ollama 做 RAG，embedding 接口输入和输出分别是什么？"

        legacy = plan_query(query)
        conservative = plan_query_v3(query)

        self.assertGreater(len(legacy.aspects), 0)
        self.assertEqual([], conservative.aspects)
        self.assertEqual([query], conservative.sub_queries)

    def test_ingestion_update_keeps_two_explicit_aspects_and_original_terms(self) -> None:
        query = (
            "知识库增量更新时，文档ID、去重和向量写入怎样保持幂等，"
            "缓存又应该什么时候清理？"
        )

        plan = plan_query_v3(query)

        self.assertEqual(
            ["ingestion_identity", "ingestion_cache"],
            [aspect.name for aspect in plan.aspects],
        )
        self.assertEqual("rules_v3_conservative", plan.planner_version)
        self.assertTrue(all(query in aspect.search_query for aspect in plan.aspects))
        self.assertEqual(3, len(plan.sub_queries))

    def test_generic_metric_does_not_create_citation_aspect(self) -> None:
        plan = plan_query_v3("contextual recall 和 faithfulness 这些评估指标分别说明什么？")

        self.assertEqual(["evaluation"], [aspect.name for aspect in plan.aspects])
        self.assertEqual(1, len(plan.sub_queries))

    def test_explicit_citation_badcase_remains_multi_aspect(self) -> None:
        plan = plan_query_v3(
            "如何组合正确性、faithfulness 和引用质量指标分析 RAG badcase？"
        )

        self.assertEqual(
            ["evaluation", "citation_quality", "badcase_analysis"],
            [aspect.name for aspect in plan.aspects],
        )
        self.assertEqual(4, len(plan.sub_queries))
