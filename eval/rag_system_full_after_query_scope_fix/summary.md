# RAG System Smoke Evaluation

## Summary

- total: 10
- passed: 8 (80.00%)
- quality_pass: 8 (80.00%)
- avg_seconds: 28.96
- llm_provider: deepseek
- retrieval_mode: planned
- rerank_mode: lexical
- top_k/candidate_k: 7/16
- max_context_chars: 15000
- audit_answer: True

## Case Results

| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rag_compound_overview | Y | Y | Y | Y | Y | 10 | 46.39 |  |
| chunking_not_fixed_window | Y | Y | Y | Y | Y | 7 | 19.49 |  |
| rag_evaluation_reliability | Y | Y | Y | Y | Y | 10 | 40.92 |  |
| long_memory_learning_preference | Y | Y | - | - | Y | 0 | 1.41 |  |
| embedding_bge_m3_role | Y | Y | - | Y | Y | 7 | 9.98 |  |
| reranking_role | N | N | Y | Y | Y | 7 | 37.99 | quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a... |
| query_planning_expansion | Y | Y | - | Y | Y | 7 | 33.65 |  |
| vector_db_chroma_role | N | N | - | Y | Y | 7 | 47.87 | quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a... |
| enterprise_rag_retrieval | Y | Y | Y | Y | Y | 7 | 30.86 |  |
| knowledge_boundary_weather | Y | Y | - | Y | Y | 7 | 21.03 |  |

## Badcases

### reranking_role

- question: 重排在RAG里解决什么问题？它和普通向量检索是什么关系？
- quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a...

### vector_db_chroma_role

- question: Chroma在这个RAG项目里承担什么角色？metadata filter有什么用？
- quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a...
