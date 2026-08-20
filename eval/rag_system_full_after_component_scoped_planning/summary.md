# RAG System Smoke Evaluation

## Summary

- total: 10
- passed: 8 (80.00%)
- quality_pass: 8 (80.00%)
- avg_seconds: 22.79
- llm_provider: deepseek
- retrieval_mode: planned
- rerank_mode: lexical
- top_k/candidate_k: 7/16
- max_context_chars: 15000
- audit_answer: True

## Case Results

| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rag_compound_overview | Y | Y | Y | Y | Y | 10 | 36.83 |  |
| chunking_not_fixed_window | Y | Y | Y | Y | Y | 7 | 17.03 |  |
| rag_evaluation_reliability | Y | Y | Y | Y | Y | 10 | 35.03 |  |
| long_memory_learning_preference | Y | Y | - | - | Y | 0 | 1.56 |  |
| embedding_bge_m3_role | Y | Y | - | Y | Y | 7 | 7.52 |  |
| reranking_role | Y | Y | Y | Y | Y | 7 | 26.26 |  |
| query_planning_expansion | Y | Y | - | Y | Y | 7 | 28.28 |  |
| vector_db_chroma_role | N | N | - | Y | Y | 7 | 34.79 | quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a... |
| enterprise_rag_retrieval | N | N | Y | Y | Y | 7 | 32.82 | quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a... |
| knowledge_boundary_weather | Y | Y | - | Y | Y | 7 | 7.81 |  |

## Badcases

### vector_db_chroma_role

- question: Chroma在这个RAG项目里承担什么角色？metadata filter有什么用？
- quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a...

### enterprise_rag_retrieval

- question: 企业级RAG为什么需要混合检索、metadata过滤和重排，而不是只做一次向量相似度搜索？
- quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a...
