# RAG System Smoke Evaluation

## Summary

- total: 10
- passed: 8 (80.00%)
- quality_pass: 9 (90.00%)
- avg_seconds: 36.2
- llm_provider: deepseek
- retrieval_mode: planned
- rerank_mode: lexical
- top_k/candidate_k: 7/16
- max_context_chars: 15000
- audit_answer: True

## Case Results

| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rag_compound_overview | Y | Y | Y | Y | Y | 10 | 55.3 |  |
| chunking_not_fixed_window | Y | Y | Y | Y | Y | 7 | 39.14 |  |
| rag_evaluation_reliability | Y | Y | Y | Y | Y | 10 | 55.99 |  |
| long_memory_learning_preference | Y | Y | - | - | Y | 0 | 6.06 |  |
| embedding_bge_m3_role | Y | Y | - | Y | Y | 7 | 20.4 |  |
| reranking_role | Y | Y | Y | Y | Y | 7 | 59.02 |  |
| query_planning_expansion | N | N | - | Y | Y | 7 | 53.04 | quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a... |
| vector_db_chroma_role | Y | Y | - | Y | Y | 7 | 21.15 |  |
| enterprise_rag_retrieval | N | Y | N | N | Y | 7 | 26.3 | aspects: {"expected": ["techniques", "bottlenecks"], "actual": [], "hits": [], "min_hits": 1}<br>categories: {"expected": ["retrieval", "reranking", "vector db", "RAG challenges"], "actual": ["RAG overview", "vector db"], "hits": ["vector db"], "min_hits": 2} |
| knowledge_boundary_weather | Y | Y | - | Y | Y | 7 | 25.65 |  |

## Badcases

### query_planning_expansion

- question: 复杂问题在RAG里为什么要做query rewrite或query expansion？
- quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a...

### enterprise_rag_retrieval

- question: 企业级RAG为什么需要混合检索、metadata过滤和重排，而不是只做一次向量相似度搜索？
- aspects: {"expected": ["techniques", "bottlenecks"], "actual": [], "hits": [], "min_hits": 1}
- categories: {"expected": ["retrieval", "reranking", "vector db", "RAG challenges"], "actual": ["RAG overview", "vector db"], "hits": ["vector db"], "min_hits": 2}
