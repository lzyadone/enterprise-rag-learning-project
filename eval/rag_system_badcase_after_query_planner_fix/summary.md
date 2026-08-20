# RAG System Smoke Evaluation

## Summary

- total: 2
- passed: 1 (50.00%)
- quality_pass: 1 (50.00%)
- avg_seconds: 32.92
- llm_provider: deepseek
- retrieval_mode: planned
- rerank_mode: lexical
- top_k/candidate_k: 7/16
- max_context_chars: 15000
- audit_answer: True

## Case Results

| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| query_planning_expansion | Y | Y | - | Y | Y | 7 | 22.37 |  |
| enterprise_rag_retrieval | N | N | Y | Y | Y | 7 | 43.47 | quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a... |

## Badcases

### enterprise_rag_retrieval

- question: 企业级RAG为什么需要混合检索、metadata过滤和重排，而不是只做一次向量相似度搜索？
- quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a...
