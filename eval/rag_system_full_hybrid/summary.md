# RAG System Smoke Evaluation

## Summary

- total: 10
- passed: 10 (100.00%)
- quality_pass: 10 (100.00%)
- avg_seconds: 21.22
- llm_provider: deepseek
- retrieval_mode: planned
- retrieval_strategy: hybrid
- rerank_mode: lexical
- top_k/candidate_k: 7/16
- max_context_chars: 15000
- audit_answer: True

## Case Results

| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rag_compound_overview | Y | Y | Y | Y | Y | 10 | 41.57 |  |
| chunking_not_fixed_window | Y | Y | Y | Y | Y | 7 | 15.36 |  |
| rag_evaluation_reliability | Y | Y | Y | Y | Y | 10 | 39.65 |  |
| long_memory_learning_preference | Y | Y | - | - | Y | 0 | 2.03 |  |
| embedding_bge_m3_role | Y | Y | - | Y | Y | 7 | 9.05 |  |
| reranking_role | Y | Y | Y | Y | Y | 7 | 19.47 |  |
| query_planning_expansion | Y | Y | - | Y | Y | 7 | 23.22 |  |
| vector_db_chroma_role | Y | Y | - | Y | Y | 7 | 18.15 |  |
| enterprise_rag_retrieval | Y | Y | Y | Y | Y | 7 | 31.76 |  |
| knowledge_boundary_weather | Y | Y | - | Y | Y | 7 | 11.93 |  |