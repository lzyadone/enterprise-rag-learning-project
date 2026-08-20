# RAG System Smoke Evaluation

## Summary

- total: 10
- passed: 10 (100.00%)
- quality_pass: 10 (100.00%)
- avg_seconds: 19.52
- llm_provider: deepseek
- retrieval_mode: planned
- rerank_mode: lexical
- top_k/candidate_k: 7/16
- max_context_chars: 15000
- audit_answer: True

## Case Results

| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rag_compound_overview | Y | Y | Y | Y | Y | 10 | 33.75 |  |
| chunking_not_fixed_window | Y | Y | Y | Y | Y | 7 | 13.69 |  |
| rag_evaluation_reliability | Y | Y | Y | Y | Y | 10 | 37.52 |  |
| long_memory_learning_preference | Y | Y | - | - | Y | 0 | 1.35 |  |
| embedding_bge_m3_role | Y | Y | - | Y | Y | 7 | 8.97 |  |
| reranking_role | Y | Y | Y | Y | Y | 7 | 18.47 |  |
| query_planning_expansion | Y | Y | - | Y | Y | 7 | 15.73 |  |
| vector_db_chroma_role | Y | Y | - | Y | Y | 7 | 19.37 |  |
| enterprise_rag_retrieval | Y | Y | Y | Y | Y | 7 | 32.89 |  |
| knowledge_boundary_weather | Y | Y | - | Y | Y | 7 | 13.46 |  |