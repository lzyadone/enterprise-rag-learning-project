# RAG System Smoke Evaluation

## Summary

- total: 2
- passed: 2 (100.00%)
- quality_pass: 2 (100.00%)
- avg_seconds: 27.73
- llm_provider: deepseek
- retrieval_mode: planned
- rerank_mode: lexical
- top_k/candidate_k: 7/16
- max_context_chars: 15000
- audit_answer: True

## Case Results

| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| query_planning_expansion | Y | Y | - | Y | Y | 7 | 20.75 |  |
| enterprise_rag_retrieval | Y | Y | Y | Y | Y | 7 | 34.72 |  |