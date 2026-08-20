# RAG System Smoke Evaluation

## Summary

- total: 2
- passed: 2 (100.00%)
- quality_pass: 2 (100.00%)
- avg_seconds: 23.25
- llm_provider: deepseek
- retrieval_mode: planned
- rerank_mode: lexical
- top_k/candidate_k: 7/16
- max_context_chars: 15000
- audit_answer: True

## Case Results

| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| vector_db_chroma_role | Y | Y | - | Y | Y | 7 | 23.99 |  |
| enterprise_rag_retrieval | Y | Y | Y | Y | Y | 7 | 22.51 |  |