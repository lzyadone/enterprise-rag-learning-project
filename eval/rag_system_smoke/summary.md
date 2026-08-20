# RAG System Smoke Evaluation

## Summary

- total: 4
- passed: 3 (75.00%)
- quality_pass: 3 (75.00%)
- avg_seconds: 39.81
- llm_provider: deepseek
- retrieval_mode: planned
- rerank_mode: lexical
- top_k/candidate_k: 7/16
- max_context_chars: 15000
- audit_answer: True

## Case Results

| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rag_compound_overview | Y | Y | Y | Y | Y | 10 | 70.13 |  |
| chunking_not_fixed_window | Y | Y | Y | Y | Y | 7 | 29.08 |  |
| rag_evaluation_reliability | N | N | Y | Y | Y | 7 | 54.13 | quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a... |
| long_memory_learning_preference | Y | Y | - | - | Y | 0 | 5.89 |  |

## Badcases

### rag_evaluation_reliability

- question: 如何评估RAG答案是否可靠？需要看哪些指标和badcase？
- quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a...
