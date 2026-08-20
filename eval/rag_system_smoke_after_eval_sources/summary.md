# RAG System Smoke Evaluation

## Summary

- total: 1
- passed: 0 (0.00%)
- quality_pass: 0 (0.00%)
- avg_seconds: 53.69
- llm_provider: deepseek
- retrieval_mode: planned
- rerank_mode: lexical
- top_k/candidate_k: 7/16
- max_context_chars: 15000
- audit_answer: True

## Case Results

| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rag_evaluation_reliability | N | N | Y | Y | Y | 7 | 53.69 | quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a... |

## Badcases

### rag_evaluation_reliability

- question: 如何评估RAG答案是否可靠？需要看哪些指标和badcase？
- quality: {"quality_pass": false, "overall_pass": true, "coverage_pass": false, "faithfulness_score": 5, "citation_score": 5, "relevance_score": 5, "repair": {"attempted": true, "used": true, "original_quality_pass": false}, "a...
