# Answer Faithfulness Smoke Evaluation

## Summary

- total: 4
- llm_provider: deepseek
- llm_audit: True
- repair_with_deepseek: False
- top_k: 5
- candidate_k: 12
- rule pass: 4 (100.00%)
- overall pass: 1 (25.00%)
- audit errors: 0
- repair errors: 0

## Cases

### 1. 文档切分为什么不能只用固定窗口？

- overall_pass: False
- rule_pass: True
- repair_error: none
- audit_error: none
- sources: LangChain Text Splitter Integrations, Haystack DocumentSplitter, LangChain Recursive Text Splitter, LangChain Text Splitter Integrations, LangChain Text Splitter Integrations

### 2. metadata filter 在 RAG 检索中有什么作用？

- overall_pass: False
- rule_pass: True
- repair_error: none
- audit_error: none
- sources: Chroma Docs, MongoDB Self-Query Retrieval with LangChain, MongoDB Self-Query Retrieval with LangChain, MongoDB Self-Query Retrieval with LangChain, MongoDB Self-Query Retrieval with LangChain

### 3. rerank 和普通向量检索有什么关系？

- overall_pass: False
- rule_pass: True
- repair_error: none
- audit_error: none
- sources: Cohere Rerank, MongoDB Self-Query Retrieval with LangChain, Cohere Rerank, Haystack Rankers, Cohere Rerank

### 4. 如何评估 RAG 回答是否忠实于检索上下文？

- overall_pass: True
- rule_pass: True
- repair_error: none
- audit_error: none
- sources: LangSmith Evaluate RAG Tutorial, LangChain Retrieval, LangSmith Evaluate RAG Tutorial, LangSmith Evaluate RAG Tutorial, LangChain Retrieval
