# Planned Retrieval Smoke Evaluation

## Summary

- total: 8
- direct hit@1: 4 (50.00%)
- planned hit@1: 8 (100.00%)
- direct hit@3: 6 (75.00%)
- planned hit@3: 8 (100.00%)

## Cases

### 1. RAG 的完整流程包括哪些阶段？

- expected: RAG overview
- plan categories: RAG overview
- direct categories: RAG overview, RAG overview, evaluation
- planned categories: RAG overview, RAG overview, RAG overview
- direct hit@1: True
- planned hit@1: True

### 2. 文档切分为什么不能只用固定窗口？

- expected: chunking
- plan categories: chunking
- direct categories: evaluation, chunking, chunking
- planned categories: chunking, chunking, chunking
- direct hit@1: False
- planned hit@1: True

### 3. metadata filter 在 RAG 检索中有什么作用？

- expected: vector db, retrieval
- plan categories: vector db, retrieval
- direct categories: RAG overview, RAG paper, RAG overview
- planned categories: vector db, vector db, retrieval
- direct hit@1: False
- planned hit@1: True

### 4. 如何评估 RAG 回答是否忠实于检索上下文？

- expected: evaluation
- plan categories: evaluation, RAG overview
- direct categories: RAG paper, evaluation, evaluation
- planned categories: evaluation, evaluation, RAG overview
- direct hit@1: False
- planned hit@1: True

### 5. bge-m3 这种 embedding 模型在知识库里负责什么？

- expected: embedding
- plan categories: embedding, RAG overview
- direct categories: embedding, embedding, embedding
- planned categories: embedding, embedding, embedding
- direct hit@1: True
- planned hit@1: True

### 6. rerank 和普通向量检索有什么关系？

- expected: reranking, retrieval
- plan categories: reranking, retrieval
- direct categories: indexing, evaluation, RAG paper
- planned categories: retrieval, reranking, reranking
- direct hit@1: False
- planned hit@1: True

### 7. Ollama 本地 API 怎么用于生成或 embedding？

- expected: local model
- plan categories: embedding, local model
- direct categories: local model, local model, local model
- planned categories: local model, local model, local model
- direct hit@1: True
- planned hit@1: True

### 8. ingestion pipeline 为什么要包含 transformation 和 metadata？

- expected: ingestion
- plan categories: ingestion, vector db
- direct categories: ingestion, ingestion, ingestion
- planned categories: ingestion, ingestion, ingestion
- direct hit@1: True
- planned hit@1: True
