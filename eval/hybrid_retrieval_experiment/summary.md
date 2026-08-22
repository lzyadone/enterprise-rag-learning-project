# Hybrid Retrieval Comparison

This experiment isolates retrieval quality before answer generation. BM25 and Chroma reuse the same chunks.

## Setup

- cases: 8
- top_k: 7
- candidate_k: 16
- dense channel: Chroma + bge-m3
- sparse channel: BM25 with English tokens and Chinese bi/tri-grams
- fusion: Reciprocal Rank Fusion (RRF)

## Variant Results

| variant | category pass | evidence-term pass | both pass | category MRR | avg term recall | avg seconds |
|---|---:|---:|---:|---:|---:|---:|
| dense | 62.50% | 100.00% | 62.50% | 0.573 | 0.738 | 0.35 |
| dense_lexical_rerank | 62.50% | 100.00% | 62.50% | 0.594 | 0.738 | 0.35 |
| hybrid | 75.00% | 100.00% | 75.00% | 0.708 | 0.838 | 0.37 |
| hybrid_lexical_rerank | 75.00% | 100.00% | 75.00% | 0.688 | 0.838 | 0.36 |
| planned_dense_lexical_rerank | 100.00% | 100.00% | 100.00% | 0.875 | 0.950 | 8.31 |
| planned_hybrid_lexical_rerank | 100.00% | 100.00% | 100.00% | 0.938 | 0.950 | 8.66 |

## Per-case Both-pass

| case | dense | dense_lexical_rerank | hybrid | hybrid_lexical_rerank | planned_dense_lexical_rerank | planned_hybrid_lexical_rerank |
|---|---:|---:|---:|---:|---:|---:|
| rag_compound_overview | FAIL | FAIL | FAIL | FAIL | PASS | PASS |
| chunking_not_fixed_window | FAIL | FAIL | PASS | PASS | PASS | PASS |
| rag_evaluation_reliability | PASS | PASS | PASS | PASS | PASS | PASS |
| embedding_bge_m3_role | PASS | PASS | PASS | PASS | PASS | PASS |
| reranking_role | PASS | PASS | PASS | PASS | PASS | PASS |
| query_planning_expansion | PASS | PASS | PASS | PASS | PASS | PASS |
| vector_db_chroma_role | PASS | PASS | PASS | PASS | PASS | PASS |
| enterprise_rag_retrieval | FAIL | FAIL | FAIL | FAIL | PASS | PASS |

## Reading the Metrics

- category pass checks whether the retrieved set covers the required knowledge areas.
- evidence-term pass checks whether the chunks contain enough expected technical evidence, not only the right label.
- category MRR rewards placing the first relevant category near rank 1.
- avg term recall measures how much of the expected evidence vocabulary appears in top-k chunks.
