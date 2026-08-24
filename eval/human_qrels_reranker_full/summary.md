# Cross-encoder Reranker Comparison

Each query uses one fixed hybrid candidate pool. Only the reranking method changes.

## Setup

- cases: 8
- collection chunks: 938
- retrieval strategy: planned-hybrid
- top_k/candidate_k: 7/16
- relevance source: human_qrels
- relevant threshold: >= 2
- qrels: eval\benchmarks\rag_retrieval_v1\qrels.jsonl
- rerankers: cross_encoder_multilingual=BAAI/bge-reranker-v2-m3 (transformers/cuda:0), cross_encoder_fused=BAAI/bge-reranker-v2-m3 (transformers/cuda:0)
- warmup seconds: {"cross_encoder_multilingual": 66.1274, "cross_encoder_fused": 0.058}
- metric note: Metrics use complete human qrels within each fixed candidate pool.

## Results

| mode | Recall@k | Precision@k | MRR | nDCG@k | top-1 grade | proxy both pass | rerank seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.504 | 0.839 | 0.875 | 0.750 | 2.500 | 100.00% | 0.01 |
| lexical | 0.504 | 0.839 | 1.000 | 0.758 | 2.750 | 100.00% | 0.01 |
| cross_encoder_multilingual | 0.475 | 0.786 | 0.938 | 0.735 | 2.625 | 100.00% | 5.17 |
| cross_encoder_fused | 0.507 | 0.839 | 0.938 | 0.747 | 2.375 | 100.00% | 5.24 |

## Per-case nDCG@k / Recall@k

| case | none | lexical | cross_encoder_multilingual | cross_encoder_fused |
|---|---:|---:|---:|---:|
| rag_compound_overview | 0.882 / 0.462 | 0.774 / 0.462 | 0.869 / 0.462 | 0.774 / 0.462 |
| chunking_not_fixed_window | 0.497 / 0.500 | 0.615 / 0.500 | 0.516 / 0.300 | 0.536 / 0.400 |
| rag_evaluation_reliability | 0.866 / 0.429 | 0.866 / 0.429 | 0.798 / 0.429 | 0.866 / 0.429 |
| embedding_bge_m3_role | 0.794 / 0.538 | 0.794 / 0.538 | 0.895 / 0.462 | 0.853 / 0.538 |
| reranking_role | 0.611 / 0.500 | 0.626 / 0.500 | 0.778 / 0.625 | 0.622 / 0.625 |
| query_planning_expansion | 0.687 / 0.600 | 0.720 / 0.600 | 0.687 / 0.600 | 0.746 / 0.600 |
| vector_db_chroma_role | 0.712 / 0.462 | 0.721 / 0.462 | 0.717 / 0.462 | 0.717 / 0.462 |
| enterprise_rag_retrieval | 0.950 / 0.538 | 0.950 / 0.538 | 0.616 / 0.462 | 0.860 / 0.538 |

## Interpretation Boundary

Human qrels exhaustively cover each fixed 16-candidate pool, but not all collection chunks.
Recall@k therefore measures reranking recall inside the judged pool, not end-to-end corpus recall.
Use this report to choose ordering; evaluate candidate generation on a judged union pool separately.
