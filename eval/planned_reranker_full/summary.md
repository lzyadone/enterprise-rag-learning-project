# Cross-encoder Reranker Comparison

Each query uses one fixed hybrid candidate pool. Only the reranking method changes.

## Setup

- cases: 8
- collection chunks: 938
- retrieval strategy: planned-hybrid
- top_k/candidate_k: 7/16
- rerankers: cross_encoder_multilingual=BAAI/bge-reranker-v2-m3 (transformers/cuda:0), cross_encoder_fused=BAAI/bge-reranker-v2-m3 (transformers/cuda:0)
- warmup seconds: {"cross_encoder_multilingual": 71.4015, "cross_encoder_fused": 0.0848}
- nDCG relevance: category match plus expected evidence-term coverage within the fixed pool

## Results

| mode | both pass | nDCG@k | category MRR | term recall | top-1 grade | categories | rerank seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 100.00% | 0.725 | 0.938 | 0.950 | 2.833 | 3.38 | 0.00 |
| lexical | 100.00% | 0.722 | 0.938 | 0.950 | 2.900 | 3.38 | 0.01 |
| cross_encoder_multilingual | 100.00% | 0.700 | 0.938 | 0.950 | 2.892 | 3.50 | 4.79 |
| cross_encoder_fused | 100.00% | 0.701 | 0.875 | 0.950 | 2.550 | 3.50 | 4.83 |

## Per-case nDCG@k

| case | none | lexical | cross_encoder_multilingual | cross_encoder_fused |
|---|---:|---:|---:|---:|
| rag_compound_overview | 0.810 | 0.749 | 0.827 | 0.749 |
| chunking_not_fixed_window | 0.616 | 0.686 | 0.201 | 0.502 |
| rag_evaluation_reliability | 0.858 | 0.858 | 0.858 | 0.858 |
| embedding_bge_m3_role | 0.595 | 0.595 | 0.783 | 0.777 |
| reranking_role | 0.637 | 0.576 | 0.541 | 0.540 |
| query_planning_expansion | 0.822 | 0.843 | 0.888 | 0.829 |
| vector_db_chroma_role | 0.614 | 0.620 | 0.787 | 0.618 |
| enterprise_rag_retrieval | 0.846 | 0.846 | 0.713 | 0.735 |

## Interpretation Boundary

These are automatic pool-based relevance grades, not exhaustive human judgments over all 938 chunks.
Use them to compare ordering on the same candidates; inspect changed rankings before deciding the default reranker.
