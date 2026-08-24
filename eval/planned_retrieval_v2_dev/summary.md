# Planned Retrieval V2 Development A/B

This is a development/calibration result, not an independent holdout.
Anchored v2 keeps the original query dominant, caps total expansion weight, deduplicates runs,
and limits forced plan coverage to at most two slots.

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| direct_hybrid | 0.400 | 0.550 | 1.000 | 0.576 |
| planned_hybrid | 0.451 | 0.600 | 0.854 | 0.614 |
| planned_v2_hybrid | 0.403 | 0.550 | 0.917 | 0.552 |

Anchored v2 median latency: 10.28s.

## Per-case nDCG@10

| case | direct | legacy planned | anchored v2 | v2 seconds |
|---|---:|---:|---:|---:|
| rag_compound_overview | 0.333 | 0.469 | 0.392 | 11.76 |
| chunking_not_fixed_window | 0.436 | 0.651 | 0.359 | 6.85 |
| rag_evaluation_reliability | 0.605 | 0.671 | 0.536 | 29.20 |
| embedding_bge_m3_role | 0.673 | 0.828 | 0.688 | 8.48 |
| reranking_role | 0.408 | 0.515 | 0.455 | 16.97 |
| query_planning_expansion | 0.679 | 0.590 | 0.671 | 8.79 |
| vector_db_chroma_role | 0.894 | 0.652 | 0.784 | 7.68 |
| enterprise_rag_retrieval | 0.577 | 0.539 | 0.531 | 20.83 |

All ranked chunks were required to have complete qrels before metrics were computed.
A separate untouched holdout is required after development decisions are finalized.
