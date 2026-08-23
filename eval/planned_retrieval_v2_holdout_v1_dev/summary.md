# Planned Retrieval V2 Development A/B

This is a development/calibration result, not an independent holdout.
Anchored v2 keeps the original query dominant, caps total expansion weight, deduplicates runs,
and limits forced plan coverage to at most two slots.

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| direct_hybrid | 0.797 | 0.487 | 0.812 | 0.730 |
| planned_hybrid | 0.413 | 0.281 | 0.498 | 0.453 |
| planned_v2_hybrid | 0.803 | 0.494 | 0.823 | 0.726 |

Anchored v2 median latency: 4.77s.

## Per-case nDCG@10

| case | direct | legacy planned | anchored v2 | v2 seconds |
|---|---:|---:|---:|---:|
| holdout_rag_limits | 0.554 | 0.231 | 0.565 | 5.74 |
| holdout_recursive_splitter | 0.775 | 0.757 | 0.825 | 1.86 |
| holdout_bge_m3_modes | 0.666 | 0.870 | 0.866 | 3.00 |
| holdout_chroma_filters | 0.563 | 0.350 | 0.499 | 5.88 |
| holdout_colbert_interaction | 0.916 | 0.691 | 0.902 | 1.74 |
| holdout_faithfulness_metric | 0.618 | 0.515 | 0.776 | 6.20 |
| holdout_context_precision | 0.860 | 0.496 | 0.822 | 4.57 |
| holdout_ollama_embedding | 0.692 | 0.204 | 0.575 | 5.16 |
| holdout_ingestion_design | 0.771 | 0.670 | 0.780 | 10.55 |
| holdout_hybrid_tradeoffs | 0.936 | 0.142 | 0.870 | 4.58 |
| holdout_evaluation_layers | 0.369 | 0.819 | 0.560 | 4.89 |
| holdout_wrong_citation_diagnosis | 0.703 | 0.246 | 0.672 | 4.42 |
| holdout_topk_rerank_context | 0.625 | 0.301 | 0.594 | 6.91 |
| holdout_incremental_updates | 0.937 | 0.161 | 0.631 | 4.95 |
| holdout_lost_in_middle | 0.719 | 0.798 | 0.834 | 4.66 |
| holdout_rag_architecture_choice | 0.974 | 0.000 | 0.848 | 3.92 |

All ranked chunks were required to have complete qrels before metrics were computed.
A separate untouched holdout is required after development decisions are finalized.
