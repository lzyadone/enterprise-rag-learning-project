# Automatic Retrieval Routing Holdout

## Setup

- cases: 16
- evaluation role: holdout
- qrels: eval\benchmarks\rag_routing_holdout_v1\qrels_llm.jsonl
- latency budget: 12000 ms
- route counts: {'direct': 9, 'planned': 7}
- optimized planned latency used: False

## Results

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds |
|---|---:|---:|---:|---:|---:|
| direct_hybrid | 0.862 | 0.487 | 0.812 | 0.750 | 0.83 |
| planned_hybrid | 0.448 | 0.281 | 0.498 | 0.465 | 4.62 |
| auto | 0.673 | 0.369 | 0.652 | 0.596 | 1.32 |

Oracle agreement: 10/16 (62.5%).

## Per-case Decisions

| case | selected | oracle | score | nDCG direct/planned | seconds | reasons |
|---|---|---|---:|---:|---:|---|
| holdout_rag_limits | planned_hybrid | direct_hybrid | 3 | 0.674 / 0.281 | 6.43 | complexity_threshold_reached, aspect_detected_with_low_category_confidence |
| holdout_recursive_splitter | direct_hybrid | direct_hybrid | 0 | 0.775 / 0.757 | 0.81 | simple_or_specific_query |
| holdout_bge_m3_modes | planned_hybrid | planned_hybrid | 3 | 0.666 / 0.870 | 3.72 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| holdout_chroma_filters | direct_hybrid | direct_hybrid | 0 | 0.600 / 0.373 | 0.62 | simple_or_specific_query |
| holdout_colbert_interaction | direct_hybrid | direct_hybrid | 0 | 0.916 / 0.691 | 0.73 | simple_or_specific_query |
| holdout_faithfulness_metric | direct_hybrid | direct_hybrid | 0 | 0.618 / 0.515 | 0.95 | simple_or_specific_query |
| holdout_context_precision | direct_hybrid | direct_hybrid | 0 | 0.860 / 0.496 | 0.85 | simple_or_specific_query |
| holdout_ollama_embedding | direct_hybrid | direct_hybrid | 0 | 0.692 / 0.204 | 1.19 | simple_or_specific_query |
| holdout_ingestion_design | planned_hybrid | direct_hybrid | 4 | 0.821 / 0.714 | 7.27 | complexity_threshold_reached, multiple_answer_aspects |
| holdout_hybrid_tradeoffs | planned_hybrid | direct_hybrid | 6 | 0.936 / 0.142 | 5.12 | complexity_threshold_reached, multiple_answer_aspects, cross_category_comparison |
| holdout_evaluation_layers | direct_hybrid | planned_hybrid | 0 | 0.385 / 0.855 | 1.02 | simple_or_specific_query |
| holdout_wrong_citation_diagnosis | planned_hybrid | direct_hybrid | 3 | 0.734 / 0.257 | 4.15 | complexity_threshold_reached, aspect_detected_with_low_category_confidence |
| holdout_topk_rerank_context | direct_hybrid | direct_hybrid | 0 | 0.655 / 0.315 | 0.79 | simple_or_specific_query |
| holdout_incremental_updates | direct_hybrid | direct_hybrid | 0 | 0.978 / 0.168 | 1.45 | simple_or_specific_query |
| holdout_lost_in_middle | planned_hybrid | planned_hybrid | 3 | 0.719 / 0.798 | 2.80 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| holdout_rag_architecture_choice | planned_hybrid | direct_hybrid | 4 | 0.974 / 0.000 | 3.34 | complexity_threshold_reached, multiple_answer_aspects |

## Interpretation Boundary

Questions were frozen before candidate generation and were not used to tune the current routing threshold. LLM relevance labels are an independent holdout signal, but a future human audit is still required.
