# Automatic Retrieval Routing Holdout

## Setup

- cases: 20
- evaluation role: holdout
- qrels: eval\benchmarks\rag_routing_holdout_v2\qrels_llm.jsonl
- latency budget: 12000 ms
- direct system: direct_hybrid
- planned system: planned_v2_hybrid
- route counts: {'direct': 7, 'planned': 13}
- optimized planned latency used: False

## Results

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds |
|---|---:|---:|---:|---:|---:|
| direct_hybrid | 0.973 | 0.480 | 0.800 | 0.825 | 1.48 |
| planned_v2_hybrid | 0.955 | 0.475 | 0.702 | 0.752 | 4.95 |
| auto | 0.980 | 0.480 | 0.825 | 0.804 | 4.05 |

Oracle agreement: 11/20 (55.0%).

## Pre-registered Acceptance

Overall: FAIL

| check | passed | actual | required |
|---|---|---:|---:|
| auto_ndcg_noninferior | False | 0.8043 | 0.805 |
| auto_recall_noninferior | True | 0.98 | 0.9533 |
| auto_mrr_noninferior | True | 0.825 | 0.75 |
| oracle_agreement | False | 0.55 | 0.75 |
| planned_route_worst_case | True | 0 | 0 |
| median_latency_within_budget | True | 4.0519 | 12.0 |

## Per-case Decisions

| case | selected | oracle | score | nDCG direct/planned | seconds | reasons |
|---|---|---|---:|---:|---:|---|
| holdout2_loader_interface | planned_v2_hybrid | planned_v2_hybrid | 3 | 0.656 / 0.755 | 4.11 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| holdout2_collection_metadata | direct_hybrid | direct_hybrid | 0 | 0.959 / 0.524 | 1.89 | simple_or_specific_query |
| holdout2_embedding_function | planned_v2_hybrid | direct_hybrid | 3 | 0.749 / 0.628 | 4.72 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| holdout2_index_types | planned_v2_hybrid | direct_hybrid | 5 | 0.984 / 0.889 | 4.26 | complexity_threshold_reached, multi_category_expansion_without_aspects, cross_category_comparison |
| holdout2_ingestion_cache | direct_hybrid | direct_hybrid | 0 | 0.748 / 0.530 | 2.12 | simple_or_specific_query |
| holdout2_self_query | planned_v2_hybrid | direct_hybrid | 4 | 0.978 / 0.974 | 9.77 | complexity_threshold_reached, multiple_answer_aspects |
| holdout2_citation_recall | planned_v2_hybrid | direct_hybrid | 4 | 0.690 / 0.625 | 5.61 | complexity_threshold_reached, multiple_answer_aspects |
| holdout2_contextual_recall | planned_v2_hybrid | direct_hybrid | 3 | 0.854 / 0.607 | 4.63 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| holdout2_sbert_efficiency | planned_v2_hybrid | direct_hybrid | 3 | 0.951 / 0.894 | 4.83 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| holdout2_postprocessors | direct_hybrid | direct_hybrid | 0 | 0.833 / 0.766 | 2.74 | simple_or_specific_query |
| holdout2_multisource_ingestion | direct_hybrid | planned_v2_hybrid | 0 | 0.696 / 0.849 | 1.66 | simple_or_specific_query |
| holdout2_idempotent_updates | planned_v2_hybrid | planned_v2_hybrid | 3 | 0.862 / 0.929 | 4.94 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| holdout2_query_pipeline | planned_v2_hybrid | direct_hybrid | 3 | 0.979 / 0.910 | 6.21 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| holdout2_citation_precision_recall | planned_v2_hybrid | planned_v2_hybrid | 3 | 0.733 / 0.810 | 2.90 | complexity_threshold_reached, aspect_detected_with_low_category_confidence |
| holdout2_metric_diagnosis | direct_hybrid | direct_hybrid | 0 | 0.939 / 0.902 | 2.18 | simple_or_specific_query |
| holdout2_collection_lifecycle | direct_hybrid | direct_hybrid | 0 | 0.946 / 0.629 | 1.35 | simple_or_specific_query |
| holdout2_self_query_boundary | direct_hybrid | direct_hybrid | 0 | 0.826 / 0.696 | 0.75 | simple_or_specific_query |
| holdout2_index_choice | planned_v2_hybrid | planned_v2_hybrid | 3 | 0.683 / 0.777 | 3.64 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| holdout2_sbert_colbert_pipeline | planned_v2_hybrid | direct_hybrid | 4 | 0.902 / 0.756 | 7.44 | complexity_threshold_reached, multiple_answer_aspects |
| holdout2_failure_localization | planned_v2_hybrid | planned_v2_hybrid | 3 | 0.533 / 0.585 | 3.99 | complexity_threshold_reached, multi_category_expansion_without_aspects |

## Interpretation Boundary

Questions were frozen before candidate generation and were not used to tune the current routing threshold. LLM relevance labels are an independent holdout signal, but a future human audit is still required.
