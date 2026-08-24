# Automatic Retrieval Routing Holdout

## Setup

- cases: 32
- evaluable cases: 32
- candidate-pool coverage gaps: 0
- evaluable pool rate: 100.0%
- evaluation role: holdout
- routing planner: conservative
- qrels: eval\benchmarks\rag_natural_query_holdout_v3\qrels_llm.jsonl
- latency budget: 12000 ms
- direct system: direct_hybrid
- planned system: planned_v3_hybrid
- route counts: {'direct': 31, 'planned': 1}
- optimized planned latency used: True

## Results

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds |
|---|---:|---:|---:|---:|---:|
| direct_hybrid | 1.000 | 0.503 | 0.839 | 0.843 | 0.70 |
| planned_v3_hybrid | 1.000 | 0.503 | 0.839 | 0.843 | 0.53 |
| auto | 1.000 | 0.503 | 0.839 | 0.843 | 0.70 |

Oracle agreement: 31/32 (96.9%).

## Pre-registered Acceptance

Overall: PASS

| check | passed | actual | required |
|---|---|---:|---:|
| auto_ndcg_noninferior | True | 0.8434 | 0.8234 |
| auto_recall_noninferior | True | 1.0 | 0.98 |
| auto_mrr_noninferior | True | 0.8385 | 0.7885 |
| oracle_agreement | True | 0.9688 | 0.75 |
| planned_route_worst_case | True | 0 | 0 |
| median_latency_within_budget | True | 0.7018 | 12.0 |

## Per-case Decisions

| case | selected | oracle | score | nDCG direct/planned | seconds | reasons |
|---|---|---|---:|---:|---:|---|
| natural_holdout3_001 | direct_hybrid | direct_hybrid | 0 | 0.986 / 0.986 | 8.85 | simple_or_specific_query |
| natural_holdout3_002 | direct_hybrid | direct_hybrid | 0 | 0.996 / 0.996 | 0.79 | simple_or_specific_query |
| natural_holdout3_003 | direct_hybrid | direct_hybrid | 0 | 0.885 / 0.885 | 0.66 | simple_or_specific_query |
| natural_holdout3_004 | direct_hybrid | direct_hybrid | 0 | 0.621 / 0.621 | 0.67 | simple_or_specific_query |
| natural_holdout3_005 | direct_hybrid | direct_hybrid | 0 | 0.869 / 0.869 | 0.52 | simple_or_specific_query |
| natural_holdout3_006 | direct_hybrid | direct_hybrid | 0 | 0.572 / 0.572 | 0.78 | simple_or_specific_query |
| natural_holdout3_007 | direct_hybrid | direct_hybrid | 0 | 0.986 / 0.986 | 0.54 | simple_or_specific_query |
| natural_holdout3_008 | direct_hybrid | direct_hybrid | 0 | 0.644 / 0.644 | 0.69 | simple_or_specific_query |
| natural_holdout3_009 | direct_hybrid | direct_hybrid | 0 | 0.914 / 0.914 | 0.83 | simple_or_specific_query |
| natural_holdout3_010 | direct_hybrid | direct_hybrid | 0 | 0.927 / 0.927 | 0.84 | simple_or_specific_query |
| natural_holdout3_011 | direct_hybrid | direct_hybrid | 0 | 0.939 / 0.939 | 0.66 | simple_or_specific_query |
| natural_holdout3_012 | direct_hybrid | direct_hybrid | 0 | 0.968 / 0.968 | 0.69 | simple_or_specific_query |
| natural_holdout3_013 | direct_hybrid | direct_hybrid | 0 | 0.686 / 0.686 | 0.75 | simple_or_specific_query |
| natural_holdout3_014 | planned_v3_hybrid | direct_hybrid | 3 | 0.966 / 0.966 | 0.48 | complexity_threshold_reached, aspect_detected_with_low_category_confidence |
| natural_holdout3_015 | direct_hybrid | direct_hybrid | 0 | 0.968 / 0.968 | 0.94 | simple_or_specific_query |
| natural_holdout3_016 | direct_hybrid | direct_hybrid | 0 | 0.631 / 0.631 | 0.61 | simple_or_specific_query |
| natural_holdout3_017 | direct_hybrid | direct_hybrid | 0 | 0.792 / 0.792 | 0.52 | simple_or_specific_query |
| natural_holdout3_018 | direct_hybrid | direct_hybrid | 0 | 0.570 / 0.570 | 0.63 | simple_or_specific_query |
| natural_holdout3_019 | direct_hybrid | direct_hybrid | 0 | 0.765 / 0.765 | 0.55 | simple_or_specific_query |
| natural_holdout3_020 | direct_hybrid | direct_hybrid | 0 | 0.710 / 0.710 | 0.70 | simple_or_specific_query |
| natural_holdout3_021 | direct_hybrid | direct_hybrid | 0 | 0.882 / 0.882 | 0.71 | simple_or_specific_query |
| natural_holdout3_022 | direct_hybrid | direct_hybrid | 0 | 0.943 / 0.943 | 0.78 | simple_or_specific_query |
| natural_holdout3_023 | direct_hybrid | direct_hybrid | 0 | 0.942 / 0.942 | 0.68 | simple_or_specific_query |
| natural_holdout3_024 | direct_hybrid | direct_hybrid | 0 | 0.957 / 0.957 | 0.70 | simple_or_specific_query |
| natural_holdout3_025 | direct_hybrid | direct_hybrid | 0 | 0.690 / 0.690 | 0.59 | simple_or_specific_query |
| natural_holdout3_026 | direct_hybrid | direct_hybrid | 2 | 0.777 / 0.777 | 0.81 | simple_or_specific_query, cross_category_comparison |
| natural_holdout3_027 | direct_hybrid | direct_hybrid | 0 | 0.811 / 0.811 | 1.43 | simple_or_specific_query |
| natural_holdout3_028 | direct_hybrid | direct_hybrid | 0 | 1.000 / 1.000 | 0.48 | simple_or_specific_query |
| natural_holdout3_029 | direct_hybrid | direct_hybrid | 0 | 0.855 / 0.855 | 0.75 | simple_or_specific_query |
| natural_holdout3_030 | direct_hybrid | direct_hybrid | 0 | 0.794 / 0.794 | 0.76 | simple_or_specific_query |
| natural_holdout3_031 | direct_hybrid | direct_hybrid | 0 | 0.960 / 0.960 | 1.60 | simple_or_specific_query |
| natural_holdout3_032 | direct_hybrid | direct_hybrid | 0 | 0.985 / 0.985 | 0.76 | simple_or_specific_query |

## Interpretation Boundary

Questions were frozen before candidate generation and were not used to tune the current routing threshold. LLM relevance labels are an independent holdout signal, but a future human audit is still required.
