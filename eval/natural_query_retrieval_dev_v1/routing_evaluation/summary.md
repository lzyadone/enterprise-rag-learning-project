# Automatic Retrieval Routing Development

## Setup

- cases: 32
- evaluable cases: 30
- candidate-pool coverage gaps: 2
- evaluable pool rate: 93.8%
- evaluation role: development
- qrels: eval\benchmarks\rag_natural_query_dev_v1\qrels_llm.jsonl
- latency budget: 12000 ms
- direct system: direct_hybrid
- planned system: planned_v2_hybrid
- route counts: {'direct': 15, 'planned': 17}
- optimized planned latency used: False

## Results

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds |
|---|---:|---:|---:|---:|---:|
| direct_hybrid | 0.991 | 0.477 | 0.865 | 0.840 | 0.55 |
| planned_v2_hybrid | 0.935 | 0.453 | 0.691 | 0.739 | 2.25 |
| auto | 0.960 | 0.463 | 0.724 | 0.769 | 1.41 |

Oracle agreement: 10/30 (33.3%).

## Diagnostic Thresholds

Overall: FAIL

| check | passed | actual | required |
|---|---|---:|---:|
| auto_ndcg_noninferior | False | 0.7688 | 0.8203 |
| auto_recall_noninferior | False | 0.9602 | 0.9711 |
| auto_mrr_noninferior | False | 0.7239 | 0.8153 |
| oracle_agreement | False | 0.3333 | 0.75 |
| planned_route_worst_case | False | 1 | 0 |
| median_latency_within_budget | True | 1.4121 | 12.0 |

## Per-case Decisions

| case | selected | oracle | score | nDCG direct/planned | seconds | reasons |
|---|---|---|---:|---:|---:|---|
| natural_dev_001 | direct_hybrid | planned_v2_hybrid | 0 | 0.825 / 0.858 | 8.27 | simple_or_specific_query |
| natural_dev_002 | planned_v2_hybrid | direct_hybrid | 3 | 0.817 / 0.609 | 1.68 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| natural_dev_003 | planned_v2_hybrid | direct_hybrid | 3 | 0.943 / 0.900 | 3.20 | complexity_threshold_reached, aspect_detected_with_low_category_confidence |
| natural_dev_004 | planned_v2_hybrid | direct_hybrid | 4 | 0.708 / 0.504 | 4.67 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_005 | planned_v2_hybrid | direct_hybrid | 3 | 0.955 / 0.778 | 1.27 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| natural_dev_006 | planned_v2_hybrid | direct_hybrid | 3 | 0.631 / 0.431 | 1.32 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| natural_dev_007 | planned_v2_hybrid | coverage_gap | 3 | 0.000 / 0.000 | 1.50 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| natural_dev_008 | direct_hybrid | direct_hybrid | 0 | 0.838 / 0.779 | 0.48 | simple_or_specific_query |
| natural_dev_009 | direct_hybrid | coverage_gap | 0 | 0.000 / 0.000 | 0.49 | simple_or_specific_query |
| natural_dev_010 | direct_hybrid | direct_hybrid | 0 | 0.904 / 0.694 | 1.28 | simple_or_specific_query |
| natural_dev_011 | direct_hybrid | direct_hybrid | 0 | 0.767 / 0.674 | 0.58 | simple_or_specific_query |
| natural_dev_012 | direct_hybrid | direct_hybrid | 0 | 0.573 / 0.509 | 0.47 | simple_or_specific_query |
| natural_dev_013 | direct_hybrid | direct_hybrid | 0 | 1.000 / 1.000 | 0.49 | simple_or_specific_query |
| natural_dev_014 | planned_v2_hybrid | planned_v2_hybrid | 3 | 0.562 / 0.583 | 1.69 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| natural_dev_016 | planned_v2_hybrid | direct_hybrid | 4 | 0.824 / 0.654 | 4.05 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_018 | planned_v2_hybrid | direct_hybrid | 4 | 0.939 / 0.901 | 1.80 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_019 | planned_v2_hybrid | direct_hybrid | 3 | 0.740 / 0.555 | 1.74 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| natural_dev_020 | direct_hybrid | planned_v2_hybrid | 0 | 0.946 / 0.955 | 0.48 | simple_or_specific_query |
| natural_dev_021 | direct_hybrid | planned_v2_hybrid | 4 | 0.771 / 0.838 | 0.78 | estimated_planned_latency_exceeds_budget, multiple_answer_aspects |
| natural_dev_022 | direct_hybrid | direct_hybrid | 0 | 0.965 / 0.734 | 0.74 | simple_or_specific_query |
| natural_dev_023 | direct_hybrid | planned_v2_hybrid | 0 | 0.905 / 0.929 | 0.54 | simple_or_specific_query |
| natural_dev_024 | direct_hybrid | planned_v2_hybrid | 0 | 0.858 / 0.980 | 0.59 | simple_or_specific_query |
| natural_dev_025 | direct_hybrid | direct_hybrid | 0 | 0.931 / 0.887 | 0.63 | simple_or_specific_query |
| natural_dev_026 | planned_v2_hybrid | direct_hybrid | 4 | 0.949 / 0.760 | 2.50 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_027 | planned_v2_hybrid | direct_hybrid | 4 | 0.941 / 0.725 | 2.34 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_028 | planned_v2_hybrid | direct_hybrid | 4 | 0.981 / 0.844 | 2.40 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_029 | planned_v2_hybrid | direct_hybrid | 4 | 1.000 / 0.749 | 2.49 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_030 | direct_hybrid | direct_hybrid | 0 | 0.531 / 0.192 | 0.45 | simple_or_specific_query |
| natural_dev_031 | direct_hybrid | direct_hybrid | 0 | 0.938 / 0.813 | 0.72 | simple_or_specific_query |
| natural_dev_032 | planned_v2_hybrid | direct_hybrid | 5 | 0.797 / 0.727 | 1.94 | complexity_threshold_reached, multi_category_expansion_without_aspects, cross_category_comparison |
| natural_dev_033 | planned_v2_hybrid | direct_hybrid | 3 | 0.873 / 0.818 | 2.36 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| natural_dev_037 | planned_v2_hybrid | direct_hybrid | 4 | 0.801 / 0.777 | 2.71 | complexity_threshold_reached, multiple_answer_aspects |

## Interpretation Boundary

These questions are development data generated to diagnose natural-language routing. They may be used for later tuning, so their metrics must not be reported as holdout evidence.
