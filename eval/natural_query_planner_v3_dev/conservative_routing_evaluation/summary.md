# Automatic Retrieval Routing Development

## Setup

- cases: 32
- evaluable cases: 30
- candidate-pool coverage gaps: 2
- evaluable pool rate: 93.8%
- evaluation role: development
- routing planner: conservative
- qrels: eval\benchmarks\rag_natural_query_planner_v3_dev\qrels_llm.jsonl
- latency budget: 12000 ms
- direct system: direct_hybrid
- planned system: planned_v3_hybrid
- route counts: {'direct': 28, 'planned': 4}
- optimized planned latency used: True

## Results

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds |
|---|---:|---:|---:|---:|---:|
| direct_hybrid | 0.977 | 0.477 | 0.865 | 0.837 | 0.52 |
| planned_v3_hybrid | 0.992 | 0.483 | 0.882 | 0.853 | 0.48 |
| auto | 0.992 | 0.483 | 0.882 | 0.853 | 0.52 |

Oracle agreement: 28/30 (93.3%).

## Diagnostic Thresholds

Overall: PASS

| check | passed | actual | required |
|---|---|---:|---:|
| auto_ndcg_noninferior | True | 0.8533 | 0.8173 |
| auto_recall_noninferior | True | 0.9915 | 0.9567 |
| auto_mrr_noninferior | True | 0.8819 | 0.8153 |
| oracle_agreement | True | 0.9333 | 0.75 |
| planned_route_worst_case | True | 0 | 0 |
| median_latency_within_budget | True | 0.5239 | 12.0 |

## Per-case Decisions

| case | selected | oracle | score | nDCG direct/planned | seconds | reasons |
|---|---|---|---:|---:|---:|---|
| natural_dev_001 | direct_hybrid | direct_hybrid | 0 | 0.825 / 0.825 | 8.36 | simple_or_specific_query |
| natural_dev_002 | direct_hybrid | direct_hybrid | 0 | 0.817 / 0.817 | 0.44 | simple_or_specific_query |
| natural_dev_003 | direct_hybrid | direct_hybrid | 0 | 0.943 / 0.943 | 0.56 | simple_or_specific_query |
| natural_dev_004 | direct_hybrid | direct_hybrid | 0 | 0.708 / 0.708 | 0.52 | simple_or_specific_query |
| natural_dev_005 | direct_hybrid | direct_hybrid | 0 | 0.955 / 0.955 | 0.37 | simple_or_specific_query |
| natural_dev_006 | direct_hybrid | direct_hybrid | 0 | 0.631 / 0.631 | 0.47 | simple_or_specific_query |
| natural_dev_007 | direct_hybrid | coverage_gap | 0 | 0.000 / 0.000 | 0.42 | simple_or_specific_query |
| natural_dev_008 | direct_hybrid | direct_hybrid | 0 | 0.838 / 0.838 | 0.51 | simple_or_specific_query |
| natural_dev_009 | direct_hybrid | coverage_gap | 0 | 0.000 / 0.000 | 0.52 | simple_or_specific_query |
| natural_dev_010 | direct_hybrid | direct_hybrid | 0 | 0.904 / 0.904 | 0.46 | simple_or_specific_query |
| natural_dev_011 | direct_hybrid | direct_hybrid | 0 | 0.767 / 0.767 | 0.52 | simple_or_specific_query |
| natural_dev_012 | direct_hybrid | direct_hybrid | 0 | 0.573 / 0.573 | 0.46 | simple_or_specific_query |
| natural_dev_013 | direct_hybrid | direct_hybrid | 0 | 1.000 / 1.000 | 0.42 | simple_or_specific_query |
| natural_dev_014 | direct_hybrid | direct_hybrid | 0 | 0.562 / 0.562 | 0.78 | simple_or_specific_query |
| natural_dev_016 | direct_hybrid | direct_hybrid | 0 | 0.824 / 0.824 | 0.45 | simple_or_specific_query |
| natural_dev_018 | direct_hybrid | direct_hybrid | 0 | 0.939 / 0.939 | 0.42 | simple_or_specific_query |
| natural_dev_019 | direct_hybrid | direct_hybrid | 0 | 0.740 / 0.740 | 0.46 | simple_or_specific_query |
| natural_dev_020 | direct_hybrid | direct_hybrid | 0 | 0.946 / 0.946 | 0.53 | simple_or_specific_query |
| natural_dev_021 | planned_v3_hybrid | planned_v3_hybrid | 4 | 0.744 / 0.797 | 2.94 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_022 | direct_hybrid | direct_hybrid | 0 | 0.965 / 0.965 | 0.73 | simple_or_specific_query |
| natural_dev_023 | direct_hybrid | direct_hybrid | 0 | 0.905 / 0.905 | 0.61 | simple_or_specific_query |
| natural_dev_024 | planned_v3_hybrid | direct_hybrid | 4 | 0.858 / 0.858 | 2.11 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_025 | direct_hybrid | direct_hybrid | 0 | 0.931 / 0.931 | 0.79 | simple_or_specific_query |
| natural_dev_026 | direct_hybrid | direct_hybrid | 0 | 0.949 / 0.949 | 0.49 | simple_or_specific_query |
| natural_dev_027 | planned_v3_hybrid | direct_hybrid | 4 | 0.941 / 0.920 | 1.94 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_028 | direct_hybrid | direct_hybrid | 0 | 0.981 / 0.981 | 0.52 | simple_or_specific_query |
| natural_dev_029 | direct_hybrid | direct_hybrid | 0 | 1.000 / 1.000 | 0.67 | simple_or_specific_query |
| natural_dev_030 | planned_v3_hybrid | planned_v3_hybrid | 4 | 0.469 / 0.915 | 1.81 | complexity_threshold_reached, multiple_answer_aspects |
| natural_dev_031 | direct_hybrid | direct_hybrid | 0 | 0.938 / 0.938 | 0.81 | simple_or_specific_query |
| natural_dev_032 | direct_hybrid | direct_hybrid | 2 | 0.797 / 0.797 | 0.60 | simple_or_specific_query, cross_category_comparison |
| natural_dev_033 | direct_hybrid | direct_hybrid | 0 | 0.873 / 0.873 | 0.45 | simple_or_specific_query |
| natural_dev_037 | direct_hybrid | direct_hybrid | 0 | 0.801 / 0.801 | 0.73 | simple_or_specific_query |

## Interpretation Boundary

These questions are development data generated to diagnose natural-language routing. They may be used for later tuning, so their metrics must not be reported as holdout evidence.
