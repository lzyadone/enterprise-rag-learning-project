# Candidate Generator Evaluation (LLM-Judged Union Pool)

## Setup

- cases: 32
- evaluable cases: 30
- candidate-pool coverage gaps: 2
- pooled query/chunk pairs: 351
- relevant threshold: >= 2
- qrels: eval\benchmarks\rag_natural_query_planner_v3_dev\qrels_llm.jsonl
- qrels type: complete_blind_llm_judgments
- independent latency manifest: data\runtime\planner_v3_latency\pool_manifest.jsonl

## Results

| system | Recall@5 | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds | max seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_hybrid | 0.633 | 0.977 | 0.477 | 0.865 | 0.837 | 0.52 | 8.36 |
| planned_v2_hybrid | 0.568 | 0.925 | 0.453 | 0.691 | 0.737 | 2.02 | 4.75 |
| planned_v3_hybrid | 0.648 | 0.992 | 0.483 | 0.882 | 0.853 | 0.48 | 7.73 |

## Per-case Recall@10 / nDCG@10

| case | direct_hybrid | planned_v2_hybrid | planned_v3_hybrid |
|---|---:|---:|---:|
| natural_dev_001 | 1.000 / 0.825 | 1.000 / 0.858 | 1.000 / 0.825 |
| natural_dev_002 | 1.000 / 0.817 | 0.857 / 0.609 | 1.000 / 0.817 |
| natural_dev_003 | 1.000 / 0.943 | 0.800 / 0.900 | 1.000 / 0.943 |
| natural_dev_004 | 1.000 / 0.708 | 0.750 / 0.504 | 1.000 / 0.708 |
| natural_dev_005 | 1.000 / 0.955 | 1.000 / 0.778 | 1.000 / 0.955 |
| natural_dev_006 | 1.000 / 0.631 | 1.000 / 0.431 | 1.000 / 0.631 |
| natural_dev_007 | coverage gap | coverage gap | coverage gap |
| natural_dev_008 | 1.000 / 0.838 | 1.000 / 0.779 | 1.000 / 0.838 |
| natural_dev_009 | coverage gap | coverage gap | coverage gap |
| natural_dev_010 | 1.000 / 0.904 | 0.889 / 0.694 | 1.000 / 0.904 |
| natural_dev_011 | 1.000 / 0.767 | 1.000 / 0.674 | 1.000 / 0.767 |
| natural_dev_012 | 1.000 / 0.573 | 1.000 / 0.509 | 1.000 / 0.573 |
| natural_dev_013 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| natural_dev_014 | 1.000 / 0.562 | 0.667 / 0.583 | 1.000 / 0.562 |
| natural_dev_016 | 1.000 / 0.824 | 0.857 / 0.654 | 1.000 / 0.824 |
| natural_dev_018 | 1.000 / 0.939 | 1.000 / 0.901 | 1.000 / 0.939 |
| natural_dev_019 | 1.000 / 0.740 | 1.000 / 0.555 | 1.000 / 0.740 |
| natural_dev_020 | 1.000 / 0.946 | 1.000 / 0.955 | 1.000 / 0.946 |
| natural_dev_021 | 0.778 / 0.744 | 0.889 / 0.808 | 0.889 / 0.797 |
| natural_dev_022 | 1.000 / 0.965 | 0.833 / 0.734 | 1.000 / 0.965 |
| natural_dev_023 | 1.000 / 0.905 | 1.000 / 0.929 | 1.000 / 0.905 |
| natural_dev_024 | 1.000 / 0.858 | 1.000 / 0.980 | 1.000 / 0.858 |
| natural_dev_025 | 1.000 / 0.931 | 1.000 / 0.887 | 1.000 / 0.931 |
| natural_dev_026 | 0.857 / 0.949 | 1.000 / 0.760 | 0.857 / 0.949 |
| natural_dev_027 | 1.000 / 0.941 | 1.000 / 0.725 | 1.000 / 0.920 |
| natural_dev_028 | 1.000 / 0.981 | 1.000 / 0.844 | 1.000 / 0.981 |
| natural_dev_029 | 1.000 / 1.000 | 1.000 / 0.749 | 1.000 / 1.000 |
| natural_dev_030 | 0.667 / 0.469 | 0.333 / 0.169 | 1.000 / 0.915 |
| natural_dev_031 | 1.000 / 0.938 | 0.889 / 0.813 | 1.000 / 0.938 |
| natural_dev_032 | 1.000 / 0.797 | 1.000 / 0.727 | 1.000 / 0.797 |
| natural_dev_033 | 1.000 / 0.873 | 1.000 / 0.818 | 1.000 / 0.873 |
| natural_dev_037 | 1.000 / 0.801 | 1.000 / 0.777 | 1.000 / 0.801 |

## Interpretation Boundary

Metrics are computed against a depth-10 union pool, not exhaustive judgments over all collection chunks. LLM labels are consistent across the pooled pairs but are not a substitute for independent human judgments.
Use the result to choose which candidate generators deserve human validation and online testing.
