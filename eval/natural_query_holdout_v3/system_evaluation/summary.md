# Candidate Generator Evaluation (LLM-Judged Union Pool)

## Setup

- cases: 32
- evaluable cases: 32
- candidate-pool coverage gaps: 0
- pooled query/chunk pairs: 320
- relevant threshold: >= 2
- qrels: eval\benchmarks\rag_natural_query_holdout_v3\qrels_llm.jsonl
- qrels type: complete_blind_llm_judgments
- independent latency manifest: data\runtime\planner_v3_holdout_latency\pool_manifest.jsonl

## Results

| system | Recall@5 | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds | max seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_hybrid | 0.664 | 1.000 | 0.503 | 0.839 | 0.843 | 0.70 | 8.85 |
| planned_v3_hybrid | 0.664 | 1.000 | 0.503 | 0.839 | 0.843 | 0.53 | 9.05 |

## Per-case Recall@10 / nDCG@10

| case | direct_hybrid | planned_v3_hybrid |
|---|---:|---:|
| natural_holdout3_001 | 1.000 / 0.986 | 1.000 / 0.986 |
| natural_holdout3_002 | 1.000 / 0.996 | 1.000 / 0.996 |
| natural_holdout3_003 | 1.000 / 0.885 | 1.000 / 0.885 |
| natural_holdout3_004 | 1.000 / 0.621 | 1.000 / 0.621 |
| natural_holdout3_005 | 1.000 / 0.869 | 1.000 / 0.869 |
| natural_holdout3_006 | 1.000 / 0.572 | 1.000 / 0.572 |
| natural_holdout3_007 | 1.000 / 0.986 | 1.000 / 0.986 |
| natural_holdout3_008 | 1.000 / 0.644 | 1.000 / 0.644 |
| natural_holdout3_009 | 1.000 / 0.914 | 1.000 / 0.914 |
| natural_holdout3_010 | 1.000 / 0.927 | 1.000 / 0.927 |
| natural_holdout3_011 | 1.000 / 0.939 | 1.000 / 0.939 |
| natural_holdout3_012 | 1.000 / 0.968 | 1.000 / 0.968 |
| natural_holdout3_013 | 1.000 / 0.686 | 1.000 / 0.686 |
| natural_holdout3_014 | 1.000 / 0.966 | 1.000 / 0.966 |
| natural_holdout3_015 | 1.000 / 0.968 | 1.000 / 0.968 |
| natural_holdout3_016 | 1.000 / 0.631 | 1.000 / 0.631 |
| natural_holdout3_017 | 1.000 / 0.792 | 1.000 / 0.792 |
| natural_holdout3_018 | 1.000 / 0.570 | 1.000 / 0.570 |
| natural_holdout3_019 | 1.000 / 0.765 | 1.000 / 0.765 |
| natural_holdout3_020 | 1.000 / 0.710 | 1.000 / 0.710 |
| natural_holdout3_021 | 1.000 / 0.882 | 1.000 / 0.882 |
| natural_holdout3_022 | 1.000 / 0.943 | 1.000 / 0.943 |
| natural_holdout3_023 | 1.000 / 0.942 | 1.000 / 0.942 |
| natural_holdout3_024 | 1.000 / 0.957 | 1.000 / 0.957 |
| natural_holdout3_025 | 1.000 / 0.690 | 1.000 / 0.690 |
| natural_holdout3_026 | 1.000 / 0.777 | 1.000 / 0.777 |
| natural_holdout3_027 | 1.000 / 0.811 | 1.000 / 0.811 |
| natural_holdout3_028 | 1.000 / 1.000 | 1.000 / 1.000 |
| natural_holdout3_029 | 1.000 / 0.855 | 1.000 / 0.855 |
| natural_holdout3_030 | 1.000 / 0.794 | 1.000 / 0.794 |
| natural_holdout3_031 | 1.000 / 0.960 | 1.000 / 0.960 |
| natural_holdout3_032 | 1.000 / 0.985 | 1.000 / 0.985 |

## Interpretation Boundary

Metrics are computed against a depth-10 union pool, not exhaustive judgments over all collection chunks. LLM labels are consistent across the pooled pairs but are not a substitute for independent human judgments.
Use the result to choose which candidate generators deserve human validation and online testing.
