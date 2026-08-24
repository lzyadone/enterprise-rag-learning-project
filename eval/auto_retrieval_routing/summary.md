# Automatic Retrieval Routing Calibration

## Setup

- cases: 8
- qrels: eval\benchmarks\rag_retrieval_union_v1\qrels_llm.jsonl
- latency budget: 12000 ms
- route counts: {'direct': 3, 'planned': 5}
- optimized planned latency used: True

## Results

| system | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds |
|---|---:|---:|---:|---:|---:|
| direct_hybrid | 0.408 | 0.550 | 1.000 | 0.580 | 2.41 |
| planned_hybrid | 0.461 | 0.600 | 0.854 | 0.620 | 7.50 |
| auto | 0.470 | 0.613 | 0.938 | 0.666 | 3.43 |

Oracle agreement: 8/8 (100.0%).

## Per-case Decisions

| case | selected | oracle | score | nDCG direct/planned | seconds | reasons |
|---|---|---|---:|---:|---:|---|
| rag_compound_overview | planned_hybrid | planned_hybrid | 4 | 0.366 / 0.515 | 10.72 | complexity_threshold_reached, multiple_answer_aspects |
| chunking_not_fixed_window | planned_hybrid | planned_hybrid | 3 | 0.436 / 0.651 | 3.13 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| rag_evaluation_reliability | planned_hybrid | planned_hybrid | 4 | 0.605 / 0.671 | 11.02 | complexity_threshold_reached, multiple_answer_aspects |
| embedding_bge_m3_role | planned_hybrid | planned_hybrid | 3 | 0.673 / 0.828 | 3.73 | complexity_threshold_reached, multi_category_expansion_without_aspects |
| reranking_role | planned_hybrid | planned_hybrid | 3 | 0.408 / 0.515 | 7.48 | complexity_threshold_reached, aspect_detected_with_low_category_confidence |
| query_planning_expansion | direct_hybrid | direct_hybrid | 0 | 0.679 / 0.590 | 2.62 | simple_or_specific_query |
| vector_db_chroma_role | direct_hybrid | direct_hybrid | 0 | 0.894 / 0.652 | 2.28 | simple_or_specific_query |
| enterprise_rag_retrieval | direct_hybrid | direct_hybrid | 0 | 0.577 / 0.539 | 2.53 | simple_or_specific_query |

## Interpretation Boundary

The same eight queries informed the initial routing threshold, so oracle agreement is a calibration result. Generalization requires a larger, independently held-out query set.
