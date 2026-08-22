# Candidate Generator Evaluation (LLM-Judged Union Pool)

## Setup

- cases: 8
- pooled query/chunk pairs: 224
- relevant threshold: >= 2
- qrels: eval\benchmarks\rag_retrieval_union_v1\qrels_llm.jsonl
- qrels type: complete_blind_llm_judgments

## Results

| system | Recall@5 | Recall@10 | Precision@10 | MRR@10 | nDCG@10 | median seconds | max seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct_dense | 0.213 | 0.434 | 0.575 | 0.698 | 0.519 | 1.03 | 24.50 |
| direct_bm25 | 0.239 | 0.409 | 0.525 | 0.938 | 0.585 | 0.06 | 0.38 |
| direct_hybrid | 0.257 | 0.408 | 0.550 | 1.000 | 0.580 | 2.41 | 2.62 |
| planned_dense | 0.219 | 0.463 | 0.600 | 0.688 | 0.547 | 22.18 | 45.87 |
| planned_hybrid | 0.251 | 0.461 | 0.600 | 0.854 | 0.620 | 21.86 | 45.76 |

## Per-case Recall@10 / nDCG@10

| case | direct_dense | direct_bm25 | direct_hybrid | planned_dense | planned_hybrid |
|---|---:|---:|---:|---:|---:|
| rag_compound_overview | 0.500 / 0.453 | 0.250 / 0.262 | 0.250 / 0.366 | 0.500 / 0.601 | 0.333 / 0.515 |
| chunking_not_fixed_window | 0.143 / 0.317 | 0.714 / 0.618 | 0.286 / 0.436 | 0.429 / 0.368 | 0.571 / 0.651 |
| rag_evaluation_reliability | 0.231 / 0.321 | 0.385 / 0.744 | 0.308 / 0.605 | 0.308 / 0.540 | 0.308 / 0.671 |
| embedding_bge_m3_role | 0.562 / 0.786 | 0.188 / 0.384 | 0.438 / 0.673 | 0.562 / 0.776 | 0.562 / 0.828 |
| reranking_role | 0.429 / 0.349 | 0.571 / 0.575 | 0.429 / 0.408 | 0.429 / 0.450 | 0.429 / 0.515 |
| query_planning_expansion | 0.667 / 0.567 | 0.444 / 0.669 | 0.667 / 0.679 | 0.778 / 0.669 | 0.667 / 0.590 |
| vector_db_chroma_role | 0.714 / 0.887 | 0.357 / 0.787 | 0.571 / 0.894 | 0.429 / 0.523 | 0.500 / 0.652 |
| enterprise_rag_retrieval | 0.227 / 0.473 | 0.364 / 0.638 | 0.318 / 0.577 | 0.273 / 0.445 | 0.318 / 0.539 |

## Interpretation Boundary

Metrics are computed against a depth-10 union pool, not exhaustive judgments over all collection chunks. LLM labels are consistent across all 224 pairs but are not a substitute for independent human judgments.
Use the result to choose which candidate generators deserve human validation and online testing.
