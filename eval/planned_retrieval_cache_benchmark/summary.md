# Planned Retrieval Embedding Cache Benchmark

The baseline and optimized runs use the same current retrieval code and deterministic plan.
The only changed variable is repeated embedding calls versus one embedding per unique query.

| system | baseline median | optimized median | median speedup | same Top-10 order | same Top-10 set | avg embedding calls before/after |
|---|---:|---:|---:|---:|---:|---:|
| planned_dense | 23.13s | 6.03s | 3.95x | 8/8 | 8/8 | 22.50 / 5.50 |
| planned_hybrid | 22.78s | 7.50s | 3.13x | 8/8 | 8/8 | 22.50 / 5.50 |

## Per-case latency

| case | system | before | after | speedup | same order | same set | embed calls before/after |
|---|---|---:|---:|---:|---:|---:|---:|
| rag_compound_overview | planned_dense | 54.24s | 11.73s | 4.62x | Y | Y | 48 / 9 |
| rag_compound_overview | planned_hybrid | 58.83s | 10.72s | 5.49x | Y | Y | 48 / 9 |
| chunking_not_fixed_window | planned_dense | 8.59s | 2.46s | 3.50x | Y | Y | 9 / 3 |
| chunking_not_fixed_window | planned_hybrid | 8.92s | 3.13s | 2.85x | Y | Y | 9 / 3 |
| rag_evaluation_reliability | planned_dense | 44.90s | 11.12s | 4.04x | Y | Y | 35 / 9 |
| rag_evaluation_reliability | planned_hybrid | 43.34s | 11.02s | 3.93x | Y | Y | 35 / 9 |
| embedding_bge_m3_role | planned_dense | 10.31s | 3.56s | 2.90x | Y | Y | 9 / 3 |
| embedding_bge_m3_role | planned_hybrid | 8.77s | 3.73s | 2.35x | Y | Y | 9 / 3 |
| reranking_role | planned_dense | 24.43s | 5.47s | 4.47x | Y | Y | 20 / 5 |
| reranking_role | planned_hybrid | 26.30s | 7.48s | 3.52x | Y | Y | 20 / 5 |
| query_planning_expansion | planned_dense | 21.83s | 5.64s | 3.87x | Y | Y | 19 / 5 |
| query_planning_expansion | planned_hybrid | 21.26s | 7.03s | 3.02x | Y | Y | 19 / 5 |
| vector_db_chroma_role | planned_dense | 19.46s | 7.17s | 2.72x | Y | Y | 19 / 5 |
| vector_db_chroma_role | planned_hybrid | 19.60s | 7.54s | 2.60x | Y | Y | 19 / 5 |
| enterprise_rag_retrieval | planned_dense | 27.17s | 6.42s | 4.23x | Y | Y | 21 / 5 |
| enterprise_rag_retrieval | planned_hybrid | 24.31s | 7.52s | 3.23x | Y | Y | 21 / 5 |

Candidate equality is checked between paired runs in the same process.
The embedding model is warmed once; the in-process query cache is cleared before each run.
