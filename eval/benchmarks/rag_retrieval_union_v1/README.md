# RAG Retrieval Union Benchmark V1

This benchmark evaluates candidate generation across five retrieval systems:

- `direct_dense`
- `direct_bm25`
- `direct_hybrid`
- `planned_dense`
- `planned_hybrid`

Each system contributes its top 10 candidates. Candidates are deduplicated per
query and then deterministically shuffled before annotation. Retrieval scores,
ranks, channels, expanded queries, and plan aspects are removed from the
annotation view. The separate `pool_manifest.jsonl` preserves provenance for
evaluation after labeling is complete.

The relevance scale remains:

- `3`: direct evidence
- `2`: important supporting evidence
- `1`: topically related but insufficient
- `0`: irrelevant or misleading

The initial qrels inherit all 128 judgments from `rag_retrieval_v1`. Only newly
pooled query/chunk pairs require additional human review.
