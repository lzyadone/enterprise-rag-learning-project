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

`qrels.jsonl` inherits 128 human judgments from `rag_retrieval_v1` and remains
partial. `qrels_llm.jsonl` is a separate complete 224-pair benchmark produced by
`deepseek-v4-flash`; it does not overwrite or claim to extend the human qrels.
All model labels were regenerated from the shuffled union pool in batches of 10.

On the 128-pair human overlap, the model achieves 49.22% exact agreement,
96.09% within-one agreement, and 3.91% severe disagreement. Use the complete
LLM qrels for comparative experiments, not as a human gold-standard claim.
