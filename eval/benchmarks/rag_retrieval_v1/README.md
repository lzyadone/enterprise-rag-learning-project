# RAG Retrieval Benchmark V1

This benchmark stores human query-to-chunk relevance judgments independently
from retrieval scores and reranker outputs.

## Relevance scale

- `3`: direct evidence that answers the question
- `2`: important supporting evidence
- `1`: topically related but insufficient evidence
- `0`: irrelevant or misleading for the question

The local annotation tool reads the reproducible candidate pool from
`eval/planned_reranker_full/candidate_pools.jsonl` and writes reviewed labels to
`qrels.jsonl`. Candidate text is not duplicated in Git; each judgment refers to
the stable `query_id` and `chunk_id` pair.

Do not use category names, lexical overlap, retrieval rank, or model scores as
the final label. Read the candidate text and judge whether it provides evidence
for the question.
