# Natural Query Holdout V3 Protocol

This benchmark is frozen before candidate retrieval, query-plan inspection, qrels
labeling, or metric inspection.

## Purpose

- Evaluate whether conservative Planner v3 generalizes beyond the development
  set used to design it.
- Compare `direct_hybrid` with `planned_v3_hybrid` on unseen natural questions.
- Compute the conservative automatic route from the frozen v3 plan shape after
  both candidate systems have been evaluated.
- Keep `direct_hybrid` as the product default unless the pre-registered gates
  pass and a later human spot audit finds no blocking regressions.

## Frozen Inputs

- Dataset: `eval/datasets/rag_natural_query_holdout_v3.jsonl`
- Source anchors: `eval/benchmarks/rag_natural_query_holdout_v3/source_anchors.jsonl`
- Generation summary: `eval/benchmarks/rag_natural_query_holdout_v3/generation_summary.json`
- Questions: 32 total, with 16 focused and 16 compound.
- Source anchors: 48 unique chunks from the existing 938-chunk `llm_rag_docs`
  collection.
- Generator and critic: `deepseek-v4-flash`
- Embedding model used for semantic de-duplication: `bge-m3`
- Fixed random seed: `20260824`
- Dataset SHA-256:
  `85d8ff3c55183ba83ed6d116438d9fcbac36e0e6e31af3ef4beb1f5e16b18b01`
- Source-anchor SHA-256:
  `daf3cd40a626d4609de8b049feef8ef08cd0c0aa918a5f459e55bac81aeb9805`
- Base code commit before this holdout work:
  `620f4a3 Harden local generation fallback`

Source anchors prove intended answerability for dataset construction. They are
not relevance judgments and must not be supplied to candidate generators, query
planners, qrels judges, or metric calculators.

## Candidate Generation

- Fixed candidate systems: `direct_hybrid` and `planned_v3_hybrid`.
- Use the same corpus, Chroma index, BM25 index, embedding model, `top_k`, pool
  depth, and rerank settings for both systems.
- Pool depth: top 10 candidates per system per question.
- Build a depth-10 union candidate pool per question.
- Deduplicate candidates by stable `query_id` and `chunk_id`.
- Deterministically shuffle candidate order before labeling.
- Keep system name, rank, score, retrieval channel, expanded query, and query
  plan out of the blind labeling input.
- Preserve provenance only in a separate manifest that is inspected after qrels
  are frozen.

## Relevance Labels

Use a 0-3 relevance scale:

- `3`: direct evidence that answers the question.
- `2`: important supporting evidence, but not complete by itself.
- `1`: topically related but insufficient.
- `0`: irrelevant or misleading for the question.

Relevance threshold for retrieval metrics is grade 2 or higher. LLM qrels are a
repeatable blind signal, not a human gold standard. A later human spot audit must
sample model labels before any release decision.

## Metrics

Report these metrics for `direct_hybrid`, `planned_v3_hybrid`, and the derived
conservative automatic route:

- Recall@5
- Recall@10
- MRR@10
- nDCG@10
- Median and max retrieval-path latency
- Per-question planned-vs-direct nDCG delta
- Automatic route agreement with the per-question oracle

Measure `planned_v3_hybrid` latency in a fresh process or isolated run so prior
direct retrieval embedding caches do not make v3 look artificially faster.

## Pre-registered Gates

Planner v3 automatic routing is a release candidate only if every gate passes:

1. At least 90% of questions have at least one grade-2-or-higher candidate in
   the union pool.
2. Automatic-route nDCG@10 is no more than 0.02 below `direct_hybrid`.
3. Automatic-route Recall@10 is no more than 0.02 below `direct_hybrid`.
4. Automatic-route MRR@10 is no more than 0.05 below `direct_hybrid`.
5. Automatic route agreement with the per-question oracle is at least 75%.
6. No question routed to planned retrieval loses more than 0.25 nDCG@10 versus
   `direct_hybrid`.
7. Median end-to-end retrieval latency is no more than 12 seconds.

Passing these gates does not automatically change the Web default. It only makes
Planner v3 eligible for a human spot audit and later Web experimental-mode
exposure. Failure keeps `direct_hybrid` as the default and the holdout results
must not be used to tune Planner v3.

## Interpretation Limits

- This benchmark evaluates retrieval candidate generation, not final answer
  generation.
- Pooling evaluates the union of top-10 candidates, not every chunk in the
  corpus.
- Source anchors are construction evidence, not qrels.
- The retrieval algorithm, routing rules, qrels policy, and pass/fail gates must
  not be changed after candidate results are inspected.
