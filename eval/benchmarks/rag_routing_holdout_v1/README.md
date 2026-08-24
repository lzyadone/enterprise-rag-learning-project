# Retrieval Routing Holdout V1

This benchmark tests whether the direct/planned retrieval router generalizes beyond the eight
queries used during initial rule calibration.

## Protocol

1. Freeze the 16 questions in `eval/datasets/rag_routing_holdout_v1.jsonl` before retrieval.
2. Run `direct_hybrid` and `planned_hybrid` to depth 10 and pool their unique candidates.
3. Deterministically shuffle candidates and hide system names, ranks, scores, and query plans.
4. Ask the same DeepSeek relevance judge to assign a 0-3 grade to every pooled candidate.
5. Compare direct, planned, and automatic routing with Recall@10, MRR@10, nDCG@10, latency,
   and agreement with the per-query nDCG oracle.

The focused/compound strata describe question construction only. They are not expected route
labels and are not passed to the router or relevance judge. LLM labels make this a reproducible
holdout signal, not a replacement for a later human audit.
