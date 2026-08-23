# Automatic Retrieval Routing Holdout V2 Protocol

This benchmark is frozen before candidate generation, query planning, relevance labeling, or metric inspection.

## Purpose

- Evaluate `direct_hybrid`, anchored `planned_v2_hybrid`, and automatic routing on unseen questions.
- Test both narrow knowledge-point questions and multi-stage enterprise RAG questions.
- Make a release decision without tuning retrieval weights or routing rules on this set.

## Frozen Inputs

- Dataset: `eval/datasets/rag_routing_holdout_v2.jsonl`
- 20 questions: 10 focused and 10 compound.
- Corpus: the existing 938-chunk `llm_rag_docs` collection.
- Direct system: `direct_hybrid`.
- Planned system: `planned_v2_hybrid` using anchored fusion.
- Pool depth: 10 per system.
- Relevance threshold: grade 2 or higher.
- Labels: system-blind DeepSeek judgments over the complete union pool.

Question topics were selected only from corpus titles, categories, and headings that had not been used as prior calibration targets. No retrieval result or query plan was inspected while writing them.

## Pre-registered Gates

The automatic route is a release candidate only if every gate passes:

1. Average nDCG@10 is no more than 0.02 below direct retrieval.
2. Average Recall@10 is no more than 0.02 below direct retrieval.
3. MRR@10 is no more than 0.05 below direct retrieval.
4. Per-question oracle agreement is at least 75%.
5. No question routed to planned retrieval loses more than 0.25 nDCG@10 versus direct retrieval.
6. Median measured retrieval-path latency stays within the 12-second routing budget.

Passing these gates does not automatically change the product default. It makes auto routing eligible for a human spot audit and a canary trial. Failure keeps direct retrieval as the default. The retrieval algorithm and routing rules must not be changed using this holdout's results.

## Interpretation Limits

- Pooling judges the union of top-10 candidates, not every corpus chunk.
- LLM labels provide a repeatable blind signal but are not a substitute for domain-expert review.
- Candidate-run latency is useful for relative comparison; a later live API benchmark must measure full request latency.
