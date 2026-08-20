# Multi-aspect RAG Quality Upgrade

## Why this step

The first web demo could retrieve and answer, but broad questions such as
`RAG系统分为哪些类别，有哪些关键技术，瓶颈有哪些` exposed a real RAG problem:
faithfulness can pass while the answer is still too narrow. The root cause was
not only prompt quality. The retrieval plan selected too few categories and did
not preserve evidence coverage across different parts of a compound question.

## What changed

- Added multi-aspect query planning in `src/query_planning.py`.
  - The planner can split compound RAG questions into answer aspects such as
    classification, techniques, bottlenecks, workflow and evaluation.
  - Each aspect now has a clean `question` for answer requirements and a
    separate `search_query` for retrieval expansion.
- Added plan-aware retrieval selection in `src/retrieval.py`.
  - Retrieval runs are executed per aspect.
  - Results are fused and reranked.
  - Final selection reserves evidence for aspect coverage and category coverage.
  - Classification keeps multiple source families when useful.
  - Bottleneck questions receive heading/text boosts for evidence terms such as
    `Failure Points`, `Missing Content`, `Not in Context`, and `Incomplete`.
- Added coverage audit in `src/coverage_audit.py`.
  - This checks whether a compound answer covered the required aspects, not only
    whether individual statements are grounded.
- Connected answer repair in `webapp/server.py`.
  - The API now runs generate -> audit -> optional repair -> re-audit.
  - Repair uses the same retrieved evidence and audit feedback.
- Improved the web workbench UI.
  - The plan tab shows aspects and retrieval-oriented search queries.
  - Source rows show which aspect each source supports.
  - The summary panel distinguishes faithfulness and coverage quality.
- Expanded the curated source manifest.
  - Added RAG Survey, Seven Failure Points, Lost in the Middle, and LlamaIndex
    Advanced Retrieval.

## Data update

The knowledge base was rebuilt through the normal pipeline:

1. Fetch and normalize source documents.
2. Structure-aware chunking.
3. Rebuild Chroma index with `bge-m3`.

Current index:

- documents: 36
- chunks: 647
- collection: `llm_rag_docs`

## Validation query

Query:

`RAG系统分为哪些类别，有哪些关键技术，瓶颈有哪些`

Final test result:

- sources: 10
- source categories:
  - RAG overview
  - evaluation
  - RAG challenges
  - chunking
  - embedding
  - retrieval
  - reranking
  - querying
  - vector db
- quality_pass: true
- overall_pass: true
- coverage_pass: true
- LLM audit scores:
  - faithfulness: 5
  - citation: 5
  - relevance: 5

## Learning takeaway

For a serious RAG system, good answers come from the whole chain:

- source quality
- structure-aware chunking
- query planning
- multi-route retrieval
- reranking
- coverage-aware context selection
- grounded generation
- answer audit and repair

Prompt engineering matters, but it cannot compensate for missing sources or
poor retrieval coverage.
