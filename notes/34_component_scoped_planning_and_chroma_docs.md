# 34. Component-Scoped Planning and Chroma Query Docs

Date: 2026-08-20

## Goal

Make the RAG demo closer to a mature system by fixing badcases from full evaluation instead of only tuning prompts.

## Why This Step Was Needed

The full eval initially stayed at 8/10. The failing cases were not hallucination problems: faithfulness, citation, and relevance often passed. The real issue was coverage:

- The planner used one broad `techniques` aspect for different component questions.
- A reranking question, a Chroma metadata question, and an enterprise hybrid retrieval question were sometimes forced into the same answer requirements.
- The Chroma knowledge base lacked direct evidence for `collection.query(..., where=...)`, so the model sometimes answered conservatively that details were missing.

## Changes

1. Added component-scoped query planning in `src/query_planning.py`.

- `reranking_role`: focuses on reranking as a second-stage sorter after vector/semantic retrieval.
- `vector_db_chroma_role`: focuses on Chroma as vector database, collection/query, document/embedding/metadata storage, and metadata filtering.
- `enterprise_rag_retrieval`: focuses on hybrid retrieval, metadata filtering, reranking, and the risk of relying only on one vector similarity search.

2. Added official Chroma sources to `data/source_manifests/llm_rag_sources.csv`.

- Chroma Query and Get: https://docs.trychroma.com/docs/querying-collections/query-and-get
- Chroma Metadata Filtering: https://docs.trychroma.com/docs/querying-collections/metadata-filtering

3. Rebuilt the knowledge base through the formal pipeline.

```powershell
python experiments\16_llm_rag_sources\fetch_sources.py --priority P0 --sleep 0.1
python experiments\17_llm_rag_chunking\build_chunks.py
python experiments\18_llm_rag_index\build_index.py --rebuild --batch-size 8
```

## Data Result

- Documents: 52
- Chunks: 938
- Too long chunks: 0
- Tiny chunks: 0
- Chroma indexed count: 938

The new Chroma chunks directly contain:

- `collection.query(...)`
- `where` for metadata filtering
- `where_document`
- `$and`, `$or`, `$in`, `$contains`

## Evaluation Result

Targeted badcase eval:

```text
vector_db_chroma_role: PASS
enterprise_rag_retrieval: PASS
```

Full smoke eval:

```text
total: 10
passed: 10
failed: 0
pass_rate: 100%
quality_pass_rate: 100%
indexed_count: 938
```

Output:

- `eval/rag_system_full_after_chroma_query_docs/summary.md`
- `eval/rag_system_full_after_chroma_query_docs/results.jsonl`

## Learning Takeaway

This was a useful enterprise-RAG lesson: when an answer fails, first separate whether the problem is data, retrieval, planning, prompt, generation, or evaluation. In this case, lowering audit thresholds or writing a stronger prompt would have hidden the real issue. The better fix was to add missing authoritative source documents and make query planning more component-aware.
