# Long-term Memory Upgrade

## Why this step

The earlier web workbench only had short in-process conversation memory. That
helps with a few follow-up turns, but it disappears after server restart and
cannot carry user preferences or project state across sessions.

This upgrade adds persistent long-term memory while keeping a clean boundary:

- RAG knowledge base = factual external evidence that can be cited.
- Short memory = recent turns for resolving follow-up references.
- Long memory = user preferences, goals, project state and decisions.

Long memory can personalize and answer memory/meta questions, but it should not
be cited as external domain evidence.

## References

Design ideas were adapted from current open-source memory systems:

- LangGraph/LangChain long-term memory: namespace + key/store design.
- LangMem: extract important information from conversations and maintain memory
  over time.
- Mem0: retrieve relevant memories before generation and update memories after
  interactions.
- Letta/MemGPT style agents: stateful agents that preserve memory across
  conversations.

## What changed

- Added `src/long_memory.py`.
  - SQLite stores turns and memory records.
  - A separate Chroma collection stores memory embeddings.
  - DeepSeek extracts durable memories when available.
  - Rule-based extraction is used as fallback.
  - Memory text is sanitized to avoid storing obvious API keys or secrets.
- Updated `webapp/server.py`.
  - Status API returns long-memory stats.
  - Each normal RAG question retrieves relevant long memories.
  - Answers store durable memories after generation.
  - Memory/meta questions such as "我之前有什么要求" use memory-answer mode.
  - Memory-answer mode reads long memory but does not write another duplicate
    long memory.
- Updated `webapp/static/index.html`, `app.js`, and `styles.css`.
  - Added a long-memory switch.
  - Added a clear-long-memory button.
  - Memory tab now shows retrieved long memories, stored memories, counts by
    kind, and any memory errors.

## Storage

Runtime memory files are local-only and ignored by git:

- SQLite: `data/runtime/long_memory.sqlite3`
- Memory vectors: `data/runtime/long_memory_chroma`

The main RAG document index remains separate:

- Document index: `data/indexes/llm_rag_chroma`
- Collection: `llm_rag_docs`

## Validation

Test sequence:

1. Clear long memory.
2. Ask: `我希望后面学习RAG项目时，你每一步都告诉我为什么这么做，以及这样做的好处。`
3. The system stores:
   - one `preference`
   - one `goal`
4. Start a different session.
5. Ask: `我之前对学习方式有什么要求？`

Result:

- memory_answer_mode: true
- retrieved long memories: 2
- stored new long memories: 0
- quality_pass: true
- answer correctly recalls the learning preference.

## Learning takeaway

Long memory is not just "more chat history". A useful design separates:

- what to remember
- where to store it
- how to retrieve it
- when it is allowed to influence the answer
- when it should not be treated as factual source evidence

This makes the project closer to a real assistant/RAG system instead of a simple
conversation buffer.
