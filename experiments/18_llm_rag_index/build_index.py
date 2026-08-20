"""Build a local Chroma index for the LLM/RAG knowledge base."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "llm_rag_docs" / "chunks.jsonl"
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_COLLECTION = "llm_rag_docs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma index from LLM/RAG chunks.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def embed_texts(texts: list[str], model: str, host: str) -> list[list[float]]:
    clean_texts = [normalize_for_embedding(text) for text in texts]
    try:
        response = post_json(f"{host.rstrip('/')}/api/embed", {"model": model, "input": clean_texts})
        embeddings = response.get("embeddings")
        if isinstance(embeddings, list) and len(embeddings) == len(texts):
            return embeddings
    except Exception:
        pass

    embeddings = []
    for text in clean_texts:
        response = post_json(f"{host.rstrip('/')}/api/embeddings", {"model": model, "prompt": text})
        embedding = response.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Ollama embedding response missing embedding")
        embeddings.append(embedding)
    return embeddings


def normalize_for_embedding(text: str) -> str:
    return " ".join(text.replace("\r", "\n").split())


def chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    keys = [
        "doc_id",
        "source_id",
        "title",
        "category",
        "priority",
        "source_type",
        "url",
        "heading_path",
        "heading_level",
        "section_index",
        "chunk_index",
        "chunk_in_section",
        "char_count",
        "token_estimate",
        "text_hash",
    ]
    metadata: dict[str, str | int | float | bool] = {}
    for key in keys:
        value = chunk.get(key)
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif value is not None:
            metadata[key] = str(value)
    return metadata


def batched(rows: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]


def main() -> None:
    args = parse_args()
    chunks = read_jsonl(args.chunks)
    if args.limit:
        chunks = chunks[: args.limit]
    if not chunks:
        raise ValueError(f"No chunks found: {args.chunks}")

    args.db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.db_dir))
    if args.rebuild:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=args.collection)

    existing_count = collection.count()
    if existing_count and not args.rebuild:
        print(f"collection already has {existing_count} rows. Use --rebuild to rebuild.")
        return

    start = time.time()
    batches = batched(chunks, args.batch_size)
    print(f"chunks: {len(chunks)}")
    print(f"batches: {len(batches)}")
    print(f"db_dir: {args.db_dir}")
    print(f"collection: {args.collection}")
    print(f"embedding_model: {args.model}")

    for batch_index, batch in enumerate(batches, start=1):
        texts = [str(row["text"]) for row in batch]
        ids = [str(row["chunk_id"]) for row in batch]
        metadatas = [chroma_metadata(row) for row in batch]
        embeddings = embed_texts(texts, model=args.model, host=args.ollama_host)
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
        print(f"[{batch_index}/{len(batches)}] indexed {len(batch)} chunks", flush=True)

    elapsed = round(time.time() - start, 2)
    print(f"indexed_count: {collection.count()}")
    print(f"elapsed_seconds: {elapsed}")


if __name__ == "__main__":
    main()
