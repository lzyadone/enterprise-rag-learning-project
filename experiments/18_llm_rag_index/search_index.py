"""Search the local Chroma index for the LLM/RAG knowledge base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_COLLECTION = "llm_rag_docs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the LLM/RAG Chroma index.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--category", default=None)
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def embed_query(query: str, model: str, host: str) -> list[float]:
    query = " ".join(query.replace("\r", "\n").split())
    response = post_json(f"{host.rstrip('/')}/api/embed", {"model": model, "input": [query]})
    embeddings = response.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        return embeddings[0]
    response = post_json(f"{host.rstrip('/')}/api/embeddings", {"model": model, "prompt": query})
    embedding = response.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("Ollama embedding response missing embedding")
    return embedding


def main() -> None:
    args = parse_args()
    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(name=args.collection)
    query_embedding = embed_query(args.query, model=args.model, host=args.ollama_host)
    where = {"category": args.category} if args.category else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    print(f"query: {args.query}")
    print(f"collection_count: {collection.count()}")
    print(f"category_filter: {args.category or 'none'}")
    print()
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        preview = " ".join(str(document).split())[:500]
        print(f"{rank}. distance={distance:.4f} chunk_id={chunk_id}")
        print(f"   title={metadata.get('title')}")
        print(f"   category={metadata.get('category')}")
        print(f"   section={metadata.get('heading_path')}")
        print(f"   url={metadata.get('url')}")
        print(f"   preview={preview}")
        print()


if __name__ == "__main__":
    main()
