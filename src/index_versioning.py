"""Versioned, incremental Chroma index lifecycle helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import chromadb

from src.chunking import (
    chunks_to_records,
    clean_metadata_lines,
    normalize_text,
    split_markdown_document,
    strip_frontmatter,
)
from src.ollama_http import embed_texts as ollama_embed_texts


INDEX_SCHEMA_VERSION = 1
DOCUMENT_METADATA_FIELDS = (
    "doc_id",
    "source_id",
    "title",
    "category",
    "priority",
    "source_type",
    "url",
)
CHROMA_METADATA_FIELDS = (
    *DOCUMENT_METADATA_FIELDS,
    "heading_path",
    "heading_level",
    "section_index",
    "chunk_index",
    "chunk_in_section",
    "char_count",
    "token_estimate",
    "text_hash",
)
Embedder = Callable[[list[str], str, str], list[list[float]]]


@dataclass(frozen=True)
class ChunkingConfig:
    soft_max_chars: int = 1800
    hard_max_chars: int = 3500
    min_chars: int = 280

    def as_dict(self) -> dict[str, int]:
        return {
            "soft_max_chars": self.soft_max_chars,
            "hard_max_chars": self.hard_max_chars,
            "min_chars": self.min_chars,
        }


@dataclass(frozen=True)
class DeltaPlan:
    added: tuple[str, ...]
    changed: tuple[str, ...]
    deleted: tuple[str, ...]
    unchanged: tuple[str, ...]

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.changed or self.deleted)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "added": list(self.added),
            "changed": list(self.changed),
            "deleted": list(self.deleted),
            "unchanged": list(self.unchanged),
        }


@dataclass(frozen=True)
class IncrementalChunkResult:
    chunks: tuple[dict[str, Any], ...]
    plan: DeltaPlan
    reused_documents: int
    split_documents: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            rows.append(row)
    return rows


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_documents(documents: list[dict[str, Any]]) -> None:
    if not documents:
        raise ValueError("At least one document is required")
    source_ids: set[str] = set()
    for document in documents:
        missing = [field for field in (*DOCUMENT_METADATA_FIELDS, "text") if field not in document]
        if missing:
            raise ValueError(f"Document is missing {missing}: {document.get('source_id')}")
        source_id = str(document["source_id"])
        if source_id in source_ids:
            raise ValueError(f"Duplicate source_id: {source_id}")
        source_ids.add(source_id)


def document_fingerprint(document: dict[str, Any], config: ChunkingConfig) -> str:
    body = clean_metadata_lines(
        normalize_text(strip_frontmatter(str(document["text"])))
    )
    metadata = {field: str(document[field]) for field in DOCUMENT_METADATA_FIELDS}
    return hash_payload(
        {
            "metadata": metadata,
            "body": body,
            "chunking": config.as_dict(),
        }
    )


def document_fingerprints(
    documents: list[dict[str, Any]],
    config: ChunkingConfig,
) -> dict[str, str]:
    validate_documents(documents)
    return {
        str(document["source_id"]): document_fingerprint(document, config)
        for document in documents
    }


def plan_delta(
    previous_manifest: dict[str, Any] | None,
    documents: list[dict[str, Any]],
    config: ChunkingConfig,
) -> DeltaPlan:
    new_hashes = document_fingerprints(documents, config)
    previous_sources = (previous_manifest or {}).get("sources", {})
    if not isinstance(previous_sources, dict):
        raise ValueError("Manifest sources must be an object")
    previous_ids = set(previous_sources)
    new_ids = set(new_hashes)
    config_changed = bool(
        previous_manifest
        and previous_manifest.get("chunking") != config.as_dict()
    )
    common = previous_ids & new_ids
    changed = {
        source_id
        for source_id in common
        if config_changed
        or str(previous_sources[source_id].get("document_hash", ""))
        != new_hashes[source_id]
    }
    return DeltaPlan(
        added=tuple(sorted(new_ids - previous_ids)),
        changed=tuple(sorted(changed)),
        deleted=tuple(sorted(previous_ids - new_ids)),
        unchanged=tuple(sorted(common - changed)),
    )


def chunk_document(
    document: dict[str, Any],
    config: ChunkingConfig,
) -> list[dict[str, Any]]:
    chunks = split_markdown_document(
        str(document["text"]),
        title=str(document["title"]),
        soft_max_chars=config.soft_max_chars,
        hard_max_chars=config.hard_max_chars,
        min_chars=config.min_chars,
    )
    metadata = {field: str(document[field]) for field in DOCUMENT_METADATA_FIELDS}
    return chunks_to_records(chunks, metadata)


def build_incremental_chunks(
    documents: list[dict[str, Any]],
    config: ChunkingConfig,
    previous_manifest: dict[str, Any],
    previous_chunks: list[dict[str, Any]],
) -> IncrementalChunkResult:
    plan = plan_delta(previous_manifest, documents, config)
    previous_by_source = group_chunks_by_source(previous_chunks)
    previous_sources = previous_manifest.get("sources", {})
    output: list[dict[str, Any]] = []
    reused_documents = 0
    split_documents = 0
    unchanged = set(plan.unchanged)

    for document in documents:
        source_id = str(document["source_id"])
        if source_id in unchanged:
            rows = previous_by_source.get(source_id, [])
            expected_ids = list(previous_sources[source_id].get("chunk_ids", []))
            actual_ids = [str(row.get("chunk_id", "")) for row in rows]
            if not rows or actual_ids != expected_ids:
                raise ValueError(f"Previous chunks do not match manifest for {source_id}")
            output.extend(rows)
            reused_documents += 1
        else:
            output.extend(chunk_document(document, config))
            split_documents += 1

    validate_chunks(output)
    return IncrementalChunkResult(
        chunks=tuple(output),
        plan=plan,
        reused_documents=reused_documents,
        split_documents=split_documents,
    )


def group_chunks_by_source(
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        grouped.setdefault(str(chunk.get("source_id", "")), []).append(chunk)
    return grouped


def validate_chunks(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        raise ValueError("At least one chunk is required")
    chunk_ids: set[str] = set()
    for chunk in chunks:
        missing = [
            field
            for field in ("chunk_id", "source_id", "text", "text_hash")
            if not chunk.get(field)
        ]
        if missing:
            raise ValueError(f"Chunk is missing {missing}: {chunk.get('chunk_id')}")
        chunk_id = str(chunk["chunk_id"])
        if chunk_id in chunk_ids:
            raise ValueError(f"Duplicate chunk_id: {chunk_id}")
        chunk_ids.add(chunk_id)


def source_states(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    config: ChunkingConfig,
) -> dict[str, dict[str, Any]]:
    fingerprints = document_fingerprints(documents, config)
    grouped = group_chunks_by_source(chunks)
    if set(fingerprints) != set(grouped):
        missing = sorted(set(fingerprints) - set(grouped))
        extra = sorted(set(grouped) - set(fingerprints))
        raise ValueError(f"Document/chunk source mismatch; missing={missing}, extra={extra}")
    return {
        source_id: {
            "document_hash": fingerprints[source_id],
            "chunk_count": len(grouped[source_id]),
            "chunk_ids": [str(row["chunk_id"]) for row in grouped[source_id]],
            "chunk_hashes": [str(row["text_hash"]) for row in grouped[source_id]],
        }
        for source_id in sorted(fingerprints)
    }


def store_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_stored_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_version_id(version_id: str) -> None:
    if not version_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in version_id):
        raise ValueError("version_id may contain only letters, numbers, dot, underscore and hyphen")


def chroma_metadata(chunk: dict[str, Any], version_id: str) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {"index_version": version_id}
    for key in CHROMA_METADATA_FIELDS:
        value = chunk.get(key)
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif value is not None:
            metadata[key] = str(value)
    return metadata


def collection_records(collection: chromadb.Collection) -> dict[str, dict[str, Any]]:
    payload = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = payload.get("ids") or []
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []
    embeddings = payload.get("embeddings")
    if embeddings is None:
        embeddings = []
    if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
        raise ValueError("Chroma collection returned inconsistent record lengths")
    return {
        str(chunk_id): {
            "document": str(document),
            "metadata": metadata or {},
            "embedding": [float(value) for value in embedding],
        }
        for chunk_id, document, metadata, embedding in zip(
            ids, documents, metadatas, embeddings, strict=True
        )
    }


def materialize_index_version(
    *,
    root: Path,
    versions_dir: Path,
    version_id: str,
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    config: ChunkingConfig,
    embedding_model: str,
    collection_name: str,
    ollama_host: str,
    batch_size: int,
    previous_db_dir: Path | None,
    previous_manifest: dict[str, Any] | None,
    previous_manifest_path: Path | None,
    delta: DeltaPlan,
    reused_documents: int,
    split_documents: int,
    embedder: Embedder = ollama_embed_texts,
) -> Path:
    validate_version_id(version_id)
    validate_documents(documents)
    validate_chunks(chunks)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    version_dir = (versions_dir / version_id).resolve()
    versions_root = versions_dir.resolve()
    if version_dir.parent != versions_root:
        raise ValueError("Version directory escaped versions root")
    if version_dir.exists():
        raise FileExistsError(f"Index version already exists: {version_dir}")
    version_dir.mkdir(parents=True)
    db_dir = version_dir / "chroma"
    chunks_path = version_dir / "chunks.jsonl"
    manifest_path = version_dir / "manifest.json"
    state_path = version_dir / "build_state.json"
    write_json_atomic(
        state_path,
        {"status": "building", "version_id": version_id, "started_at": utc_now()},
    )
    write_jsonl(chunks_path, chunks)

    previous_records: dict[str, dict[str, Any]] = {}
    if previous_db_dir is not None:
        previous_client = chromadb.PersistentClient(path=str(previous_db_dir))
        previous_collection = previous_client.get_collection(collection_name)
        previous_records = collection_records(previous_collection)

    can_reuse = bool(
        previous_db_dir is not None
        and (
            previous_manifest is None
            or previous_manifest.get("embedding_model") == embedding_model
        )
    )
    by_source_hash: dict[tuple[str, str], list[float]] = {}
    if can_reuse:
        for record in previous_records.values():
            metadata = record["metadata"]
            key = (str(metadata.get("source_id", "")), str(metadata.get("text_hash", "")))
            if all(key):
                by_source_hash.setdefault(key, record["embedding"])

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"embedding_model": embedding_model, "index_version": version_id},
    )
    reused_embeddings = 0
    new_embeddings = 0
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]
        vectors: list[list[float] | None] = []
        missing_indexes: list[int] = []
        for index, chunk in enumerate(batch):
            vector = None
            if can_reuse:
                same_id = previous_records.get(str(chunk["chunk_id"]))
                if same_id and str(same_id["metadata"].get("text_hash", "")) == str(chunk["text_hash"]):
                    vector = same_id["embedding"]
                if vector is None:
                    vector = by_source_hash.get(
                        (str(chunk["source_id"]), str(chunk["text_hash"]))
                    )
            vectors.append(vector)
            if vector is None:
                missing_indexes.append(index)
            else:
                reused_embeddings += 1

        if missing_indexes:
            generated = embedder(
                [str(batch[index]["text"]) for index in missing_indexes],
                embedding_model,
                ollama_host,
            )
            if len(generated) != len(missing_indexes):
                raise ValueError("Embedding provider returned the wrong number of vectors")
            for index, vector in zip(missing_indexes, generated, strict=True):
                vectors[index] = vector
            new_embeddings += len(generated)

        collection.add(
            ids=[str(row["chunk_id"]) for row in batch],
            documents=[str(row["text"]) for row in batch],
            embeddings=[vector for vector in vectors if vector is not None],
            metadatas=[chroma_metadata(row, version_id) for row in batch],
        )

    expected_ids = {str(row["chunk_id"]) for row in chunks}
    actual_ids = set(collection.get(include=[]).get("ids") or [])
    if collection.count() != len(chunks) or actual_ids != expected_ids:
        raise ValueError("Candidate Chroma collection does not match desired chunks")

    states = source_states(documents, chunks, config)
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "status": "ready",
        "version_id": version_id,
        "parent_version": (previous_manifest or {}).get("version_id"),
        "parent_manifest": (
            store_path(previous_manifest_path, root) if previous_manifest_path else None
        ),
        "created_at": utc_now(),
        "embedding_model": embedding_model,
        "collection": collection_name,
        "db_dir": store_path(db_dir, root),
        "chunks_path": store_path(chunks_path, root),
        "document_count": len(states),
        "chunk_count": len(chunks),
        "documents_hash": hash_payload(
            {source_id: state["document_hash"] for source_id, state in states.items()}
        ),
        "chunks_hash": hash_payload(
            [(row["chunk_id"], row["text_hash"]) for row in chunks]
        ),
        "chunking": config.as_dict(),
        "sources": states,
        "delta": delta.as_dict(),
        "build": {
            "reused_documents": reused_documents,
            "split_documents": split_documents,
            "reused_embeddings": reused_embeddings,
            "new_embeddings": new_embeddings,
            "removed_vectors": max(0, len(previous_records) - reused_embeddings),
        },
    }
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        state_path,
        {"status": "ready", "version_id": version_id, "completed_at": utc_now()},
    )
    validate_index_manifest(manifest_path, root)
    return manifest_path


def load_index_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError(f"Unsupported index schema: {manifest.get('schema_version')}")
    if manifest.get("status") != "ready":
        raise ValueError(f"Index version is not ready: {manifest_path}")
    return manifest


def validate_index_manifest(manifest_path: Path, root: Path) -> dict[str, Any]:
    manifest = load_index_manifest(manifest_path)
    db_dir = resolve_stored_path(str(manifest["db_dir"]), root)
    chunks_path = resolve_stored_path(str(manifest["chunks_path"]), root)
    if not db_dir.is_dir() or not chunks_path.is_file():
        raise FileNotFoundError("Index manifest references missing local artifacts")
    chunks = read_jsonl(chunks_path)
    validate_chunks(chunks)
    if len(chunks) != int(manifest["chunk_count"]):
        raise ValueError("Manifest chunk_count does not match chunks file")
    chunks_hash = hash_payload(
        [(row["chunk_id"], row["text_hash"]) for row in chunks]
    )
    if chunks_hash != manifest.get("chunks_hash"):
        raise ValueError("Manifest chunks_hash does not match chunks file")

    manifest_sources = manifest.get("sources", {})
    if not isinstance(manifest_sources, dict):
        raise ValueError("Manifest sources must be an object")
    grouped = group_chunks_by_source(chunks)
    if set(grouped) != set(manifest_sources):
        raise ValueError("Manifest sources do not match chunks file")
    for source_id, rows in grouped.items():
        source = manifest_sources[source_id]
        expected_ids = [str(row["chunk_id"]) for row in rows]
        expected_hashes = [str(row["text_hash"]) for row in rows]
        if (
            int(source.get("chunk_count", -1)) != len(rows)
            or source.get("chunk_ids") != expected_ids
            or source.get("chunk_hashes") != expected_hashes
        ):
            raise ValueError(f"Manifest source state does not match chunks: {source_id}")
    documents_hash = hash_payload(
        {
            source_id: source["document_hash"]
            for source_id, source in sorted(manifest_sources.items())
        }
    )
    if documents_hash != manifest.get("documents_hash"):
        raise ValueError("Manifest documents_hash does not match source states")

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(str(manifest["collection"]))
    payload = collection.get(include=["documents", "metadatas"])
    actual_records = {
        str(chunk_id): (str(document), metadata or {})
        for chunk_id, document, metadata in zip(
            payload.get("ids") or [],
            payload.get("documents") or [],
            payload.get("metadatas") or [],
            strict=True,
        )
    }
    expected_ids = {str(row["chunk_id"]) for row in chunks}
    if collection.count() != len(chunks) or set(actual_records) != expected_ids:
        raise ValueError("Manifest Chroma collection does not match chunks file")
    for row in chunks:
        document, metadata = actual_records[str(row["chunk_id"])]
        if document != str(row["text"]):
            raise ValueError(f"Chroma document does not match chunk: {row['chunk_id']}")
        if (
            str(metadata.get("source_id", "")) != str(row["source_id"])
            or str(metadata.get("text_hash", "")) != str(row["text_hash"])
            or str(metadata.get("index_version", "")) != str(manifest["version_id"])
        ):
            raise ValueError(f"Chroma metadata does not match chunk: {row['chunk_id']}")
    if int(manifest["document_count"]) != len(manifest_sources):
        raise ValueError("Manifest document_count does not match source states")
    return manifest


def load_active_index(
    active_index_path: Path,
    root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    pointer = read_json(active_index_path)
    if pointer.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise ValueError("Unsupported active index pointer schema")
    manifest_path = resolve_stored_path(str(pointer["manifest_path"]), root)
    manifest = validate_index_manifest(manifest_path, root)
    if pointer.get("version_id") != manifest.get("version_id"):
        raise ValueError("Active pointer version does not match manifest")
    return pointer, manifest_path, manifest


def activate_index_manifest(
    active_index_path: Path,
    manifest_path: Path,
    root: Path,
) -> dict[str, Any]:
    manifest = validate_index_manifest(manifest_path, root)
    previous_manifest = None
    if active_index_path.exists():
        current = read_json(active_index_path)
        current_manifest = resolve_stored_path(str(current["manifest_path"]), root)
        if current_manifest == manifest_path.resolve():
            return current
        previous_manifest = store_path(current_manifest, root)
    pointer = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "version_id": manifest["version_id"],
        "manifest_path": store_path(manifest_path, root),
        "previous_manifest_path": previous_manifest,
        "activated_at": utc_now(),
    }
    write_json_atomic(active_index_path, pointer)
    return pointer


def rollback_active_index(active_index_path: Path, root: Path) -> dict[str, Any]:
    pointer, current_manifest_path, _ = load_active_index(active_index_path, root)
    previous_value = pointer.get("previous_manifest_path")
    if not previous_value:
        raise ValueError("Active index has no previous version to roll back to")
    previous_manifest_path = resolve_stored_path(str(previous_value), root)
    previous_manifest = validate_index_manifest(previous_manifest_path, root)
    rolled_back = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "version_id": previous_manifest["version_id"],
        "manifest_path": store_path(previous_manifest_path, root),
        "previous_manifest_path": store_path(current_manifest_path, root),
        "activated_at": utc_now(),
        "rollback": True,
    }
    write_json_atomic(active_index_path, rolled_back)
    return rolled_back
