from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path

import chromadb
from chromadb.api.client import SharedSystemClient

from src.index_versioning import (
    ChunkingConfig,
    activate_index_manifest,
    build_incremental_chunks,
    chroma_metadata,
    chunk_document,
    hash_payload,
    load_active_index,
    materialize_index_version,
    plan_delta,
    read_jsonl,
    rollback_active_index,
    source_states,
    store_path,
    validate_index_manifest,
    write_json_atomic,
    write_jsonl,
)
from src.retrieval_cache import cache_candidates, retrieval_cache_info
from webapp.server import AppState, PROJECT_ROOT as WEB_PROJECT_ROOT


class IndexVersioningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ChunkingConfig(soft_max_chars=400, hard_max_chars=700, min_chars=40)

    def test_delta_classifies_added_changed_deleted_and_unchanged_sources(self) -> None:
        old_documents = [
            make_document("keep", "stable body"),
            make_document("change", "old body"),
            make_document("delete", "removed body"),
        ]
        old_chunks = build_chunks(old_documents, self.config)
        previous = make_manifest(old_documents, old_chunks, self.config)
        new_documents = [
            make_document("keep", "stable body"),
            make_document("change", "new body"),
            make_document("add", "added body"),
        ]

        plan = plan_delta(previous, new_documents, self.config)

        self.assertEqual(("add",), plan.added)
        self.assertEqual(("change",), plan.changed)
        self.assertEqual(("delete",), plan.deleted)
        self.assertEqual(("keep",), plan.unchanged)

    def test_candidate_reuses_embeddings_and_excludes_deleted_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            versions_dir = root / "versions"
            base_dir = root / "base_chroma"
            collection_name = "docs"
            old_documents = [
                make_document("keep", "stable body"),
                make_document("change", "old body"),
                make_document("delete", "removed body"),
            ]
            old_chunks = build_chunks(old_documents, self.config)
            base_client = chromadb.PersistentClient(path=str(base_dir))
            base_collection = base_client.get_or_create_collection(collection_name)
            base_collection.add(
                ids=[row["chunk_id"] for row in old_chunks],
                documents=[row["text"] for row in old_chunks],
                embeddings=[[float(index + 1), 0.5] for index, _ in enumerate(old_chunks)],
                metadatas=[chroma_metadata(row, "base") for row in old_chunks],
            )
            base_version_dir = versions_dir / "base"
            base_version_dir.mkdir(parents=True)
            base_chunks_path = base_version_dir / "chunks.jsonl"
            write_jsonl(base_chunks_path, old_chunks)
            base_manifest = make_manifest(old_documents, old_chunks, self.config)
            base_manifest.update(
                {
                    "db_dir": store_path(base_dir, root),
                    "chunks_path": store_path(base_chunks_path, root),
                    "collection": collection_name,
                }
            )
            base_manifest_path = base_version_dir / "manifest.json"
            write_json_atomic(base_manifest_path, base_manifest)

            new_documents = [
                make_document("keep", "stable body"),
                make_document("change", "new body"),
                make_document("add", "added body"),
            ]
            result = build_incremental_chunks(
                new_documents,
                self.config,
                base_manifest,
                old_chunks,
            )
            embedded_texts: list[str] = []

            def fake_embedder(texts: list[str], _model: str, _host: str) -> list[list[float]]:
                embedded_texts.extend(texts)
                return [[100.0 + index, 1.0] for index, _ in enumerate(texts)]

            candidate_manifest_path = materialize_index_version(
                root=root,
                versions_dir=versions_dir,
                version_id="candidate",
                documents=new_documents,
                chunks=list(result.chunks),
                config=self.config,
                embedding_model="fake-model",
                collection_name=collection_name,
                ollama_host="http://unused",
                batch_size=8,
                previous_db_dir=base_dir,
                previous_manifest=base_manifest,
                previous_manifest_path=base_manifest_path,
                delta=result.plan,
                reused_documents=result.reused_documents,
                split_documents=result.split_documents,
                embedder=fake_embedder,
            )

            candidate_manifest = load_active_manifest(candidate_manifest_path)
            candidate_db = root / candidate_manifest["db_dir"]
            candidate_client = chromadb.PersistentClient(path=str(candidate_db))
            candidate_collection = candidate_client.get_collection(collection_name)
            ids = set(candidate_collection.get(include=[])["ids"])

            self.assertEqual(3, candidate_collection.count())
            self.assertEqual(
                {"keep::chunk_0000", "change::chunk_0000", "add::chunk_0000"},
                ids,
            )
            self.assertNotIn("delete::chunk_0000", ids)
            self.assertEqual(2, len(embedded_texts))
            self.assertEqual(1, candidate_manifest["build"]["reused_embeddings"])
            self.assertEqual(2, candidate_manifest["build"]["new_embeddings"])
            self.assertEqual(1, candidate_manifest["build"]["reused_documents"])

            del candidate_collection, candidate_client, base_collection, base_client
            SharedSystemClient.clear_system_cache()
            gc.collect()

    def test_activation_and_rollback_switch_between_ready_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            active_path = root / "active.json"
            base_manifest = create_ready_version(root, "base", "docs", 1)
            candidate_manifest = create_ready_version(root, "candidate", "docs", 2)

            activate_index_manifest(active_path, base_manifest, root)
            activate_index_manifest(active_path, candidate_manifest, root)
            _, _, active = load_active_index(active_path, root)
            self.assertEqual("candidate", active["version_id"])

            rollback_active_index(active_path, root)
            pointer, _, rolled_back = load_active_index(active_path, root)
            self.assertEqual("base", rolled_back["version_id"])
            self.assertTrue(pointer["rollback"])

            SharedSystemClient.clear_system_cache()
            gc.collect()

    def test_validation_rejects_chunk_content_drift_with_unchanged_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = create_ready_version(root, "candidate", "docs", 1)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            chunks_path = root / str(manifest["chunks_path"])
            chunks = read_jsonl(chunks_path)
            chunks[0]["text_hash"] = "tampered"
            write_jsonl(chunks_path, chunks)

            with self.assertRaisesRegex(ValueError, "chunks_hash"):
                validate_index_manifest(manifest_path, root)

            SharedSystemClient.clear_system_cache()
            gc.collect()

    def test_web_state_hot_reloads_an_activated_index_without_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            active_path = root / "active.json"
            first_manifest = create_ready_version(root, "first", "docs", 1)
            second_manifest = create_ready_version(root, "second-version", "docs", 2)
            for manifest_path in (first_manifest, second_manifest):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["db_dir"] = str((root / manifest["db_dir"]).resolve())
                manifest["chunks_path"] = str((root / manifest["chunks_path"]).resolve())
                write_json_atomic(manifest_path, manifest)

            activate_index_manifest(active_path, first_manifest, WEB_PROJECT_ROOT)
            state = AppState(
                root / "unused_legacy",
                "docs",
                root / "memory.sqlite3",
                root / "memory_chroma",
                active_index_path=active_path,
            )
            first_runtime = state.index_runtime()
            self.assertEqual("first", first_runtime.version_id)
            self.assertEqual(1, first_runtime.collection.count())
            cache_candidates(("test-version",), ["cached"])
            self.assertEqual(1, retrieval_cache_info()["candidates"]["size"])

            activate_index_manifest(active_path, second_manifest, WEB_PROJECT_ROOT)
            second_runtime = state.index_runtime()
            self.assertEqual("second-version", second_runtime.version_id)
            self.assertEqual(2, second_runtime.collection.count())
            self.assertEqual(0, retrieval_cache_info()["candidates"]["size"])

            del state, first_runtime, second_runtime
            SharedSystemClient.clear_system_cache()
            gc.collect()


def make_document(source_id: str, body: str) -> dict[str, str]:
    text = f"# {source_id}\n\n{body} " + ("evidence " * 12)
    return {
        "doc_id": source_id,
        "source_id": source_id,
        "title": f"Title {source_id}",
        "category": "test",
        "priority": "P0",
        "source_type": "official_doc",
        "url": f"https://example.com/{source_id}",
        "text": text,
    }


def build_chunks(
    documents: list[dict[str, str]],
    config: ChunkingConfig,
) -> list[dict[str, object]]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, config)
    ]


def make_manifest(
    documents: list[dict[str, str]],
    chunks: list[dict[str, object]],
    config: ChunkingConfig,
) -> dict[str, object]:
    states = source_states(documents, chunks, config)
    return {
        "schema_version": 1,
        "status": "ready",
        "version_id": "base",
        "parent_version": None,
        "parent_manifest": None,
        "created_at": "2026-08-25T00:00:00+00:00",
        "embedding_model": "fake-model",
        "collection": "docs",
        "db_dir": "base_chroma",
        "chunks_path": "versions/base/chunks.jsonl",
        "document_count": len(states),
        "chunk_count": len(chunks),
        "documents_hash": hash_payload({key: value["document_hash"] for key, value in states.items()}),
        "chunks_hash": hash_payload([(row["chunk_id"], row["text_hash"]) for row in chunks]),
        "chunking": config.as_dict(),
        "sources": states,
        "delta": {"added": [], "changed": [], "deleted": [], "unchanged": []},
        "build": {
            "reused_documents": 0,
            "split_documents": len(documents),
            "reused_embeddings": 0,
            "new_embeddings": len(chunks),
            "removed_vectors": 0,
        },
    }


def load_active_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def create_ready_version(
    root: Path,
    version_id: str,
    collection_name: str,
    count: int,
) -> Path:
    version_dir = root / "versions" / version_id
    db_dir = version_dir / "chroma"
    chunks_path = version_dir / "chunks.jsonl"
    version_dir.mkdir(parents=True)
    rows = []
    for index in range(count):
        source_id = f"{version_id}-{index}"
        document = make_document(source_id, "version body")
        rows.extend(chunk_document(document, ChunkingConfig(400, 700, 40)))
    write_jsonl(chunks_path, rows)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(collection_name)
    collection.add(
        ids=[row["chunk_id"] for row in rows],
        documents=[row["text"] for row in rows],
        embeddings=[[float(index + 1), 0.5] for index, _ in enumerate(rows)],
        metadatas=[chroma_metadata(row, version_id) for row in rows],
    )
    states = {
        str(row["source_id"]): {
            "document_hash": str(row["text_hash"]),
            "chunk_count": 1,
            "chunk_ids": [str(row["chunk_id"])],
            "chunk_hashes": [str(row["text_hash"])],
        }
        for row in rows
    }
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "version_id": version_id,
        "parent_version": None,
        "parent_manifest": None,
        "created_at": "2026-08-25T00:00:00+00:00",
        "embedding_model": "fake-model",
        "collection": collection_name,
        "db_dir": store_path(db_dir, root),
        "chunks_path": store_path(chunks_path, root),
        "document_count": len(states),
        "chunk_count": len(rows),
        "documents_hash": hash_payload(
            {source_id: state["document_hash"] for source_id, state in states.items()}
        ),
        "chunks_hash": hash_payload([(row["chunk_id"], row["text_hash"]) for row in rows]),
        "chunking": ChunkingConfig(400, 700, 40).as_dict(),
        "sources": states,
        "delta": {"added": [], "changed": [], "deleted": [], "unchanged": []},
        "build": {},
    }
    manifest_path = version_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    del collection, client
    gc.collect()
    return manifest_path


if __name__ == "__main__":
    unittest.main()
