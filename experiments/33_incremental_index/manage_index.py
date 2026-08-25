"""Build, activate and roll back versioned incremental knowledge-base indexes."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.index_versioning import (  # noqa: E402
    ChunkingConfig,
    DeltaPlan,
    activate_index_manifest,
    build_incremental_chunks,
    load_active_index,
    materialize_index_version,
    plan_delta,
    read_jsonl,
    rollback_active_index,
    validate_index_manifest,
)


DEFAULT_DOCUMENTS = PROJECT_ROOT / "data" / "processed" / "llm_rag_docs" / "documents.jsonl"
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "llm_rag_docs" / "chunks.jsonl"
DEFAULT_BASE_DB = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_VERSIONS_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_versions"
DEFAULT_ACTIVE_INDEX = PROJECT_ROOT / "data" / "runtime" / "active_index.json"
DEFAULT_COLLECTION = "llm_rag_docs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--base-db-dir", type=Path, default=DEFAULT_BASE_DB)
    parser.add_argument("--versions-dir", type=Path, default=DEFAULT_VERSIONS_DIR)
    parser.add_argument("--active-index", type=Path, default=DEFAULT_ACTIVE_INDEX)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--soft-max-chars", type=int, default=1800)
    parser.add_argument("--hard-max-chars", type=int, default=3500)
    parser.add_argument("--min-chars", type=int, default=280)
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap", help="Create the first immutable version from the current index.")
    bootstrap.add_argument("--version-id", default=None)

    subparsers.add_parser("plan", help="Show source changes without building an index.")

    build = subparsers.add_parser("build", help="Build an incremental candidate index.")
    build.add_argument("--version-id", default=None)
    build.add_argument("--force", action="store_true", help="Materialize a version even when no source changed.")

    validate = subparsers.add_parser("validate", help="Validate a candidate manifest and its artifacts.")
    validate.add_argument("--manifest", type=Path, required=True)

    activate = subparsers.add_parser("activate", help="Activate a ready candidate manifest.")
    activate.add_argument("--manifest", type=Path, required=True)

    subparsers.add_parser("rollback", help="Atomically switch back to the previous active version.")
    subparsers.add_parser("status", help="Show the active index version and counts.")
    return parser.parse_args()


def default_version_id(prefix: str = "index") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}"


def chunking_config(args: argparse.Namespace) -> ChunkingConfig:
    return ChunkingConfig(
        soft_max_chars=args.soft_max_chars,
        hard_max_chars=args.hard_max_chars,
        min_chars=args.min_chars,
    )


def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def manifest_summary(manifest: dict[str, object]) -> dict[str, object]:
    delta = manifest.get("delta", {})
    return {
        "version_id": manifest["version_id"],
        "parent_version": manifest.get("parent_version"),
        "document_count": manifest["document_count"],
        "chunk_count": manifest["chunk_count"],
        "embedding_model": manifest["embedding_model"],
        "delta_counts": {
            key: len(delta.get(key, []))
            for key in ("added", "changed", "deleted", "unchanged")
        },
        "build": manifest["build"],
    }


def pointer_summary(pointer: dict[str, object] | None) -> dict[str, object] | None:
    if pointer is None:
        return None
    return {
        "version_id": pointer["version_id"],
        "manifest_path": pointer["manifest_path"],
        "previous_manifest_path": pointer.get("previous_manifest_path"),
        "rollback": bool(pointer.get("rollback", False)),
    }


def bootstrap(args: argparse.Namespace) -> None:
    if args.active_index.exists():
        raise FileExistsError(f"Active index already exists: {args.active_index}")
    documents = read_jsonl(args.documents)
    chunks = read_jsonl(args.chunks)
    source_ids = tuple(sorted({str(row["source_id"]) for row in documents}))
    delta = DeltaPlan(added=source_ids, changed=(), deleted=(), unchanged=())
    manifest_path = materialize_index_version(
        root=PROJECT_ROOT,
        versions_dir=args.versions_dir,
        version_id=args.version_id or default_version_id("baseline"),
        documents=documents,
        chunks=chunks,
        config=chunking_config(args),
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        ollama_host=args.ollama_host,
        batch_size=args.batch_size,
        previous_db_dir=args.base_db_dir,
        previous_manifest=None,
        previous_manifest_path=None,
        delta=delta,
        reused_documents=0,
        split_documents=0,
    )
    pointer = activate_index_manifest(args.active_index, manifest_path, PROJECT_ROOT)
    _, _, manifest = load_active_index(args.active_index, PROJECT_ROOT)
    print_json(
        {
            "action": "bootstrap",
            "active": pointer_summary(pointer),
            "manifest": manifest_summary(manifest),
        }
    )


def active_inputs(args: argparse.Namespace):
    pointer, manifest_path, manifest = load_active_index(args.active_index, PROJECT_ROOT)
    previous_chunks = read_jsonl(
        Path(manifest["chunks_path"])
        if Path(str(manifest["chunks_path"])).is_absolute()
        else PROJECT_ROOT / str(manifest["chunks_path"])
    )
    documents = read_jsonl(args.documents)
    return pointer, manifest_path, manifest, previous_chunks, documents


def show_plan(args: argparse.Namespace) -> None:
    _, _, manifest, _, documents = active_inputs(args)
    plan = plan_delta(manifest, documents, chunking_config(args))
    print_json({"active_version": manifest["version_id"], "delta": plan.as_dict(), "has_changes": plan.has_changes})


def build(args: argparse.Namespace) -> None:
    _, previous_manifest_path, previous_manifest, previous_chunks, documents = active_inputs(args)
    result = build_incremental_chunks(
        documents,
        chunking_config(args),
        previous_manifest,
        previous_chunks,
    )
    if not result.plan.has_changes and not args.force:
        print_json({"action": "no_changes", "active_version": previous_manifest["version_id"], "delta": result.plan.as_dict()})
        return
    previous_db_dir = Path(str(previous_manifest["db_dir"]))
    if not previous_db_dir.is_absolute():
        previous_db_dir = PROJECT_ROOT / previous_db_dir
    manifest_path = materialize_index_version(
        root=PROJECT_ROOT,
        versions_dir=args.versions_dir,
        version_id=args.version_id or default_version_id(),
        documents=documents,
        chunks=list(result.chunks),
        config=chunking_config(args),
        embedding_model=args.embedding_model,
        collection_name=args.collection,
        ollama_host=args.ollama_host,
        batch_size=args.batch_size,
        previous_db_dir=previous_db_dir,
        previous_manifest=previous_manifest,
        previous_manifest_path=previous_manifest_path,
        delta=result.plan,
        reused_documents=result.reused_documents,
        split_documents=result.split_documents,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print_json(
        {
            "action": "build",
            "manifest_path": str(manifest_path),
            "manifest": manifest_summary(manifest),
            "active": None,
        }
    )


def validate(args: argparse.Namespace) -> None:
    manifest = validate_index_manifest(args.manifest, PROJECT_ROOT)
    print_json(
        {
            "action": "validate",
            "manifest_path": str(args.manifest.resolve()),
            "manifest": manifest_summary(manifest),
            "valid": True,
        }
    )


def activate(args: argparse.Namespace) -> None:
    pointer = activate_index_manifest(args.active_index, args.manifest, PROJECT_ROOT)
    print_json({"action": "activate", "active": pointer_summary(pointer)})


def rollback(args: argparse.Namespace) -> None:
    pointer = rollback_active_index(args.active_index, PROJECT_ROOT)
    print_json({"action": "rollback", "active": pointer_summary(pointer)})


def status(args: argparse.Namespace) -> None:
    pointer, manifest_path, manifest = load_active_index(args.active_index, PROJECT_ROOT)
    client = chromadb.PersistentClient(path=str(
        Path(manifest["db_dir"]) if Path(str(manifest["db_dir"])).is_absolute() else PROJECT_ROOT / str(manifest["db_dir"])
    ))
    collection = client.get_collection(str(manifest["collection"]))
    print_json(
        {
            "active": pointer_summary(pointer),
            "manifest_path": str(manifest_path),
            "document_count": manifest["document_count"],
            "chunk_count": manifest["chunk_count"],
            "indexed_count": collection.count(),
            "build": manifest["build"],
        }
    )


def main() -> None:
    args = parse_args()
    handlers = {
        "bootstrap": bootstrap,
        "plan": show_plan,
        "build": build,
        "validate": validate,
        "activate": activate,
        "rollback": rollback,
        "status": status,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
