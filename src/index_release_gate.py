"""Offline release gate for immutable knowledge-base index versions."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import chromadb

from src.bm25_retrieval import clear_bm25_cache
from src.index_versioning import (
    activate_index_manifest,
    resolve_stored_path,
    store_path,
    validate_index_manifest,
    write_json_atomic,
)
from src.retrieval import RetrievedChunk, retrieve_with_strategy


GATE_SCHEMA_VERSION = 1
Retriever = Callable[[str], list[RetrievedChunk]]
RetrievalRunner = Callable[[dict[str, Any], dict[str, Any], Path], dict[str, Any]]
TestRunner = Callable[[Path], dict[str, Any]]
Activator = Callable[[Path, Path, Path], dict[str, Any]]
StructuralValidator = Callable[[Path, Path], dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gate_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Release gate spec must be a JSON object")
    if payload.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported release gate schema: {payload.get('schema_version')}")
    retrieval = payload.get("retrieval")
    cases = payload.get("cases")
    if not isinstance(retrieval, dict) or not isinstance(cases, list) or not cases:
        raise ValueError("Release gate spec requires retrieval settings and cases")
    min_pass_rate = float(retrieval.get("min_pass_rate", -1))
    if not 0.0 <= min_pass_rate <= 1.0:
        raise ValueError("retrieval.min_pass_rate must be between 0 and 1")
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not str(case.get("id", "")).strip():
            raise ValueError("Every release gate case requires an id")
        if not str(case.get("question", "")).strip():
            raise ValueError(f"Release gate case requires a question: {case.get('id')}")
        case_id = str(case["id"])
        if case_id in case_ids:
            raise ValueError(f"Duplicate release gate case: {case_id}")
        case_ids.add(case_id)
    return payload


def searchable_text(chunk: RetrievedChunk) -> str:
    return "\n".join(
        [
            str(chunk.metadata.get("title", "")),
            str(chunk.metadata.get("category", "")),
            str(chunk.metadata.get("heading_path", "")),
            chunk.document,
        ]
    )


def evaluate_retrieval_case(
    case: dict[str, Any],
    chunks: list[RetrievedChunk],
    top_k: int,
) -> dict[str, Any]:
    selected = chunks[:top_k]
    ranked_chunk_ids = [chunk.chunk_id for chunk in selected]
    actual_categories = [str(chunk.metadata.get("category", "")) for chunk in selected]

    expected_categories = [str(value) for value in case.get("expected_categories") or []]
    category_hits = sorted(set(actual_categories).intersection(expected_categories))
    min_category_hits = int(case.get("min_category_hits", 0))
    category_pass = len(category_hits) >= min_category_hits

    source_blob = "\n".join(searchable_text(chunk) for chunk in selected).casefold()
    expected_terms = [str(value) for value in case.get("expected_source_terms") or []]
    source_term_hits = [term for term in expected_terms if term.casefold() in source_blob]
    min_source_term_hits = int(case.get("min_source_term_hits", 0))
    source_terms_pass = len(source_term_hits) >= min_source_term_hits

    expected_chunk_ids = [str(value) for value in case.get("expected_chunk_ids") or []]
    max_chunk_rank = int(case.get("max_chunk_rank", top_k))
    anchor_window = set(ranked_chunk_ids[:max_chunk_rank])
    chunk_hits = [chunk_id for chunk_id in expected_chunk_ids if chunk_id in anchor_window]
    min_chunk_hits = int(case.get("min_chunk_hits", 0))
    chunks_pass = len(chunk_hits) >= min_chunk_hits

    return {
        "case_id": str(case["id"]),
        "required": bool(case.get("required", False)),
        "passed": category_pass and source_terms_pass and chunks_pass,
        "category_pass": category_pass,
        "category_hits": category_hits,
        "min_category_hits": min_category_hits,
        "source_terms_pass": source_terms_pass,
        "source_term_hits": source_term_hits,
        "min_source_term_hits": min_source_term_hits,
        "chunks_pass": chunks_pass,
        "chunk_hits": chunk_hits,
        "min_chunk_hits": min_chunk_hits,
        "max_chunk_rank": max_chunk_rank,
        "ranked_chunk_ids": ranked_chunk_ids,
    }


def run_retrieval_gate(
    manifest: dict[str, Any],
    spec: dict[str, Any],
    root: Path,
    retriever: Retriever | None = None,
) -> dict[str, Any]:
    settings = spec["retrieval"]
    top_k = int(settings.get("top_k", 7))
    candidate_k = int(settings.get("candidate_k", 16))
    if top_k <= 0 or candidate_k < top_k:
        raise ValueError("Release gate requires candidate_k >= top_k > 0")

    if retriever is None:
        db_dir = resolve_stored_path(str(manifest["db_dir"]), root)
        chunks_path = resolve_stored_path(str(manifest["chunks_path"]), root)
        collection = chromadb.PersistentClient(path=str(db_dir)).get_collection(
            str(manifest["collection"])
        )
        clear_bm25_cache()

        def retrieve(query: str) -> list[RetrievedChunk]:
            return retrieve_with_strategy(
                collection,
                query,
                str(manifest["embedding_model"]),
                str(settings.get("ollama_host", "http://127.0.0.1:11434")),
                top_k=top_k,
                candidate_k=candidate_k,
                rerank_mode=str(settings.get("rerank_mode", "lexical")),
                retrieval_strategy=str(settings.get("strategy", "hybrid")),
                chunks_path=chunks_path,
            )

        retriever = retrieve

    started = time.perf_counter()
    results = [
        evaluate_retrieval_case(case, retriever(str(case["question"])), top_k)
        for case in spec["cases"]
    ]
    passed_count = sum(1 for result in results if result["passed"])
    pass_rate = passed_count / len(results)
    required_failures = [
        str(result["case_id"])
        for result in results
        if result["required"] and not result["passed"]
    ]
    min_pass_rate = float(settings["min_pass_rate"])
    return {
        "status": "passed" if pass_rate >= min_pass_rate and not required_failures else "failed",
        "case_count": len(results),
        "passed_count": passed_count,
        "pass_rate": round(pass_rate, 4),
        "min_pass_rate": min_pass_rate,
        "required_failures": required_failures,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "settings": {
            "strategy": str(settings.get("strategy", "hybrid")),
            "rerank_mode": str(settings.get("rerank_mode", "lexical")),
            "top_k": top_k,
            "candidate_k": candidate_k,
        },
        "cases": results,
    }


def run_unit_tests(project_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    combined = "\n".join([completed.stdout, completed.stderr])
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    test_count = int(match.group(1)) if match else 0
    passed = completed.returncode == 0 and test_count > 0 and "OK" in combined
    return {
        "status": "passed" if passed else "failed",
        "test_count": test_count,
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def execute_release_gate(
    *,
    root: Path,
    manifest_path: Path,
    gate_spec_path: Path,
    report_dir: Path,
    active_index_path: Path,
    activate: bool = False,
    retrieval_runner: RetrievalRunner = run_retrieval_gate,
    test_runner: TestRunner = run_unit_tests,
    activator: Activator = activate_index_manifest,
    structural_validator: StructuralValidator = validate_index_manifest,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate_id": None,
        "started_at": utc_now(),
        "completed_at": None,
        "status": "running",
        "manifest_path": store_path(manifest_path, root),
        "manifest_sha256": sha256_file(manifest_path),
        "gate_spec_path": store_path(gate_spec_path, root),
        "gate_spec_sha256": sha256_file(gate_spec_path),
        "candidate": None,
        "stages": {
            "structure": {"status": "pending"},
            "retrieval": {"status": "pending"},
            "tests": {"status": "pending"},
        },
        "activation": {
            "requested": activate,
            "status": "pending" if activate else "not_requested",
            "version_id": None,
        },
    }
    try:
        spec = load_gate_spec(gate_spec_path)
        report["gate_id"] = spec.get("gate_id")
        manifest = structural_validator(manifest_path, root)
        report["candidate"] = {
            "version_id": manifest["version_id"],
            "parent_version": manifest.get("parent_version"),
            "document_count": manifest["document_count"],
            "chunk_count": manifest["chunk_count"],
            "embedding_model": manifest["embedding_model"],
            "documents_hash": manifest["documents_hash"],
            "chunks_hash": manifest["chunks_hash"],
            "delta_counts": {
                key: len(manifest.get("delta", {}).get(key, []))
                for key in ("added", "changed", "deleted", "unchanged")
            },
            "build": manifest["build"],
        }
        report["stages"]["structure"] = {"status": "passed"}
    except Exception as exc:
        report["stages"]["structure"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        report["stages"]["retrieval"] = {"status": "skipped"}
        report["stages"]["tests"] = {"status": "skipped"}
    else:
        try:
            report["stages"]["retrieval"] = retrieval_runner(manifest, spec, root)
        except Exception as exc:
            report["stages"]["retrieval"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        try:
            report["stages"]["tests"] = test_runner(root)
        except Exception as exc:
            report["stages"]["tests"] = {
                "status": "failed",
                "test_count": 0,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    stages_passed = all(
        stage.get("status") == "passed" for stage in report["stages"].values()
    )
    report["status"] = "passed" if stages_passed else "failed"
    if activate and stages_passed:
        report["activation"]["status"] = "approved"
        report["completed_at"] = utc_now()
        write_release_report(report_dir, report)
        try:
            pointer = activator(active_index_path, manifest_path, root)
            report["activation"] = {
                "requested": True,
                "status": "activated",
                "version_id": pointer["version_id"],
            }
        except Exception as exc:
            report["status"] = "failed"
            report["activation"] = {
                "requested": True,
                "status": "failed",
                "version_id": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    elif activate:
        report["activation"]["status"] = "blocked"

    report["completed_at"] = utc_now()
    write_release_report(report_dir, report)
    return report


def write_release_report(report_dir: Path, report: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_dir / "report.json", report)
    markdown_path = report_dir / "report.md"
    temp_path = markdown_path.with_name(markdown_path.name + ".tmp")
    try:
        temp_path.write_text(report_markdown(report), encoding="utf-8")
        temp_path.replace(markdown_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def report_markdown(report: dict[str, Any]) -> str:
    candidate = report.get("candidate") or {}
    lines = [
        "# Index Release Gate",
        "",
        f"- status: {report['status']}",
        f"- gate: {report.get('gate_id')}",
        f"- version: {candidate.get('version_id')}",
        f"- documents: {candidate.get('document_count')}",
        f"- chunks: {candidate.get('chunk_count')}",
        f"- activation: {report['activation']['status']}",
        "",
        "## Stages",
        "",
        "| stage | status | detail |",
        "|---|---|---|",
    ]
    retrieval = report["stages"]["retrieval"]
    tests = report["stages"]["tests"]
    details = {
        "structure": "manifest, hashes, source state, chunks and Chroma",
        "retrieval": (
            f"{retrieval.get('passed_count', 0)}/{retrieval.get('case_count', 0)}; "
            f"required failures: {', '.join(retrieval.get('required_failures', [])) or 'none'}"
        ),
        "tests": f"{tests.get('test_count', 0)} tests",
    }
    for name, stage in report["stages"].items():
        lines.append(f"| {name} | {stage['status']} | {details[name]} |")
    if retrieval.get("cases"):
        lines.extend(
            [
                "",
                "## Retrieval Cases",
                "",
                "| case | required | status | anchors |",
                "|---|---:|---|---|",
            ]
        )
        for case in retrieval["cases"]:
            lines.append(
                f"| {case['case_id']} | {'yes' if case['required'] else 'no'} | "
                f"{'passed' if case['passed'] else 'failed'} | "
                f"{', '.join(case['chunk_hits']) or '-'} |"
            )
    lines.append("")
    return "\n".join(lines)
