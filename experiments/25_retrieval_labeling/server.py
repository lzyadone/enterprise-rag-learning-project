"""Local web tool for human retrieval relevance annotation."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval_judgments import JudgmentStore, load_candidate_pools  # noqa: E402


STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CANDIDATE_POOLS = PROJECT_ROOT / "eval" / "planned_reranker_full" / "candidate_pools.jsonl"
DEFAULT_QRELS = PROJECT_ROOT / "eval" / "benchmarks" / "rag_retrieval_v1" / "qrels.jsonl"


class AnnotationState:
    def __init__(self, candidate_pool_path: Path, qrels_path: Path) -> None:
        self.candidate_pool_path = candidate_pool_path
        self.qrels_path = qrels_path
        self.pools = load_candidate_pools(candidate_pool_path)
        self.pool_by_id = {str(pool["case_id"]): pool for pool in self.pools}
        self.store = JudgmentStore(qrels_path, self.pools)

    def overview(self) -> dict[str, Any]:
        return {
            "progress": self.store.progress(),
            "cases": [
                {
                    "case_id": pool["case_id"],
                    "question": pool["question"],
                    "progress": self.store.progress_for(str(pool["case_id"])),
                }
                for pool in self.pools
            ],
            "qrels_path": portable_path(self.qrels_path),
            "candidate_pool_path": portable_path(self.candidate_pool_path),
        }

    def case(self, case_id: str) -> dict[str, Any]:
        pool = self.pool_by_id.get(case_id)
        if pool is None:
            raise KeyError(f"unknown case_id: {case_id}")
        candidates = []
        for pool_rank, candidate in enumerate(pool["candidates"], start=1):
            metadata = candidate.get("metadata") or {}
            judgment = self.store.get(case_id, str(candidate["chunk_id"]))
            candidates.append(
                {
                    "pool_rank": pool_rank,
                    "chunk_id": candidate["chunk_id"],
                    "document": candidate.get("document") or "",
                    "title": metadata.get("title") or metadata.get("source_id") or "Untitled source",
                    "category": metadata.get("category") or "uncategorized",
                    "heading_path": metadata.get("heading_path") or "",
                    "source_type": metadata.get("source_type") or "",
                    "url": metadata.get("url") or "",
                    "retrieval_channels": candidate.get("retrieval_channels") or [],
                    "retrieval_rank": candidate.get("rank"),
                    "retrieval_score": candidate.get("score"),
                    "aspect": candidate.get("aspect") or "",
                    "judgment": judgment,
                }
            )
        return {
            "case_id": case_id,
            "question": pool["question"],
            "progress": self.store.progress_for(case_id),
            "candidates": candidates,
        }


STATE: AnnotationState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retrieval relevance annotation tool.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--candidate-pools", type=Path, default=DEFAULT_CANDIDATE_POOLS)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    return parser.parse_args()


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    server_version = "RAGRetrievalLabeler/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/state":
            self.write_json({"ok": True, **STATE.overview()})
            return
        if path == "/api/case":
            case_id = str(parse_qs(parsed.query).get("case_id", [""])[0]).strip()
            if not case_id:
                self.write_json({"ok": False, "error": "case_id is required"}, status=400)
                return
            try:
                self.write_json({"ok": True, **STATE.case(case_id)})
            except KeyError as exc:
                self.write_json({"ok": False, "error": str(exc)}, status=404)
            return

        if path in {"", "/"}:
            path = "/index.html"
        file_path = (STATIC_DIR / path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(STATIC_DIR.resolve())) or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/judgment":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self.read_json()
            query_id = str(payload.get("query_id") or "")
            chunk_id = str(payload.get("chunk_id") or "")
            relevance = payload.get("relevance")
            if relevance is None:
                removed = STATE.store.delete(query_id, chunk_id)
                judgment = None
            else:
                removed = False
                judgment = STATE.store.upsert(
                    query_id,
                    chunk_id,
                    relevance,
                    note=str(payload.get("note") or ""),
                )
            self.write_json(
                {
                    "ok": True,
                    "judgment": judgment,
                    "removed": removed,
                    "progress": STATE.store.progress(),
                    "case_progress": STATE.store.progress_for(query_id),
                }
            )
        except ValueError as exc:
            self.write_json({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001
            self.write_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def write_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    global STATE
    args = parse_args()
    STATE = AnnotationState(args.candidate_pools, args.qrels)
    server = ThreadingHTTPServer((args.host, args.port), AnnotationRequestHandler)
    print(f"Retrieval labeler: http://{args.host}:{args.port}")
    print(f"Candidate pools: {portable_path(args.candidate_pools)}")
    print(f"Qrels: {portable_path(args.qrels)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
