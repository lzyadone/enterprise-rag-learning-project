"""Run a managed direct/planned-v3 end-to-end regression against the Web API."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "rag_web_mode_smoke_v1.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "runtime" / "web_e2e_regression"
MODES = ("direct", "planned")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the managed Web RAG mode regression.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-url", help="Use an existing server instead of starting a temporary one.")
    parser.add_argument("--case-id", action="append", default=[], help="Run only selected case ids.")
    parser.add_argument("--mode", action="append", choices=MODES, default=[], help="Run only selected modes.")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.dataset)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["id"] in wanted]
    if not cases:
        raise SystemExit("No Web regression cases found")

    process: subprocess.Popen[str] | None = None
    temp_memory: tempfile.TemporaryDirectory[str] | None = None
    if args.server_url:
        base_url = args.server_url.rstrip("/")
    else:
        port = available_port()
        base_url = f"http://127.0.0.1:{port}"
        temp_memory = tempfile.TemporaryDirectory(prefix="rag-web-e2e-")
        process = start_server(port, Path(temp_memory.name))

    started = time.time()
    try:
        wait_until_ready(base_url, process, timeout=60)
        modes = tuple(args.mode) or MODES
        records = run_matrix(base_url, cases, modes, args.timeout, args.max_seconds)
        summary = build_summary(base_url, records, modes, time.time() - started)
        write_results(args.output_dir, summary, records)
        print(summary_markdown(summary, records))
        if summary["failed"]:
            raise SystemExit(1)
    finally:
        if process is not None:
            stop_server(process)
        if temp_memory is not None:
            temp_memory.cleanup()


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("id") or not value.get("question"):
            raise ValueError(f"{path}:{line_no} must contain id and question")
        cases.append(value)
    return cases


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def start_server(port: int, memory_dir: Path) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "webapp" / "server.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--memory-sqlite",
        str(memory_dir / "memory.sqlite3"),
        "--memory-chroma-dir",
        str(memory_dir / "memory_chroma"),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )


def wait_until_ready(base_url: str, process: subprocess.Popen[str] | None, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            _, stderr = process.communicate(timeout=5)
            raise RuntimeError(f"Web server exited before readiness: {stderr[-1200:]}")
        try:
            status = get_json(f"{base_url}/api/status", timeout=5)
            if status.get("ok") and status.get("default_retrieval_mode") == "direct":
                return
        except (OSError, ValueError, URLError):
            pass
        time.sleep(0.5)
    raise TimeoutError("Web server did not become ready")


def run_matrix(
    base_url: str,
    cases: list[dict[str, Any]],
    modes: tuple[str, ...],
    timeout: int,
    max_seconds: float,
) -> list[dict[str, Any]]:
    records = []
    for case in cases:
        for mode in modes:
            started = time.time()
            response = post_json(
                f"{base_url}/api/ask",
                build_payload(str(case["question"]), mode),
                timeout=timeout,
            )
            wall_seconds = round(time.time() - started, 2)
            checks = evaluate_response(response, case, mode, wall_seconds, max_seconds)
            record = {
                "case_id": case["id"],
                "mode": mode,
                "passed": all(checks.values()),
                "checks": checks,
                "wall_seconds": wall_seconds,
                "reported_seconds": (response.get("timings") or {}).get("total_seconds"),
                "source_count": len(response.get("sources") or []),
                "source_categories": sorted(
                    {str(source.get("category") or "") for source in response.get("sources") or []}
                ),
                "provider_path": (response.get("generation") or {}).get("provider_path") or [],
            }
            records.append(record)
            print(
                f"[{case['id']}:{mode}] {'PASS' if record['passed'] else 'FAIL'} "
                f"sources={record['source_count']} time={wall_seconds}s",
                flush=True,
            )
    return records


def build_payload(question: str, mode: str) -> dict[str, Any]:
    return {
        "session_id": f"web-e2e-{uuid.uuid4()}",
        "query": question,
        "llm_provider": "ollama",
        "retrieval_mode": mode,
        "planned_fusion_mode": "conservative",
        "retrieval_strategy": "hybrid",
        "rerank_mode": "lexical",
        "top_k": 7,
        "candidate_k": 16,
        "max_context_chars": 9000,
        "latency_budget_ms": 12000,
        "use_memory": False,
        "use_long_memory": False,
        "audit_answer": False,
    }


def evaluate_response(
    response: dict[str, Any],
    case: dict[str, Any],
    mode: str,
    wall_seconds: float,
    max_seconds: float,
) -> dict[str, bool]:
    settings = response.get("settings") or {}
    routing = response.get("routing") or {}
    generation = response.get("generation") or {}
    sources = response.get("sources") or []
    answer = str(response.get("answer") or "")
    source_categories = {str(source.get("category") or "") for source in sources}
    expected_categories = {
        str(value)
        for value in case.get("expected_categories") or [case.get("expected_category")]
        if value
    }
    min_category_hits = int(case.get("min_category_hits", 1 if expected_categories else 0))
    citation_ids = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    valid_source_ids = {int(source.get("index")) for source in sources if source.get("index") is not None}
    plan = response.get("plan")
    plan_ok = plan is None if mode == "direct" else bool(
        isinstance(plan, dict) and plan.get("planner_version") == "rules_v3_conservative"
    )
    return {
        "response_ok": bool(response.get("ok")),
        "answer_nonempty": bool(answer.strip()),
        "sources_present": bool(sources),
        "source_urls_present": all(bool(str(source.get("url") or "").strip()) for source in sources),
        "expected_category_present": len(expected_categories.intersection(source_categories)) >= min_category_hits,
        "citation_present": bool(citation_ids.intersection(valid_source_ids)),
        "requested_mode": settings.get("requested_retrieval_mode") == mode,
        "selected_mode": settings.get("retrieval_mode") == mode and routing.get("selected_mode") == mode,
        "conservative_fusion": settings.get("planned_fusion_mode") == "conservative",
        "planner_shape": plan_ok,
        "local_generation": generation.get("provider") == "ollama",
        "latency_budget": wall_seconds <= max_seconds,
    }


def get_json(url: str, timeout: int) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("error") or f"HTTP {exc.code}"
        except json.JSONDecodeError:
            detail = f"HTTP {exc.code}"
        raise RuntimeError(detail) from exc


def build_summary(
    base_url: str,
    records: list[dict[str, Any]],
    modes: tuple[str, ...],
    elapsed: float,
) -> dict[str, Any]:
    passed = sum(1 for record in records if record["passed"])
    return {
        "base_url": base_url,
        "total": len(records),
        "passed": passed,
        "failed": len(records) - passed,
        "elapsed_seconds": round(elapsed, 2),
        "modes": list(modes),
    }


def write_results(output_dir: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(summary_markdown(summary, records), encoding="utf-8")


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Web E2E Regression",
        "",
        f"- Result: {'PASS' if not summary['failed'] else 'FAIL'}",
        f"- Cases: {summary['passed']}/{summary['total']}",
        f"- Elapsed: {summary['elapsed_seconds']}s",
        "",
        "| case | mode | result | sources | categories | seconds | failed checks |",
        "|---|---|---|---:|---|---:|---|",
    ]
    for record in records:
        failed_checks = ", ".join(name for name, passed in record["checks"].items() if not passed) or "-"
        lines.append(
            f"| {record['case_id']} | {record['mode']} | "
            f"{'PASS' if record['passed'] else 'FAIL'} | {record['source_count']} | "
            f"{', '.join(record['source_categories'])} | {record['wall_seconds']} | {failed_checks} |"
        )
    return "\n".join(lines) + "\n"


def stop_server(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


if __name__ == "__main__":
    main()
