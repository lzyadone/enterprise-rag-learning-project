"""End-to-end smoke evaluation for the local RAG web API.

This script evaluates the same `/api/ask` path used by the web UI, so it tests
planning, retrieval, context assembly, generation, answer audit, repair, and
long-term memory together.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "datasets" / "rag_system_smoke_eval.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "rag_system_smoke"


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the local RAG web API with a JSONL dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--api-url", default="http://127.0.0.1:8765/api/ask")
    parser.add_argument("--clear-memory-url", default="http://127.0.0.1:8765/api/memory/clear")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases when greater than 0.")
    parser.add_argument("--case-id", action="append", default=[], help="Run only selected case ids. Repeatable.")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--llm-provider", choices=["deepseek", "ollama"], default="deepseek")
    parser.add_argument("--retrieval-mode", choices=["planned", "direct"], default="planned")
    parser.add_argument("--rerank-mode", choices=["lexical", "none"], default="lexical")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--candidate-k", type=int, default=16)
    parser.add_argument("--max-context-chars", type=int, default=15000)
    parser.add_argument("--memory-namespace", default="rag_eval")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--fail-on-required", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_cases(args.dataset)
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["id"] in wanted]
    if args.limit > 0:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No eval cases selected.")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    namespace = f"{args.memory_namespace}_{run_id}"
    clear_memory(args.clear_memory_url, namespace, args.timeout)

    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        record = run_case(args, case, namespace, index, len(cases))
        records.append(record)
        status = "PASS" if record["case_pass"] else "FAIL"
        print(
            f"[{index}/{len(cases)}] {status} id={case['id']} "
            f"quality={record['checks']['quality']['pass']} "
            f"sources={record['metrics']['source_count']} "
            f"time={record['metrics']['total_seconds']}s",
            flush=True,
        )

    summary = build_summary(args, namespace, records)
    results_path = args.output_dir / "results.jsonl"
    summary_json_path = args.output_dir / "summary.json"
    summary_md_path = args.output_dir / "summary.md"
    results_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_md_path.write_text(summary_markdown(summary, records), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {results_path}")
    print(f"summary: {summary_md_path}")

    if args.fail_on_required and summary["failed"]:
        raise SystemExit(1)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object")
        if not value.get("id") or not value.get("question"):
            raise ValueError(f"{path}:{line_no} must contain id and question")
        cases.append(value)
    return cases


def run_case(args: argparse.Namespace, case: dict[str, Any], namespace: str, index: int, total: int) -> dict[str, Any]:
    session_id = f"eval-{case['id']}-{uuid.uuid4()}"
    setup_response = None
    if case.get("setup_question"):
        setup_payload = build_payload(args, str(case["setup_question"]), namespace, session_id)
        setup_payload["audit_answer"] = False
        setup_payload["use_memory"] = True
        setup_payload["use_long_memory"] = True
        setup_payload["extract_long_memory"] = True
        setup_response = post_json(args.api_url, setup_payload, args.timeout)

    ask_session_id = f"eval-{case['id']}-ask-{uuid.uuid4()}"
    payload = build_payload(args, str(case["question"]), namespace, ask_session_id)
    if case.get("type") == "memory_check":
        payload["use_memory"] = True
        payload["use_long_memory"] = True
        payload["extract_long_memory"] = False
    response = post_json(args.api_url, payload, args.timeout)
    checks = evaluate_response(case, response)
    case_pass = all(check["pass"] for check in checks.values() if check["required"])

    return {
        "index": index,
        "total": total,
        "case": case,
        "case_pass": case_pass,
        "checks": checks,
        "metrics": response_metrics(response),
        "setup": summarize_setup(setup_response),
        "response": summarize_response(response),
    }


def build_payload(args: argparse.Namespace, question: str, namespace: str, session_id: str) -> dict[str, Any]:
    return {
        "query": question,
        "session_id": session_id,
        "llm_provider": args.llm_provider,
        "retrieval_mode": args.retrieval_mode,
        "rerank_mode": args.rerank_mode,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "max_context_chars": args.max_context_chars,
        "audit_answer": not args.skip_audit,
        "use_memory": False,
        "use_long_memory": False,
        "memory_namespace": namespace,
    }


def evaluate_response(case: dict[str, Any], response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not response.get("ok"):
        return {
            "response_ok": {
                "required": True,
                "pass": False,
                "detail": response.get("error", "response ok flag is false"),
            }
        }

    source_blob = build_source_blob(response)
    answer_blob = str(response.get("answer") or "")
    plan = response.get("plan") or {}
    sources = response.get("sources") or []
    audit = response.get("audit") or {}
    memory = ((response.get("memory") or {}).get("long_term") or {})

    checks: dict[str, dict[str, Any]] = {
        "response_ok": {"required": True, "pass": True, "detail": "ok"},
        "quality": {
            "required": bool(case.get("must_quality_pass")),
            "pass": (not case.get("must_quality_pass")) or bool(audit.get("quality_pass")),
            "detail": audit_quality_detail(audit),
        },
    }

    expected_aspects = list(case.get("expected_aspects") or [])
    if expected_aspects:
        actual_aspects = sorted(
            set(
                [str(aspect.get("name")) for aspect in plan.get("aspects", []) if aspect.get("name")]
                + [str(source.get("aspect")) for source in sources if source.get("aspect")]
            )
        )
        min_hits = int(case.get("min_aspect_hits", len(expected_aspects)))
        hits = sorted(set(expected_aspects).intersection(actual_aspects))
        checks["aspects"] = {
            "required": min_hits > 0,
            "pass": len(hits) >= min_hits,
            "detail": {"expected": expected_aspects, "actual": actual_aspects, "hits": hits, "min_hits": min_hits},
        }

    expected_categories = list(case.get("expected_categories") or [])
    if expected_categories:
        actual_categories = sorted(set(str(source.get("category")) for source in sources if source.get("category")))
        min_hits = int(case.get("min_category_hits", len(expected_categories)))
        hits = sorted(set(expected_categories).intersection(actual_categories))
        checks["categories"] = {
            "required": min_hits > 0,
            "pass": len(hits) >= min_hits,
            "detail": {"expected": expected_categories, "actual": actual_categories, "hits": hits, "min_hits": min_hits},
        }

    expected_source_terms = list(case.get("expected_source_terms") or [])
    if expected_source_terms:
        min_hits = int(case.get("min_source_term_hits", len(expected_source_terms)))
        hits = term_hits(source_blob, expected_source_terms)
        checks["source_terms"] = {
            "required": min_hits > 0,
            "pass": len(hits) >= min_hits,
            "detail": {"expected": expected_source_terms, "hits": hits, "min_hits": min_hits},
        }

    expected_answer_terms = list(case.get("expected_answer_terms") or [])
    if expected_answer_terms:
        min_hits = int(case.get("min_answer_term_hits", len(expected_answer_terms)))
        hits = term_hits(answer_blob, expected_answer_terms)
        checks["answer_terms"] = {
            "required": min_hits > 0,
            "pass": len(hits) >= min_hits,
            "detail": {"expected": expected_answer_terms, "hits": hits, "min_hits": min_hits},
        }

    if "expect_memory_answer_mode" in case:
        actual = bool((response.get("settings") or {}).get("memory_answer_mode"))
        expected = bool(case["expect_memory_answer_mode"])
        checks["memory_answer_mode"] = {
            "required": True,
            "pass": actual == expected,
            "detail": {"expected": expected, "actual": actual},
        }

    if "min_long_memory_retrieved" in case:
        retrieved_count = len(memory.get("retrieved") or [])
        min_count = int(case["min_long_memory_retrieved"])
        checks["long_memory_retrieved"] = {
            "required": True,
            "pass": retrieved_count >= min_count,
            "detail": {"expected_min": min_count, "actual": retrieved_count},
        }

    if "max_long_memory_stored" in case:
        stored_count = len(memory.get("stored") or [])
        max_count = int(case["max_long_memory_stored"])
        checks["long_memory_stored"] = {
            "required": True,
            "pass": stored_count <= max_count,
            "detail": {"expected_max": max_count, "actual": stored_count},
        }

    insufficient_terms = list(case.get("expected_insufficient_terms") or [])
    if insufficient_terms:
        hits = term_hits(answer_blob, insufficient_terms)
        checks["insufficient_boundary"] = {
            "required": True,
            "pass": bool(hits),
            "detail": {"expected_any": insufficient_terms, "hits": hits},
        }

    return checks


def build_source_blob(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for source in response.get("sources") or []:
        parts.extend(
            str(source.get(key) or "")
            for key in ["title", "category", "heading_path", "url", "aspect", "source_query", "preview"]
        )
    return "\n".join(parts)


def term_hits(text: str, terms: list[str]) -> list[str]:
    normalized = text.casefold()
    return [term for term in terms if str(term).casefold() in normalized]


def audit_quality_detail(audit: dict[str, Any]) -> dict[str, Any]:
    coverage = audit.get("coverage_audit") or {}
    llm = audit.get("llm_audit") or {}
    return {
        "quality_pass": audit.get("quality_pass"),
        "overall_pass": audit.get("overall_pass"),
        "coverage_pass": coverage.get("coverage_pass"),
        "faithfulness_score": llm.get("faithfulness_score"),
        "citation_score": llm.get("citation_score"),
        "relevance_score": llm.get("relevance_score"),
        "repair": audit.get("repair"),
        "audit_error": audit.get("audit_error"),
    }


def response_metrics(response: dict[str, Any]) -> dict[str, Any]:
    context = response.get("context") or {}
    timings = response.get("timings") or {}
    return {
        "source_count": len(response.get("sources") or []),
        "context_used_chars": context.get("used_chars"),
        "context_max_chars": context.get("max_chars"),
        "total_seconds": timings.get("total_seconds"),
    }


def summarize_setup(response: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response:
        return None
    memory = ((response.get("memory") or {}).get("long_term") or {})
    return {
        "ok": response.get("ok"),
        "query": response.get("query"),
        "stored_long_memory": len(memory.get("stored") or []),
        "memory_stats": memory.get("stats"),
        "error": response.get("error") or memory.get("error"),
    }


def summarize_response(response: dict[str, Any]) -> dict[str, Any]:
    memory = ((response.get("memory") or {}).get("long_term") or {})
    plan = response.get("plan") or {}
    return {
        "ok": response.get("ok"),
        "query": response.get("query"),
        "effective_query": response.get("effective_query"),
        "settings": response.get("settings"),
        "plan": plan,
        "sources": response.get("sources"),
        "answer": response.get("answer"),
        "audit": response.get("audit"),
        "context": response.get("context"),
        "memory": {
            "retrieved_count": len(memory.get("retrieved") or []),
            "stored_count": len(memory.get("stored") or []),
            "retrieved": memory.get("retrieved"),
            "stored": memory.get("stored"),
            "stats": memory.get("stats"),
            "error": memory.get("error"),
        },
        "timings": response.get("timings"),
        "error": response.get("error"),
    }


def build_summary(args: argparse.Namespace, namespace: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    passed = sum(1 for record in records if record["case_pass"])
    failed_records = [record for record in records if not record["case_pass"]]
    quality_pass = sum(1 for record in records if record["checks"].get("quality", {}).get("pass"))
    times = [
        float(record["metrics"]["total_seconds"])
        for record in records
        if isinstance(record["metrics"].get("total_seconds"), (int, float))
    ]
    return {
        "dataset": str(args.dataset),
        "api_url": args.api_url,
        "memory_namespace": namespace,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "quality_pass": quality_pass,
        "quality_pass_rate": round(quality_pass / total, 4) if total else 0,
        "avg_seconds": round(statistics.mean(times), 2) if times else None,
        "settings": {
            "llm_provider": args.llm_provider,
            "retrieval_mode": args.retrieval_mode,
            "rerank_mode": args.rerank_mode,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "max_context_chars": args.max_context_chars,
            "audit_answer": not args.skip_audit,
        },
        "badcases": [
            {
                "id": record["case"]["id"],
                "question": record["case"]["question"],
                "issues": collect_issues(record),
            }
            for record in failed_records
        ],
    }


def collect_issues(record: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    for name, check in record["checks"].items():
        if check["required"] and not check["pass"]:
            issues.append(f"{name}: {compact_detail(check['detail'])}")
    return issues


def compact_detail(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 220 else text[:217] + "..."


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# RAG System Smoke Evaluation",
        "",
        "## Summary",
        "",
        f"- total: {summary['total']}",
        f"- passed: {summary['passed']} ({summary['pass_rate']:.2%})",
        f"- quality_pass: {summary['quality_pass']} ({summary['quality_pass_rate']:.2%})",
        f"- avg_seconds: {summary['avg_seconds']}",
        f"- llm_provider: {summary['settings']['llm_provider']}",
        f"- retrieval_mode: {summary['settings']['retrieval_mode']}",
        f"- rerank_mode: {summary['settings']['rerank_mode']}",
        f"- top_k/candidate_k: {summary['settings']['top_k']}/{summary['settings']['candidate_k']}",
        f"- max_context_chars: {summary['settings']['max_context_chars']}",
        f"- audit_answer: {summary['settings']['audit_answer']}",
        "",
        "## Case Results",
        "",
        "| case | pass | quality | aspect | category | answer terms | sources | seconds | issues |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        checks = record["checks"]
        issues = collect_issues(record)
        lines.append(
            "| {case} | {case_pass} | {quality} | {aspects} | {categories} | {answer_terms} | {sources} | {seconds} | {issues} |".format(
                case=record["case"]["id"],
                case_pass=mark(record["case_pass"]),
                quality=mark(checks.get("quality", {}).get("pass")),
                aspects=mark(checks.get("aspects", {}).get("pass")) if "aspects" in checks else "-",
                categories=mark(checks.get("categories", {}).get("pass")) if "categories" in checks else "-",
                answer_terms=mark(checks.get("answer_terms", {}).get("pass")) if "answer_terms" in checks else "-",
                sources=record["metrics"]["source_count"],
                seconds=record["metrics"]["total_seconds"],
                issues="<br>".join(issues) if issues else "",
            )
        )

    if summary["badcases"]:
        lines.extend(["", "## Badcases", ""])
        for badcase in summary["badcases"]:
            lines.append(f"### {badcase['id']}")
            lines.append("")
            lines.append(f"- question: {badcase['question']}")
            for issue in badcase["issues"]:
                lines.append(f"- {issue}")
            lines.append("")

    return "\n".join(lines)


def mark(value: Any) -> str:
    if value is True:
        return "Y"
    if value is False:
        return "N"
    return "-"


def clear_memory(url: str, namespace: str, timeout: int) -> None:
    post_json(url, {"namespace": namespace}, timeout)


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json; charset=utf-8"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            value = json.loads(body or "{}")
            if not isinstance(value, dict):
                raise ValueError("API response is not a JSON object")
            return value
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            value = json.loads(body or "{}")
        except json.JSONDecodeError:
            value = {"ok": False, "error": body}
        if isinstance(value, dict):
            value.setdefault("ok", False)
            value.setdefault("http_status", exc.code)
            return value
        return {"ok": False, "http_status": exc.code, "error": body}
    except URLError as exc:
        return {"ok": False, "error": f"URLError: {exc.reason}"}


if __name__ == "__main__":
    main()
