"""Run the deterministic RAG security release gate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_security import (  # noqa: E402
    assess_evidence_security,
    assess_query_security,
    deterministic_security_audit,
    security_prompt_rules,
)
from src.retrieval import RetrievedChunk  # noqa: E402


DEFAULT_CASES = PROJECT_ROOT / "eval" / "benchmarks" / "rag_security_v1" / "cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "rag_security_v1" / "summary.json"


def make_chunks(documents: list[dict[str, Any]]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=f"security-{idx}",
            document=str(document["text"]),
            metadata=dict(document.get("metadata") or {}),
            distance=0.0,
            score=1.0,
            source_query="security regression",
            category_filter=None,
            aspect=None,
            rank=idx,
        )
        for idx, document in enumerate(documents, start=1)
    ]


def run(cases: dict[str, Any]) -> dict[str, Any]:
    results = []
    for case in cases["query_cases"]:
        assessment = assess_query_security(case["query"])
        checks = {
            "action": assessment.action == case["expected_action"],
            "category": not case.get("expected_category")
            or case["expected_category"] in assessment.risk_categories,
        }
        results.append(
            {
                "id": case["id"],
                "kind": "query",
                "expected_action": case["expected_action"],
                "actual_action": assessment.action,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )

    for case in cases["evidence_cases"]:
        evidence = assess_evidence_security(make_chunks(case["documents"]))
        checks: dict[str, bool] = {}
        if "expected_injection_source_ids" in case:
            checks["injection_sources"] = (
                evidence["evidence_injection_source_ids"] == case["expected_injection_source_ids"]
            )
            rules = security_prompt_rules(evidence)
            checks["untrusted_data_rule"] = "不可信数据" in rules and "[1]" in rules
            checks["canary_rejected"] = not deterministic_security_audit(
                "SECURITY_BYPASS_OK", evidence_security=evidence
            )["security_pass"]
        if "expected_conflict_count" in case:
            checks["conflict_count"] = len(evidence["conflicts"]) == case["expected_conflict_count"]
            checks["silent_conflict_rejected"] = not deterministic_security_audit(
                "结论：采用方案 A。[1]", evidence_security=evidence
            )["security_pass"]
            checks["disclosed_conflict_accepted"] = deterministic_security_audit(
                "结论：不同来源存在冲突。[1][2]", evidence_security=evidence
            )["security_pass"]
        results.append(
            {
                "id": case["id"],
                "kind": "evidence",
                "checks": checks,
                "passed": bool(checks) and all(checks.values()),
            }
        )

    passed = sum(1 for result in results if result["passed"])
    pass_rate = passed / len(results) if results else 0.0
    required = float(cases["gates"]["required_pass_rate"])
    gate_pass = pass_rate >= required and (
        not cases["gates"].get("require_all_categories") or passed == len(results)
    )
    return {
        "benchmark": cases["benchmark"],
        "version": cases["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": pass_rate,
            "required_pass_rate": required,
            "gate_pass": gate_pass,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    report = run(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = report["summary"]
    print(
        f"rag_security_v1: {summary['passed']}/{summary['total']} passed; "
        f"gate_pass={str(summary['gate_pass']).lower()}"
    )
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
