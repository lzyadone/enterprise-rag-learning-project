"""Run the offline release gate for a candidate index version."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.index_release_gate import execute_release_gate  # noqa: E402


DEFAULT_GATE_SPEC = (
    PROJECT_ROOT / "eval" / "benchmarks" / "rag_index_release_gate_v1" / "gate.json"
)
DEFAULT_ACTIVE_INDEX = PROJECT_ROOT / "data" / "runtime" / "active_index.json"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "data" / "runtime" / "index_release_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gate-spec", type=Path, default=DEFAULT_GATE_SPEC)
    parser.add_argument("--active-index", type=Path, default=DEFAULT_ACTIVE_INDEX)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Activate only after every release gate stage passes.",
    )
    return parser.parse_args()


def default_report_dir(manifest_path: Path) -> Path:
    version_id = manifest_path.parent.name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_REPORT_ROOT / f"{version_id}-{timestamp}"


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir or default_report_dir(args.manifest)
    report = execute_release_gate(
        root=PROJECT_ROOT,
        manifest_path=args.manifest.resolve(),
        gate_spec_path=args.gate_spec.resolve(),
        report_dir=report_dir.resolve(),
        active_index_path=args.active_index.resolve(),
        activate=args.activate,
    )
    retrieval = report["stages"]["retrieval"]
    tests = report["stages"]["tests"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "version_id": (report.get("candidate") or {}).get("version_id"),
                "structure": report["stages"]["structure"]["status"],
                "retrieval": retrieval["status"],
                "retrieval_passed": retrieval.get("passed_count"),
                "retrieval_cases": retrieval.get("case_count"),
                "tests": tests["status"],
                "test_count": tests.get("test_count"),
                "activation": report["activation"]["status"],
                "report_dir": str(report_dir.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
