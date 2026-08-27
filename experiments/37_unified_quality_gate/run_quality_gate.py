"""Run the shared local and CI quality gate for the RAG project."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "runtime" / "unified_quality_gate" / "report.json"
CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]
NodeResolver = Callable[[str], str | None]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_command_runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def run_stage(
    stage_id: str,
    command: list[str],
    root: Path,
    runner: CommandRunner,
) -> dict[str, Any]:
    started = time.perf_counter()
    completed = runner(command, root)
    combined = "\n".join([completed.stdout or "", completed.stderr or ""])
    result: dict[str, Any] = {
        "id": stage_id,
        "status": "passed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    if stage_id == "unit_tests":
        match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
        result["test_count"] = int(match.group(1)) if match else 0
        result["failed_tests"] = re.findall(
            r"^(?:FAIL|ERROR):\s+([^\s(]+)",
            combined,
            flags=re.MULTILINE,
        )
        if result["status"] == "passed" and result["test_count"] == 0:
            result["status"] = "failed"
    elif stage_id == "security_gate":
        match = re.search(r"rag_security_v1:\s+(\d+)/(\d+)\s+passed", combined)
        result["passed_count"] = int(match.group(1)) if match else 0
        result["case_count"] = int(match.group(2)) if match else 0
        if result["status"] == "passed" and result["case_count"] == 0:
            result["status"] = "failed"
    return result


def run_quality_gate(
    *,
    root: Path,
    profile: str,
    security_output: Path,
    manifest: Path | None = None,
    index_report_dir: Path | None = None,
    require_node: bool = False,
    runner: CommandRunner = default_command_runner,
    node_path: str | None = None,
    node_resolver: NodeResolver = shutil.which,
) -> dict[str, Any]:
    started_at = utc_now()
    stages = [
        run_stage("dependency_check", [sys.executable, "-m", "pip", "check"], root, runner),
        run_stage(
            "compile_python",
            [sys.executable, "-m", "compileall", "-q", "src", "experiments", "webapp", "tests"],
            root,
            runner,
        ),
        run_stage(
            "unit_tests",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
            root,
            runner,
        ),
        run_stage(
            "security_gate",
            [
                sys.executable,
                "experiments/36_rag_security_regression/run_security_regression.py",
                "--output",
                str(security_output),
            ],
            root,
            runner,
        ),
    ]

    resolved_node = node_path or node_resolver("node")
    if resolved_node:
        stages.append(
            run_stage(
                "javascript_syntax",
                [resolved_node, "--check", "webapp/static/app.js"],
                root,
                runner,
            )
        )
    else:
        node_required = require_node or profile == "ci"
        stages.append(
            {
                "id": "javascript_syntax",
                "status": "failed" if node_required else "skipped",
                "reason": "node executable not found",
            }
        )

    if manifest is not None:
        command = [
            sys.executable,
            "experiments/34_index_release_gate/run_gate.py",
            "--manifest",
            str(manifest),
        ]
        if index_report_dir is not None:
            command.extend(["--report-dir", str(index_report_dir)])
        stages.append(run_stage("index_release_gate", command, root, runner))
    else:
        stages.append(
            {
                "id": "index_release_gate",
                "status": "skipped",
                "reason": "no candidate index manifest supplied",
            }
        )

    failed = [stage["id"] for stage in stages if stage["status"] == "failed"]
    return {
        "schema_version": 1,
        "gate_id": "rag_unified_quality_gate_v1",
        "profile": profile,
        "started_at": started_at,
        "completed_at": utc_now(),
        "status": "failed" if failed else "passed",
        "failed_stages": failed,
        "stages": stages,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["local", "ci"], default="local")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--index-report-dir", type=Path)
    parser.add_argument("--require-node", action="store_true")
    parser.add_argument("--node-executable", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    security_output = output.parent / "security-summary.json"
    report = run_quality_gate(
        root=PROJECT_ROOT,
        profile=args.profile,
        security_output=security_output,
        manifest=args.manifest.resolve() if args.manifest else None,
        index_report_dir=args.index_report_dir.resolve() if args.index_report_dir else None,
        require_node=args.require_node,
        node_path=str(args.node_executable.resolve()) if args.node_executable else None,
    )
    write_report(output, report)
    for stage in report["stages"]:
        detail = ""
        if "test_count" in stage:
            detail = f" ({stage['test_count']} tests)"
        elif "case_count" in stage:
            detail = f" ({stage['passed_count']}/{stage['case_count']} cases)"
        print(f"{stage['id']}: {stage['status']}{detail}")
        if stage.get("failed_tests"):
            print(f"  failed_tests: {', '.join(stage['failed_tests'])}")
    print(f"unified_quality_gate: {report['status']}")
    print(f"report: {output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
