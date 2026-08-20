"""Generate and audit RAG answers on a small answer-quality smoke set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.answer_audit import DEFAULT_AUDIT_MODEL, audit_answer_with_deepseek, combine_audits, deterministic_audit  # noqa: E402
from src.answer_repair import repair_answer_with_deepseek  # noqa: E402
from src.deepseek_client import DEFAULT_BASE_URL, DEFAULT_MODEL, chat_completion  # noqa: E402
from src.ollama_http import generate  # noqa: E402
from src.retrieval import RetrievedChunk, planned_retrieve  # noqa: E402


DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_COLLECTION = "llm_rag_docs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "answer_faithfulness_smoke"

ANSWER_CASES = [
    "文档切分为什么不能只用固定窗口？",
    "metadata filter 在 RAG 检索中有什么作用？",
    "rerank 和普通向量检索有什么关系？",
    "如何评估 RAG 回答是否忠实于检索上下文？",
]


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate answer faithfulness with deterministic and DeepSeek audit.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-provider", choices=["ollama", "deepseek"], default="ollama")
    parser.add_argument("--ollama-model", default="qwen2.5:1.5b")
    parser.add_argument("--deepseek-model", default=DEFAULT_MODEL)
    parser.add_argument("--deepseek-audit-model", default=DEFAULT_AUDIT_MODEL)
    parser.add_argument("--deepseek-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--skip-llm-audit", action="store_true")
    parser.add_argument("--repair-with-deepseek", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(name=args.collection)

    records = []
    for idx, query in enumerate(ANSWER_CASES, start=1):
        plan, retrieved = planned_retrieve(
            collection,
            query,
            args.embedding_model,
            args.ollama_host,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            rerank_mode="lexical",
        )
        context = build_context(retrieved, args.max_context_chars)
        prompt = build_prompt(query, context)
        draft_answer = generate_answer(args, prompt)
        answer = draft_answer
        rule_audit = deterministic_audit(answer, source_count=len(retrieved))
        repair_error = None
        if args.repair_with_deepseek and not rule_audit["rule_pass"]:
            try:
                answer = repair_answer_with_deepseek(
                    query,
                    context,
                    draft_answer,
                    rule_audit,
                    model=args.deepseek_model,
                    base_url=args.deepseek_base_url,
                )
                rule_audit = deterministic_audit(answer, source_count=len(retrieved))
            except Exception as exc:  # noqa: BLE001 - preserve the failed case.
                repair_error = f"{type(exc).__name__}: {exc}"

        llm_audit: dict[str, Any] | None = None
        audit_error = None
        if not args.skip_llm_audit:
            try:
                llm_audit = audit_answer_with_deepseek(
                    query,
                    context,
                    answer,
                    model=args.deepseek_audit_model,
                    base_url=args.deepseek_base_url,
                )
            except Exception as exc:  # noqa: BLE001 - keep batch results inspectable.
                audit_error = f"{type(exc).__name__}: {exc}"

        combined = combine_audits(rule_audit, llm_audit)
        record = {
            "index": idx,
            "query": query,
            "plan": plan.as_dict(),
            "sources": chunks_to_rows(retrieved),
            "draft_answer": draft_answer,
            "answer": answer,
            "audit": combined,
            "repair_error": repair_error,
            "audit_error": audit_error,
        }
        records.append(record)
        print(
            f"[{idx}/{len(ANSWER_CASES)}] overall_pass={combined['overall_pass']} "
            f"rule_pass={rule_audit['rule_pass']} repair_error={repair_error or 'none'} "
            f"audit_error={audit_error or 'none'} "
            f"query={query}",
            flush=True,
        )

    summary = build_summary(args, records)
    (args.output_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(summary_markdown(summary, records), encoding="utf-8")

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"results: {args.output_dir / 'results.jsonl'}")
    print(f"summary: {args.output_dir / 'summary.md'}")


def build_context(retrieved: list[RetrievedChunk], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for idx, item in enumerate(retrieved, start=1):
        metadata = item.metadata
        header = (
            f"[{idx}] title={metadata.get('title')} | "
            f"section={metadata.get('heading_path')} | "
            f"url={metadata.get('url')} | "
            f"chunk_id={item.chunk_id}"
        )
        block = f"{header}\n{item.document.strip()}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    return f"""你是一个大模型工程知识库助手。请只根据下面的资料回答问题。

领域规则：
- 本项目中，RAG 固定指 Retrieval-Augmented Generation，即检索增强生成。
- 不要把 RAG 解释成其他缩写。
- 如果检索资料没有支持某个判断，不要写这个判断。

强制输出格式：
结论：
- 一句话直接回答用户问题，并在句末标注来源编号，例如 [1]。

要点：
1. 一个关键点。必须在句末标注来源编号，例如 [1]。
2. 一个关键点。必须在句末标注来源编号，例如 [2]。

来源：
[1] 标题 - URL
[2] 标题 - URL

要求：
1. 用中文回答。
2. 每个关键结论和每个要点都必须标注来源编号。
3. 来源编号只能使用检索资料中已有的 [1]、[2]、[3] 等编号。
4. 如果资料不足，不要编造，直接说“根据当前知识库资料不足以回答”。
5. 不要引用资料外的事实。
6. 不要复制资料里的 Markdown 锚点、HTML 标签、代码装饰或页面导航文本。
7. 每个要点尽量说明“为什么有用/解决什么问题”。
8. 最后必须输出“来源：”列表。

用户问题：
{query}

检索资料：
{context}

请用中文回答：
"""


def generate_answer(args: argparse.Namespace, prompt: str) -> str:
    if args.llm_provider == "ollama":
        return generate(prompt, args.ollama_model, args.ollama_host, num_ctx=8192, num_predict=700)
    return chat_completion(
        [
            {"role": "system", "content": "你是严谨的大模型工程知识库助手。"},
            {"role": "user", "content": prompt},
        ],
        model=args.deepseek_model,
        base_url=args.deepseek_base_url,
        temperature=0.1,
        max_tokens=900,
    )


def chunks_to_rows(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    rows = []
    for chunk in chunks:
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.metadata.get("title"),
                "category": chunk.metadata.get("category"),
                "heading_path": chunk.metadata.get("heading_path"),
                "url": chunk.metadata.get("url"),
                "score": chunk.score,
                "distance": chunk.distance,
            }
        )
    return rows


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    rule_pass = sum(1 for record in records if record["audit"]["rule_audit"]["rule_pass"])
    overall_pass = sum(1 for record in records if record["audit"]["overall_pass"])
    audit_errors = sum(1 for record in records if record["audit_error"])
    repair_errors = sum(1 for record in records if record["repair_error"])
    return {
        "total": total,
        "llm_provider": args.llm_provider,
        "llm_audit": not args.skip_llm_audit,
        "deepseek_model": args.deepseek_model,
        "deepseek_audit_model": args.deepseek_audit_model,
        "repair_with_deepseek": args.repair_with_deepseek,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "rule_pass": rule_pass,
        "overall_pass": overall_pass,
        "audit_errors": audit_errors,
        "repair_errors": repair_errors,
        "rule_pass_rate": round(rule_pass / total, 4) if total else 0,
        "overall_pass_rate": round(overall_pass / total, 4) if total else 0,
    }


def summary_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Answer Faithfulness Smoke Evaluation",
        "",
        "## Summary",
        "",
        f"- total: {summary['total']}",
        f"- llm_provider: {summary['llm_provider']}",
        f"- llm_audit: {summary['llm_audit']}",
        f"- deepseek_model: {summary['deepseek_model']}",
        f"- deepseek_audit_model: {summary['deepseek_audit_model']}",
        f"- repair_with_deepseek: {summary['repair_with_deepseek']}",
        f"- top_k: {summary['top_k']}",
        f"- candidate_k: {summary['candidate_k']}",
        f"- rule pass: {summary['rule_pass']} ({summary['rule_pass_rate']:.2%})",
        f"- overall pass: {summary['overall_pass']} ({summary['overall_pass_rate']:.2%})",
        f"- audit errors: {summary['audit_errors']}",
        f"- repair errors: {summary['repair_errors']}",
        "",
        "## Cases",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"### {record['index']}. {record['query']}",
                "",
                f"- overall_pass: {record['audit']['overall_pass']}",
                f"- rule_pass: {record['audit']['rule_audit']['rule_pass']}",
                f"- repair_error: {record['repair_error'] or 'none'}",
                f"- audit_error: {record['audit_error'] or 'none'}",
                f"- sources: {', '.join(str(row['title']) for row in record['sources'])}",
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
