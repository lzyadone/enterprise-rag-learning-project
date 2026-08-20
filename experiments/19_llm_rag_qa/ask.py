"""Ask the local LLM/RAG engineering knowledge base."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ollama_http import generate  # noqa: E402
from src.answer_audit import DEFAULT_AUDIT_MODEL, audit_answer_with_deepseek, combine_audits, deterministic_audit  # noqa: E402
from src.answer_repair import repair_answer_with_deepseek  # noqa: E402
from src.deepseek_client import DEFAULT_BASE_URL, DEFAULT_MODEL, chat_completion  # noqa: E402
from src.query_planning import QueryPlan  # noqa: E402
from src.retrieval import RetrievedChunk, direct_retrieve, planned_retrieve  # noqa: E402

DEFAULT_DB_DIR = PROJECT_ROOT / "data" / "indexes" / "llm_rag_chroma"
DEFAULT_COLLECTION = "llm_rag_docs"


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve evidence and generate an answer.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--llm-provider", choices=["ollama", "deepseek"], default="ollama")
    parser.add_argument("--llm-model", default="qwen2.5:1.5b")
    parser.add_argument("--deepseek-model", default=DEFAULT_MODEL)
    parser.add_argument("--deepseek-audit-model", default=DEFAULT_AUDIT_MODEL)
    parser.add_argument("--deepseek-base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--category", default=None)
    parser.add_argument("--retrieval-mode", choices=["direct", "planned"], default="planned")
    parser.add_argument("--rerank-mode", choices=["none", "lexical"], default="lexical")
    parser.add_argument("--max-plan-categories", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=9000)
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--audit-answer", action="store_true")
    parser.add_argument("--repair-answer", action="store_true")
    parser.add_argument("--show-plan", action="store_true", default=True)
    return parser.parse_args()


def retrieve(args: argparse.Namespace) -> tuple[QueryPlan | None, list[RetrievedChunk]]:
    client = chromadb.PersistentClient(path=str(args.db_dir))
    collection = client.get_collection(name=args.collection)
    if args.retrieval_mode == "direct":
        return None, direct_retrieve(
            collection,
            args.query,
            args.embedding_model,
            args.ollama_host,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            category=args.category,
            rerank_mode=args.rerank_mode,
        )
    return planned_retrieve(
        collection,
        args.query,
        args.embedding_model,
        args.ollama_host,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_categories=args.max_plan_categories,
        manual_category=args.category,
        rerank_mode=args.rerank_mode,
    )


def build_context(retrieved: list[RetrievedChunk], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for idx, item in enumerate(retrieved, start=1):
        metadata = item.metadata
        text = item.document.strip()
        header = (
            f"[{idx}] title={metadata.get('title')} | "
            f"section={metadata.get('heading_path')} | "
            f"url={metadata.get('url')} | "
            f"chunk_id={item.chunk_id}"
        )
        block = f"{header}\n{text}"
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

请给出答案：
"""


def generate_answer(args: argparse.Namespace, prompt: str) -> str:
    if args.llm_provider == "ollama":
        return generate(prompt, args.llm_model, args.ollama_host, num_ctx=8192, num_predict=700)

    answer = chat_completion(
        [
            {
                "role": "system",
                "content": "你是严谨的大模型工程知识库助手。必须只根据用户给定的检索资料回答。",
            },
            {"role": "user", "content": prompt},
        ],
        model=args.deepseek_model,
        base_url=args.deepseek_base_url,
        temperature=0.1,
        max_tokens=900,
    )
    if answer.strip():
        return answer
    if args.deepseek_model != "deepseek-chat":
        answer = chat_completion(
            [
                {
                    "role": "system",
                    "content": "你是严谨的大模型工程知识库助手。必须只根据用户给定的检索资料回答。",
                },
                {"role": "user", "content": prompt},
            ],
            model="deepseek-chat",
            base_url=args.deepseek_base_url,
            temperature=0.1,
            max_tokens=1200,
        )
        if answer.strip():
            return answer
    raise RuntimeError("DeepSeek returned an empty answer")


def print_plan(plan: QueryPlan | None) -> None:
    if plan is None:
        return
    print("=== Query Plan ===")
    print(f"intent: {plan.intent}")
    print(f"confidence: {plan.confidence}")
    print(f"category_filters: {plan.category_filters or ['none']}")
    print("sub_queries:")
    for idx, sub_query in enumerate(plan.sub_queries, start=1):
        print(f"  {idx}. {sub_query}")
    if plan.warnings:
        print(f"warnings: {plan.warnings}")


def print_sources(retrieved: list[RetrievedChunk]) -> None:
    print("=== Sources ===")
    for idx, item in enumerate(retrieved, start=1):
        metadata = item.metadata
        rerank_info = f" rerank={item.rerank_score:.4f}" if item.rerank_score else ""
        print(
            f"[{idx}] score={item.score:.4f}{rerank_info} distance={item.distance:.4f} "
            f"title={metadata.get('title')} "
            f"section={metadata.get('heading_path')}"
        )
        if item.rerank_reason:
            print(f"    rerank_reason={item.rerank_reason}")
        print(f"    url={metadata.get('url')}")
        print(f"    chunk_id={item.chunk_id}")
        print(f"    retrieval_query={item.source_query}")
        print(f"    category_filter={item.category_filter or 'none'}")


def main() -> None:
    configure_console_output()
    args = parse_args()
    plan, retrieved = retrieve(args)
    context = build_context(retrieved, args.max_context_chars)

    print(f"query: {args.query}")
    print(f"retrieval_mode: {args.retrieval_mode}")
    print(f"rerank_mode: {args.rerank_mode}")
    print(f"llm_provider: {args.llm_provider}")
    print(f"candidate_k: {args.candidate_k}")
    print(f"top_k: {args.top_k}")
    print(f"manual_category_filter: {args.category or 'none'}")
    if args.show_plan:
        print_plan(plan)
    print_sources(retrieved)

    if args.no_generate:
        print("\n=== Context Preview ===")
        print(context[:3000])
        return

    prompt = build_prompt(args.query, context)
    answer = generate_answer(args, prompt)

    if args.repair_answer:
        print("\n=== Draft Answer ===")
        print(answer)
        rule_audit = deterministic_audit(answer, source_count=len(retrieved))
        if not rule_audit["rule_pass"]:
            try:
                answer = repair_answer_with_deepseek(
                    args.query,
                    context,
                    answer,
                    rule_audit,
                    model=args.deepseek_model,
                    base_url=args.deepseek_base_url,
                )
            except Exception as exc:  # noqa: BLE001 - keep the draft visible for debugging.
                print(f"\nrepair_error: {type(exc).__name__}: {exc}")
        print("\n=== Repaired Answer ===")
        print(answer)
    else:
        print("\n=== Answer ===")
        print(answer)

    if args.audit_answer:
        print("\n=== Answer Audit ===")
        rule_audit = deterministic_audit(answer, source_count=len(retrieved))
        llm_audit = None
        audit_error = None
        try:
            llm_audit = audit_answer_with_deepseek(
                args.query,
                context,
                answer,
                model=args.deepseek_audit_model,
                base_url=args.deepseek_base_url,
            )
        except Exception as exc:  # noqa: BLE001 - deterministic audit is still useful.
            audit_error = f"{type(exc).__name__}: {exc}"
        audit = combine_audits(rule_audit, llm_audit)
        if audit_error:
            audit["audit_error"] = audit_error
        print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
