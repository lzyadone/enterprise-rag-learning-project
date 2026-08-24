"""Generate a source-grounded, natural-language retrieval holdout with DeepSeek."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deepseek_client import DEFAULT_MODEL, chat_completion  # noqa: E402
from src.ollama_http import embed_texts, unload_embedding_model  # noqa: E402


DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "llm_rag_docs" / "chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "datasets" / "rag_natural_query_holdout_v3.jsonl"
DEFAULT_ANCHORS = (
    PROJECT_ROOT
    / "eval"
    / "benchmarks"
    / "rag_natural_query_holdout_v3"
    / "source_anchors.jsonl"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "eval"
    / "benchmarks"
    / "rag_natural_query_holdout_v3"
    / "generation_summary.json"
)
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "runtime" / "holdout_v3_generation.jsonl"

PERSONAS = (
    "刚入门的学习者",
    "正在接入知识库的开发者",
    "排查线上问题的工程师",
    "负责方案选型的技术负责人",
    "维护内部资料的数据工程师",
    "负责效果验收的测试或产品人员",
)


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    stratum: str
    categories: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceBundle:
    target: TargetSpec
    chunks: tuple[dict[str, Any], ...]


TARGET_SPECS = (
    TargetSpec("h3_001", "focused", ("RAG overview",), ("langchain_retrieval::chunk_0005",)),
    TargetSpec("h3_002", "focused", ("document loading",), ("langchain_document_loaders::chunk_0000",)),
    TargetSpec("h3_003", "focused", ("chunking",), ("langchain_recursive_splitter::chunk_0001",)),
    TargetSpec("h3_004", "focused", ("ingestion",), ("llamaindex_ingestion_pipeline::chunk_0000",)),
    TargetSpec("h3_005", "focused", ("indexing",), ("llamaindex_indexing::chunk_0004",)),
    TargetSpec("h3_006", "focused", ("embedding",), ("bge_m3_model_card::chunk_0004",)),
    TargetSpec("h3_007", "focused", ("retrieval",), ("langchain_self_query::chunk_0000",)),
    TargetSpec("h3_008", "focused", ("retrieval",), ("llamaindex_advanced_retrieval::chunk_0000",)),
    TargetSpec("h3_009", "focused", ("reranking",), ("cohere_rerank_docs::chunk_0000",)),
    TargetSpec("h3_010", "focused", ("vector db",), ("chroma_metadata_filtering::chunk_0000",)),
    TargetSpec("h3_011", "focused", ("vector db",), ("chroma_collections::chunk_0002",)),
    TargetSpec("h3_012", "focused", ("querying",), ("llamaindex_querying::chunk_0000",)),
    TargetSpec("h3_013", "focused", ("evaluation",), ("ragas_faithfulness::chunk_0000",)),
    TargetSpec("h3_014", "focused", ("evaluation",), ("phoenix_document_relevance::chunk_0000",)),
    TargetSpec("h3_015", "focused", ("local model",), ("ollama_api::chunk_0071",)),
    TargetSpec("h3_016", "focused", ("RAG challenges",), ("lost_in_the_middle::chunk_0026",)),
    TargetSpec(
        "h3_017",
        "compound",
        ("document loading", "chunking"),
        ("langchain_document_loaders::chunk_0001", "langchain_text_splitters::chunk_0000"),
    ),
    TargetSpec(
        "h3_018",
        "compound",
        ("ingestion", "vector db"),
        ("llamaindex_ingestion_pipeline::chunk_0002", "chroma_collections::chunk_0006"),
    ),
    TargetSpec(
        "h3_019",
        "compound",
        ("ingestion", "indexing"),
        ("llamaindex_ingestion_pipeline::chunk_0003", "llamaindex_indexing::chunk_0001"),
    ),
    TargetSpec(
        "h3_020",
        "compound",
        ("embedding", "retrieval"),
        ("bge_m3_paper::chunk_0009", "langchain_self_query::chunk_0016"),
    ),
    TargetSpec(
        "h3_021",
        "compound",
        ("retrieval", "reranking"),
        ("langchain_retrievers::chunk_0000", "cohere_rerank_docs::chunk_0005"),
    ),
    TargetSpec(
        "h3_022",
        "compound",
        ("retrieval", "vector db"),
        ("langchain_self_query::chunk_0017", "chroma_metadata_filtering::chunk_0001"),
    ),
    TargetSpec(
        "h3_023",
        "compound",
        ("querying", "evaluation"),
        ("llamaindex_querying::chunk_0001", "llamaindex_evaluating::chunk_0001"),
    ),
    TargetSpec(
        "h3_024",
        "compound",
        ("evaluation", "retrieval"),
        ("ragas_context_recall::chunk_0000", "langchain_retrievers::chunk_0002"),
    ),
    TargetSpec(
        "h3_025",
        "compound",
        ("evaluation", "RAG challenges"),
        ("ragas_context_precision::chunk_0000", "rag_failure_points::chunk_0008"),
    ),
    TargetSpec(
        "h3_026",
        "compound",
        ("RAG overview", "retrieval"),
        ("langchain_retrieval::chunk_0010", "llamaindex_advanced_retrieval::chunk_0002"),
    ),
    TargetSpec(
        "h3_027",
        "compound",
        ("chunking", "RAG challenges"),
        ("llamaindex_node_parsers::chunk_0012", "lost_in_the_middle::chunk_0015"),
    ),
    TargetSpec(
        "h3_028",
        "compound",
        ("local model", "embedding"),
        ("ollama_api::chunk_0075", "bge_m3_paper::chunk_0001"),
    ),
    TargetSpec(
        "h3_029",
        "compound",
        ("evaluation", "reranking"),
        ("phoenix_document_relevance::chunk_0001", "colbert_github::chunk_0000"),
    ),
    TargetSpec(
        "h3_030",
        "compound",
        ("RAG overview", "RAG challenges"),
        ("langchain_retrieval::chunk_0002", "rag_failure_points::chunk_0016"),
    ),
    TargetSpec(
        "h3_031",
        "compound",
        ("chunking", "vector db"),
        ("langchain_text_splitters::chunk_0003", "chroma_metadata_filtering::chunk_0002"),
    ),
    TargetSpec(
        "h3_032",
        "compound",
        ("retrieval", "evaluation"),
        ("langchain_self_query::chunk_0013", "ragas_context_precision::chunk_0002"),
    ),
)

EXAM_STYLE_MARKERS = (
    "请根据资料",
    "请结合资料",
    "请分别说明",
    "请列举",
    "请阐述",
    "本题",
    "参考答案",
)
MAX_GENERATED_INFORMATION_NEEDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the source-grounded planner v3 holdout.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-lexical-similarity", type=float, default=0.82)
    parser.add_argument("--max-semantic-similarity", type=float, default=0.91)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    validate_args(args)
    if args.output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {args.output}; pass --force to replace it")
    if args.restart and args.checkpoint.exists():
        args.checkpoint.unlink()

    chunks = load_jsonl(args.chunks)
    bundles = select_evidence_bundles(chunks, TARGET_SPECS, seed=args.seed)
    prior_questions = load_prior_questions(args.output)
    checkpoint_rows = load_jsonl(args.checkpoint) if args.checkpoint.exists() else []
    accepted = {str(row["target_id"]): row for row in checkpoint_rows}
    validate_checkpoint(accepted, bundles)
    rejected: Counter[str] = Counter()

    semantic_filter = SemanticQuestionFilter(
        prior_questions + [str(row["question"]) for row in accepted.values()],
        model=args.embedding_model,
        host=args.ollama_host,
    )
    bundle_by_id = {bundle.target.target_id: bundle for bundle in bundles}
    try:
        for attempt in range(1, args.max_attempts + 1):
            pending = [bundle for bundle in bundles if bundle.target.target_id not in accepted]
            if not pending:
                break
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                generated = request_questions(batch, model=args.model)
                valid: list[tuple[EvidenceBundle, dict[str, Any]]] = []
                for bundle in batch:
                    item = generated.get(bundle.target.target_id)
                    issue = validate_generated_item(
                        item,
                        bundle.target,
                        prior_questions + [str(row["question"]) for row in accepted.values()],
                        args.max_lexical_similarity,
                    )
                    if issue:
                        rejected[issue] += 1
                        continue
                    valid.append((bundle, item))

                reviews = review_questions(valid, model=args.model)
                review_valid = []
                for bundle, item in valid:
                    review = reviews.get(bundle.target.target_id)
                    issue = review_issue(review, bundle.target)
                    if issue:
                        rejected[issue] += 1
                        continue
                    review_valid.append((bundle, item, review))

                if review_valid:
                    vectors = embed_texts(
                        [str(item["question"]) for _, item, _ in review_valid],
                        args.embedding_model,
                        args.ollama_host,
                    )
                    for (bundle, item, review), vector in zip(review_valid, vectors, strict=True):
                        similarity = semantic_filter.max_similarity(vector)
                        if similarity >= args.max_semantic_similarity:
                            rejected["semantic_near_duplicate"] += 1
                            continue
                        row = normalize_row(item, bundle, review, similarity, args.model)
                        accepted[bundle.target.target_id] = row
                        semantic_filter.add(str(row["question"]), vector)
                write_checkpoint(args.checkpoint, accepted, bundles)
                print(
                    f"attempt {attempt}: accepted {len(accepted)}/{len(bundles)}",
                    flush=True,
                )
        if len(accepted) != len(bundles):
            missing = [bundle.target.target_id for bundle in bundles if bundle.target.target_id not in accepted]
            raise RuntimeError(f"holdout generation incomplete; missing={missing}; rejected={dict(rejected)}")
    finally:
        unload_embedding_model(args.embedding_model, args.ollama_host)

    ordered = [accepted[bundle.target.target_id] for bundle in bundles]
    dataset_rows = [dataset_row(row, index) for index, row in enumerate(ordered, start=1)]
    anchor_rows = [anchor_row(row, bundle_by_id[str(row["target_id"])], index) for index, row in enumerate(ordered, start=1)]
    write_jsonl(args.output, dataset_rows)
    write_jsonl(args.anchors, anchor_rows)
    summary = build_summary(dataset_rows, anchor_rows, prior_questions, rejected, args)
    write_json(args.summary, summary)
    args.checkpoint.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"dataset: {portable_path(args.output)}")


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    for value, name in (
        (args.max_lexical_similarity, "max-lexical-similarity"),
        (args.max_semantic_similarity, "max-semantic-similarity"),
    ):
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1")


def select_evidence_bundles(
    chunks: list[dict[str, Any]],
    specs: tuple[TargetSpec, ...],
    *,
    seed: int,
) -> list[EvidenceBundle]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    for chunk in chunks:
        if is_usable_evidence(chunk):
            by_category.setdefault(str(chunk["category"]), []).append(chunk)

    used_ids: set[str] = set()
    bundles = []
    for spec in specs:
        if spec.evidence_ids:
            if len(spec.evidence_ids) != len(spec.categories):
                raise ValueError(f"evidence/category count mismatch for {spec.target_id}")
            selected = []
            for category, chunk_id in zip(spec.categories, spec.evidence_ids, strict=True):
                chunk = by_id.get(chunk_id)
                if chunk is None:
                    raise ValueError(f"unknown evidence chunk for {spec.target_id}: {chunk_id}")
                if str(chunk.get("category")) != category:
                    raise ValueError(
                        f"evidence category mismatch for {spec.target_id}: "
                        f"expected {category}, got {chunk.get('category')}"
                    )
                if not is_curated_evidence(chunk):
                    raise ValueError(f"low-quality evidence chunk for {spec.target_id}: {chunk_id}")
                selected.append(chunk)
            bundles.append(EvidenceBundle(spec, tuple(selected)))
            continue

        selected = []
        used_sources: set[str] = set()
        for category in spec.categories:
            candidates = by_category.get(category, [])
            if not candidates:
                raise ValueError(f"no usable evidence chunks for category: {category}")
            ordered = sorted(
                candidates,
                key=lambda row: stable_rank(seed, spec.target_id, str(row["chunk_id"])),
            )
            choice = next(
                (
                    row
                    for row in ordered
                    if str(row["chunk_id"]) not in used_ids
                    and str(row.get("source_id") or row.get("doc_id")) not in used_sources
                ),
                next((row for row in ordered if str(row["chunk_id"]) not in used_ids), ordered[0]),
            )
            selected.append(choice)
            used_ids.add(str(choice["chunk_id"]))
            used_sources.add(str(choice.get("source_id") or choice.get("doc_id")))
        bundles.append(EvidenceBundle(spec, tuple(selected)))
    return bundles


def is_usable_evidence(chunk: dict[str, Any]) -> bool:
    text = str(chunk.get("text") or "")
    heading = str(chunk.get("heading_path") or "").casefold()
    char_count = int(chunk.get("char_count") or len(text))
    if not 450 <= char_count <= 2600:
        return False
    if any(marker in heading for marker in ("references", "bibliography", "installation", "news:")):
        return False
    if text.count("```") >= 4:
        return False
    alphanumeric = sum(character.isalnum() for character in text)
    return alphanumeric / max(1, len(text)) >= 0.45


def is_curated_evidence(chunk: dict[str, Any]) -> bool:
    text = str(chunk.get("text") or "")
    heading = str(chunk.get("heading_path") or "").casefold()
    return len(text) >= 300 and not any(
        marker in heading for marker in ("references", "bibliography", "instructions for reporting")
    )


def stable_rank(seed: int, target_id: str, chunk_id: str) -> str:
    return hashlib.sha256(f"{seed}:{target_id}:{chunk_id}".encode("utf-8")).hexdigest()


def request_questions(
    bundles: list[EvidenceBundle],
    *,
    model: str,
) -> dict[str, dict[str, Any]]:
    cards = [prompt_bundle(bundle) for bundle in bundles]
    prompt = f"""
下面是从“大模型与 RAG 工程知识库”真实资料中抽取的证据卡。请为每张卡模拟一位真实用户在聊天窗口里会提出的问题。

要求：
1. 问题必须完全能由对应证据卡回答，不得引入卡片之外的事实、版本或产品行为。
2. focused 卡只问一个主要信息需求；compound 卡必须自然地包含 2-3 个需要结合证据的信息需求。
3. 使用真实场景和不完全专业的表达，允许“我正在……”“线上遇到……”；不要写成考试题或知识点标题。
4. 不得出现“根据资料”“证据卡”“文档中说”，也不要直接复制完整句子。
5. 不得输出答案。persona 必须从给定列表原样选择。

persona 列表：{json.dumps(PERSONAS, ensure_ascii=False)}

证据卡：
{json.dumps(cards, ensure_ascii=False)}

只返回严格 JSON：
{{"questions":[{{"target_id":"h3_001","question":"用户原话","persona":"persona 列表中的值","scenario":"一句话场景","information_needs":["需要查清的信息"]}}]}}
""".strip()
    raw = chat_completion(
        [
            {"role": "system", "content": "你是企业 AI 产品的真实用户问题设计师，只输出严格 JSON。"},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.82,
        max_tokens=5000,
        response_format={"type": "json_object"},
        thinking=False,
    )
    payload = json.loads(raw)
    rows = payload.get("questions")
    if not isinstance(rows, list):
        raise ValueError("DeepSeek response must contain a questions array")
    result = {}
    for row in rows:
        if isinstance(row, dict) and row.get("target_id"):
            result[str(row["target_id"])] = row
    return result


def review_questions(
    candidates: list[tuple[EvidenceBundle, dict[str, Any]]],
    *,
    model: str,
) -> dict[str, dict[str, Any]]:
    if not candidates:
        return {}
    review_cards = [
        {
            "target_id": bundle.target.target_id,
            "question": item["question"],
            "evidence": [evidence_prompt(chunk) for chunk in bundle.chunks],
        }
        for bundle, item in candidates
    ]
    prompt = f"""
独立审查下面的问题和证据。你不知道生成目标，请只根据问题本身判断。

- naturalness 取 1-5，4 或 5 才像真实聊天输入。
- standalone 表示脱离前文仍可理解。
- observed_stratum：只有一个主要信息需求为 focused；存在 2-3 个需结合的信息需求为 compound。
- answerable 表示所有关键要求都能由给定证据回答。
- evidence_coverage 取 0-1，表示证据对问题要求的覆盖比例。
- unsupported_assumptions 表示回答是否必须补充证据外事实。

候选：
{json.dumps(review_cards, ensure_ascii=False)}

只返回严格 JSON：
{{"reviews":[{{"target_id":"h3_001","naturalness":4,"standalone":true,"observed_stratum":"focused","information_need_count":1,"answerable":true,"evidence_coverage":1.0,"unsupported_assumptions":false,"reason":"简短原因"}}]}}
""".strip()
    raw = chat_completion(
        [
            {"role": "system", "content": "你是检索评测集质量审稿人，只输出严格 JSON。"},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.1,
        max_tokens=4500,
        response_format={"type": "json_object"},
        thinking=False,
    )
    payload = json.loads(raw)
    rows = payload.get("reviews")
    if not isinstance(rows, list):
        raise ValueError("DeepSeek response must contain a reviews array")
    return {
        str(row["target_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("target_id")
    }


def prompt_bundle(bundle: EvidenceBundle) -> dict[str, Any]:
    return {
        "target_id": bundle.target.target_id,
        "shape": bundle.target.stratum,
        "evidence": [evidence_prompt(chunk) for chunk in bundle.chunks],
    }


def evidence_prompt(chunk: dict[str, Any]) -> dict[str, str]:
    return {
        "category": str(chunk["category"]),
        "title": str(chunk["title"]),
        "section": str(chunk.get("heading_path") or ""),
        "excerpt": str(chunk["text"])[:2200],
    }


def validate_generated_item(
    item: dict[str, Any] | None,
    target: TargetSpec,
    seen_questions: list[str],
    max_similarity: float,
) -> str | None:
    if not isinstance(item, dict):
        return "missing_generation"
    question = str(item.get("question") or "").strip()
    if len(question) < 10 or len(question) > 180 or "\n" in question:
        return "invalid_question_length"
    if any(marker in question for marker in EXAM_STYLE_MARKERS):
        return "exam_style"
    if re.search(r"证据卡|参考答案|target_id|focused|compound", question, flags=re.IGNORECASE):
        return "generation_label_leakage"
    if str(item.get("persona") or "") not in PERSONAS:
        return "unknown_persona"
    if not str(item.get("scenario") or "").strip():
        return "missing_scenario"
    needs = item.get("information_needs")
    if (
        not isinstance(needs, list)
        or not needs
        or len(needs) > MAX_GENERATED_INFORMATION_NEEDS
        or any(not str(need).strip() for need in needs)
    ):
        return "invalid_information_needs"
    if max_question_similarity(question, seen_questions) >= max_similarity:
        return "lexical_near_duplicate"
    return None


def review_issue(review: dict[str, Any] | None, target: TargetSpec) -> str | None:
    if not isinstance(review, dict):
        return "missing_review"
    try:
        naturalness = int(review.get("naturalness"))
        need_count = int(review.get("information_need_count"))
        coverage = float(review.get("evidence_coverage"))
    except (TypeError, ValueError):
        return "invalid_review"
    if naturalness < 4:
        return "critic_low_naturalness"
    if review.get("standalone") is not True:
        return "critic_not_standalone"
    if str(review.get("observed_stratum") or "") != target.stratum:
        return "critic_stratum_mismatch"
    if target.stratum == "focused" and need_count != 1:
        return "critic_focused_need_count"
    if target.stratum == "compound" and not 2 <= need_count <= 3:
        return "critic_compound_need_count"
    if review.get("answerable") is not True or coverage < 0.9:
        return "critic_not_fully_answerable"
    if review.get("unsupported_assumptions") is not False:
        return "critic_unsupported_assumptions"
    return None


def normalize_row(
    item: dict[str, Any],
    bundle: EvidenceBundle,
    review: dict[str, Any],
    semantic_similarity: float,
    model: str,
) -> dict[str, Any]:
    return {
        "target_id": bundle.target.target_id,
        "question": str(item["question"]).strip(),
        "stratum": bundle.target.stratum,
        "persona": str(item["persona"]),
        "scenario": str(item["scenario"]).strip(),
        "topics": list(bundle.target.categories),
        "information_needs": [str(value).strip() for value in item["information_needs"]],
        "generator": f"llm:{model}",
        "critic": {
            "model": model,
            "naturalness": int(review["naturalness"]),
            "observed_stratum": str(review["observed_stratum"]),
            "information_need_count": int(review["information_need_count"]),
            "answerable": bool(review["answerable"]),
            "evidence_coverage": float(review["evidence_coverage"]),
            "reason": str(review.get("reason") or ""),
        },
        "semantic_similarity_to_seen": round(semantic_similarity, 4),
    }


def dataset_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    result = dict(row)
    result.pop("target_id", None)
    return {"id": f"natural_holdout3_{index:03d}", **result}


def anchor_row(row: dict[str, Any], bundle: EvidenceBundle, index: int) -> dict[str, Any]:
    return {
        "id": f"natural_holdout3_{index:03d}",
        "question": row["question"],
        "generation_target": bundle.target.stratum,
        "source_anchors_are_not_qrels": True,
        "evidence": [
            {
                "chunk_id": str(chunk["chunk_id"]),
                "source_id": str(chunk.get("source_id") or chunk.get("doc_id")),
                "title": str(chunk["title"]),
                "category": str(chunk["category"]),
                "heading_path": str(chunk.get("heading_path") or ""),
                "url": str(chunk.get("url") or ""),
                "excerpt": str(chunk["text"])[:2200],
            }
            for chunk in bundle.chunks
        ],
    }


def build_summary(
    rows: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    prior_questions: list[str],
    rejected: Counter[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "role": "holdout",
        "frozen_before_retrieval": True,
        "generator": args.model,
        "critic": args.model,
        "embedding_model": args.embedding_model,
        "seed": args.seed,
        "questions": len(rows),
        "strata": dict(Counter(str(row["stratum"]) for row in rows)),
        "topics": dict(Counter(topic for row in rows for topic in row["topics"])),
        "source_anchor_chunks": sum(len(row["evidence"]) for row in anchors),
        "unique_source_anchor_chunks": len(
            {item["chunk_id"] for row in anchors for item in row["evidence"]}
        ),
        "max_lexical_similarity_to_prior": round(
            max(max_question_similarity(str(row["question"]), prior_questions) for row in rows),
            4,
        ),
        "max_semantic_similarity_to_seen": max(
            float(row["semantic_similarity_to_seen"]) for row in rows
        ),
        "rejected": dict(rejected),
        "dataset_sha256": sha256_file(args.output),
        "source_anchors_sha256": sha256_file(args.anchors),
        "boundary": (
            "Questions were generated from frozen source excerpts before retrieval. Source anchors "
            "prove intended answerability but are not relevance judgments and must not be supplied "
            "to candidate generators or the qrels judge."
        ),
    }


def validate_checkpoint(
    accepted: dict[str, dict[str, Any]], bundles: list[EvidenceBundle]
) -> None:
    expected = {bundle.target.target_id for bundle in bundles}
    unknown = set(accepted) - expected
    if unknown:
        raise ValueError(f"checkpoint contains unknown target ids: {sorted(unknown)}")


def write_checkpoint(
    path: Path,
    accepted: dict[str, dict[str, Any]],
    bundles: list[EvidenceBundle],
) -> None:
    write_jsonl(
        path,
        [accepted[bundle.target.target_id] for bundle in bundles if bundle.target.target_id in accepted],
    )


class SemanticQuestionFilter:
    def __init__(self, questions: list[str], *, model: str, host: str) -> None:
        self.vectors = embed_texts(questions, model, host) if questions else []

    def max_similarity(self, vector: list[float]) -> float:
        return max((cosine_similarity(vector, candidate) for candidate in self.vectors), default=0.0)

    def add(self, question: str, vector: list[float]) -> None:
        del question
        self.vectors.append(vector)


def max_question_similarity(question: str, others: list[str]) -> float:
    normalized = normalize_question(question)
    if not normalized or not others:
        return 0.0
    return max(
        difflib.SequenceMatcher(None, normalized, normalize_question(other)).ratio()
        for other in others
    )


def normalize_question(question: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", question.casefold(), flags=re.UNICODE)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must be non-empty and have equal length")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def load_prior_questions(output: Path) -> list[str]:
    questions = []
    for path in sorted((PROJECT_ROOT / "eval" / "datasets").glob("*.jsonl")):
        if path.resolve() == output.resolve():
            continue
        for row in load_jsonl(path):
            question = str(row.get("question") or "").strip()
            if question:
                questions.append(question)
    return list(dict.fromkeys(questions))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
