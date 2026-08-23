"""Generate a natural-language routing development set with DeepSeek."""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.deepseek_client import DEFAULT_MODEL, chat_completion  # noqa: E402
from src.ollama_http import embed_texts, unload_embedding_model  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "datasets" / "rag_natural_query_dev_v1_raw.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "eval" / "natural_query_dev_v1" / "generation_summary.json"
DEFAULT_SUMMARY_MD = PROJECT_ROOT / "eval" / "natural_query_dev_v1" / "generation_summary.md"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "eval" / "natural_query_dev_v1" / "generation_checkpoint.jsonl"
DEFAULT_EXISTING_DATASETS = [
    PROJECT_ROOT / "eval" / "datasets" / "rag_system_smoke_eval.jsonl",
    PROJECT_ROOT / "eval" / "datasets" / "rag_routing_holdout_v1.jsonl",
    PROJECT_ROOT / "eval" / "datasets" / "rag_routing_holdout_v2.jsonl",
]

PERSONAS = (
    "刚入门的学习者",
    "正在接入知识库的开发者",
    "排查线上问题的工程师",
    "负责方案选型的技术负责人",
    "维护内部资料的数据工程师",
    "负责效果验收的测试或产品人员",
)

TOPIC_CARDS = {
    "RAG basics": "RAG 的作用、基本流程和常见架构",
    "document loading": "PDF、网页和企业工具数据如何读成标准文档并保留来源",
    "chunking": "结构化切分、递归切分、节点解析和上下文边界",
    "ingestion and indexing": "摄取流水线、缓存、去重、增量更新和索引类型",
    "embedding": "语义向量、BGE-M3、Sentence-BERT 和检索表示",
    "retrieval": "dense、sparse、hybrid、self-query、metadata filter 和 query expansion",
    "reranking": "候选重排、ColBERT、cross-encoder 和 node postprocessor",
    "vector database": "Chroma collection、embedding function、查询、过滤和生命周期",
    "query pipeline": "retriever、router、postprocessor 和 response synthesizer 的协作",
    "evaluation": "召回、排序、faithfulness、context metrics、引用质量和 badcase",
    "RAG failures": "缺内容、切分破坏、漏召回、排序靠后、上下文未利用和长上下文问题",
    "local model": "Ollama 本地模型、生成接口和 embedding 接口",
}

BATCH_SPECS = (
    {
        "name": "direct_everyday",
        "intended_route": "direct",
        "count": 5,
        "focus": "入门时只理解一个概念或某个组件的单一作用",
    },
    {
        "name": "direct_configuration",
        "intended_route": "direct",
        "count": 5,
        "focus": "已经选定工具，只询问一个具体配置、接口或 metadata 用法",
    },
    {
        "name": "direct_data_operation",
        "intended_route": "direct",
        "count": 5,
        "focus": "只询问缓存清理、文档去重、单条更新、collection 管理或索引写入中的一个具体操作",
    },
    {
        "name": "direct_measurement",
        "intended_route": "direct",
        "count": 5,
        "focus": "只询问一个评测指标、一个检索现象或一个质量概念的含义",
    },
    {
        "name": "planned_mixed_scenarios",
        "intended_route": "planned",
        "count": 18,
        "focus": (
            "在以下四类真实任务间尽量均衡且避免模板重复：跨来源入库与引用追溯；"
            "召回、过滤、重排和上下文选择的链路设计；跨阶段线上故障定位；"
            "结合多个评测或引用信号做质量诊断"
        ),
    },
)

EXAM_STYLE_MARKERS = (
    "请根据资料",
    "请结合资料",
    "请分别说明",
    "请列举",
    "请阐述",
    "本题",
    "以上内容",
    "参考答案",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate natural RAG user questions with DeepSeek.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-similarity", type=float, default=0.84)
    parser.add_argument("--max-semantic-similarity", type=float, default=0.93)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--restart", action="store_true", help="Discard a saved generation checkpoint.")
    return parser.parse_args()


def main() -> None:
    configure_console_output()
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"output already exists: {args.output}; pass --force to replace it")
    if not 0 < args.max_similarity < 1:
        raise ValueError("max-similarity must be between 0 and 1")
    if not 0 < args.max_semantic_similarity < 1:
        raise ValueError("max-semantic-similarity must be between 0 and 1")

    if args.restart and args.checkpoint.exists():
        args.checkpoint.unlink()
    prior_questions = load_existing_questions(DEFAULT_EXISTING_DATASETS)
    accepted = load_jsonl(args.checkpoint) if args.checkpoint.exists() else []
    checkpoint_rows_reused = len(accepted)
    validate_checkpoint(accepted)
    semantic_filter = SemanticQuestionFilter(
        prior_questions + [str(row["question"]) for row in accepted],
        model=args.embedding_model,
        host=args.ollama_host,
    )
    rejected_reasons: Counter[str] = Counter()
    batch_attempts: dict[str, int] = {}

    try:
        for spec in BATCH_SPECS:
            batch_rows = [
                row for row in accepted if row.get("generation_batch") == spec["name"]
            ]
            base_rows = [
                row for row in accepted if row.get("generation_batch") != spec["name"]
            ]
            if len(batch_rows) == int(spec["count"]):
                batch_attempts[str(spec["name"])] = 0
                print(f"{spec['name']}: checkpoint reused", flush=True)
                continue
            if batch_rows:
                print(
                    f"{spec['name']}: partial checkpoint {len(batch_rows)}/{spec['count']}",
                    flush=True,
                )
            for attempt in range(1, args.max_attempts + 1):
                remaining = int(spec["count"]) - len(batch_rows)
                if remaining <= 0:
                    break
                raw_items = request_questions(
                    spec,
                    remaining,
                    prior_questions + [str(row["question"]) for row in accepted + batch_rows],
                    model=args.model,
                )
                valid_items = []
                for item in raw_items:
                    issue = validate_generated_item(
                        item,
                        intended_route=str(spec["intended_route"]),
                        seen_questions=prior_questions
                        + [str(row["question"]) for row in accepted + batch_rows],
                        max_similarity=args.max_similarity,
                    )
                    if issue:
                        rejected_reasons[issue] += 1
                        continue
                    valid_items.append(item)

                diverse_items = []
                diverse_vectors: list[list[float]] = []
                if valid_items:
                    vectors = embed_texts(
                        [str(item["question"]) for item in valid_items],
                        args.embedding_model,
                        args.ollama_host,
                    )
                    for item, vector in zip(valid_items, vectors, strict=True):
                        semantic_similarity = semantic_filter.max_similarity(
                            vector,
                            extra_vectors=diverse_vectors,
                        )
                        if semantic_similarity >= args.max_semantic_similarity:
                            rejected_reasons["semantic_near_duplicate"] += 1
                            continue
                        diverse_items.append((item, vector, semantic_similarity))
                        diverse_vectors.append(vector)

                reviews = review_generated_items(
                    [item for item, _, _ in diverse_items],
                    model=args.model,
                )
                for item, vector, semantic_similarity in diverse_items:
                    review = reviews[str(item["candidate_id"])]
                    issue = review_issue(review, intended_route=str(spec["intended_route"]))
                    if issue:
                        rejected_reasons[issue] += 1
                        continue
                    batch_rows.append(
                        normalize_generated_item(
                            item,
                            spec,
                            args.model,
                            review=review,
                            semantic_similarity=semantic_similarity,
                        )
                    )
                    semantic_filter.add(str(item["question"]), vector)
                    if len(batch_rows) == int(spec["count"]):
                        break
                write_jsonl(args.checkpoint, base_rows + batch_rows)
                batch_attempts[str(spec["name"])] = attempt
                print(
                    f"{spec['name']}: attempt {attempt}, accepted {len(batch_rows)}/{spec['count']}",
                    flush=True,
                )
            if len(batch_rows) != int(spec["count"]):
                raise RuntimeError(
                    f"could not generate enough valid rows for {spec['name']}: "
                    f"{len(batch_rows)}/{spec['count']}; rejected={dict(rejected_reasons)}"
                )
            accepted = base_rows + batch_rows
            write_jsonl(args.checkpoint, accepted)
    finally:
        unload_embedding_model(args.embedding_model, args.ollama_host)

    for index, row in enumerate(accepted, start=1):
        row["id"] = f"natural_dev_{index:03d}"

    validate_dataset(accepted)
    write_jsonl(args.output, accepted)
    summary = build_summary(
        accepted,
        prior_questions,
        args,
        batch_attempts=batch_attempts,
        rejected_reasons=rejected_reasons,
        checkpoint_rows_reused=checkpoint_rows_reused,
    )
    write_json(args.summary, summary)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(summary_markdown(summary), encoding="utf-8")
    if args.checkpoint.exists():
        args.checkpoint.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"dataset: {portable_path(args.output)}")


def request_questions(
    spec: dict[str, Any],
    count: int,
    blocked_questions: list[str],
    *,
    model: str,
) -> list[dict[str, Any]]:
    route = str(spec["intended_route"])
    route_definition = (
        "严格只有一个主要信息需求，已经知道要问哪个组件，一组紧密相关的证据就能回答；"
        "不得比较多个方案，不得在多个可能原因之间诊断。"
        if route == "direct"
        else "必须包含至少两个可以独立检索的信息需求，需要跨环节比较、设计或诊断；不是简单列举同类名词。"
    )
    topic_text = "\n".join(f"- {name}: {description}" for name, description in TOPIC_CARDS.items())
    blocked = "\n".join(f"- {question}" for question in blocked_questions[-80:])
    prompt = f"""
你在模拟真实用户使用一个“大模型与 RAG 工程知识库”时会输入的问题。生成 {count} 条中文问题。

本批目标：{route}
目标定义：{route_definition}
场景重点：{spec['focus']}

可以涉及的知识范围：
{topic_text}

可选用户角色（persona 必须原样取一个）：
{json.dumps(PERSONAS, ensure_ascii=False)}

写作要求：
1. 像用户在聊天窗口中真实输入，不要写成课程考试题、论文标题或标准答案提示。
2. 多使用“我现在……”“为什么我……”“这种情况怎么查”“应该选哪个”这样的自然上下文，但每条问题必须能独立理解。
3. 长短要有变化；允许技术名词和产品名，但不要故意堆术语。
4. 不要在问题里出现 direct、planned、路由标签、资料标题、指标答案或应该检索的文档名。
5. direct 的 information_needs 必须严格只有 1 个；planned 必须是 2-4 个可独立检索需求。
6. topics 只能从上面的知识范围名称中选 1-4 个。
7. 不得复述或轻微改写下面已有问题：
{blocked}

direct 正例：“我在 Chroma 里怎么按部门字段过滤结果？”
direct 反例：“知识库有答案却答不出来，是切分、embedding 还是重排的问题？”后者需要分别排查，属于 planned。

只返回 JSON 对象，不要 Markdown：
{{
  "questions": [
    {{
      "question": "用户原话",
      "persona": "可选角色之一",
      "scenario": "一句话说明用户此刻在做什么",
      "topics": ["知识范围名称"],
      "information_needs": ["需要查清的信息，不写答案"]
    }}
  ]
}}
""".strip()
    raw = chat_completion(
        [
            {
                "role": "system",
                "content": "你是企业 AI 产品的真实用户问题设计师，只输出严格 JSON。",
            },
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.85,
        max_tokens=5000,
        response_format={"type": "json_object"},
        thinking=False,
    )
    payload = json.loads(raw)
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError("DeepSeek response must contain a questions array")
    items = [item for item in questions if isinstance(item, dict)]
    for index, item in enumerate(items, start=1):
        item["candidate_id"] = f"candidate_{index}"
    return items


def review_generated_items(items: list[dict[str, Any]], *, model: str) -> dict[str, dict[str, Any]]:
    if not items:
        return {}
    candidates = [
        {
            "candidate_id": item["candidate_id"],
            "question": item["question"],
            "persona": item["persona"],
            "scenario": item["scenario"],
        }
        for item in items
    ]
    prompt = f"""
独立审查下面的用户问题。你看不到生成目标，必须仅根据问题本身判断。

判断标准：
- naturalness 取 1-5；4 或 5 才像聊天窗口中的真实用户，考试腔、堆术语或过度工整应低于 4。
- standalone 表示脱离前文也能理解。
- route 选 direct 或 planned。direct 只有一个主要检索需求，一组紧密证据即可回答；planned 至少有两个需要独立检索再比较、设计或诊断的信息需求。
- information_need_count 是实际可独立检索的信息需求数量，不要被句子长度迷惑。

候选：
{json.dumps(candidates, ensure_ascii=False)}

只返回严格 JSON：
{{"reviews":[{{"candidate_id":"candidate_1","naturalness":4,"standalone":true,"route":"direct","information_need_count":1,"reason":"简短原因"}}]}}
""".strip()
    raw = chat_completion(
        [
            {"role": "system", "content": "你是用户问题质量审稿人，只输出严格 JSON。"},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.1,
        max_tokens=4000,
        response_format={"type": "json_object"},
        thinking=False,
    )
    payload = json.loads(raw)
    rows = payload.get("reviews")
    if not isinstance(rows, list):
        raise ValueError("DeepSeek review response must contain a reviews array")
    reviews: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in reviews:
            raise ValueError(f"duplicate review candidate_id: {candidate_id}")
        reviews[candidate_id] = row
    expected = {str(item["candidate_id"]) for item in items}
    if set(reviews) != expected:
        raise ValueError(f"review candidate ids do not match: expected {expected}, got {set(reviews)}")
    return reviews


def review_issue(review: dict[str, Any], *, intended_route: str) -> str | None:
    try:
        naturalness = int(review.get("naturalness"))
        information_need_count = int(review.get("information_need_count"))
    except (TypeError, ValueError):
        return "invalid_critic_review"
    if naturalness < 4:
        return "critic_low_naturalness"
    if review.get("standalone") is not True:
        return "critic_not_standalone"
    if str(review.get("route") or "") != intended_route:
        return "critic_route_mismatch"
    if intended_route == "direct" and information_need_count != 1:
        return "critic_direct_need_count"
    if intended_route == "planned" and information_need_count < 2:
        return "critic_planned_need_count"
    return None


def validate_generated_item(
    item: dict[str, Any],
    *,
    intended_route: str,
    seen_questions: list[str],
    max_similarity: float,
) -> str | None:
    question = str(item.get("question") or "").strip()
    if len(question) < 8 or len(question) > 180 or "\n" in question:
        return "invalid_question_length"
    if any(marker in question for marker in EXAM_STYLE_MARKERS):
        return "exam_style"
    if re.search(r"\b(?:direct|planned)\b|路由模式", question, flags=re.IGNORECASE):
        return "route_label_leakage"
    persona = str(item.get("persona") or "")
    if persona not in PERSONAS:
        return "unknown_persona"
    if not str(item.get("scenario") or "").strip():
        return "missing_scenario"
    topics = item.get("topics")
    if not isinstance(topics, list) or not 1 <= len(topics) <= 4:
        return "invalid_topics"
    if any(str(topic) not in TOPIC_CARDS for topic in topics):
        return "unknown_topic"
    needs = item.get("information_needs")
    if not isinstance(needs, list) or any(not str(need).strip() for need in needs):
        return "invalid_information_needs"
    if intended_route == "direct" and len(needs) != 1:
        return "direct_need_count"
    if intended_route == "planned" and not 2 <= len(needs) <= 4:
        return "planned_need_count"
    if max_question_similarity(question, seen_questions) >= max_similarity:
        return "near_duplicate"
    return None


def normalize_generated_item(
    item: dict[str, Any],
    spec: dict[str, Any],
    model: str,
    *,
    review: dict[str, Any],
    semantic_similarity: float,
) -> dict[str, Any]:
    return {
        "question": str(item["question"]).strip(),
        "intended_route": str(spec["intended_route"]),
        "stratum": "focused" if spec["intended_route"] == "direct" else "compound",
        "persona": str(item["persona"]),
        "scenario": str(item["scenario"]).strip(),
        "topics": list(dict.fromkeys(str(topic) for topic in item["topics"])),
        "information_needs": [str(need).strip() for need in item["information_needs"]],
        "generation_batch": str(spec["name"]),
        "generator": f"llm:{model}",
        "critic": {
            "model": model,
            "naturalness": int(review["naturalness"]),
            "route": str(review["route"]),
            "information_need_count": int(review["information_need_count"]),
            "reason": str(review.get("reason") or ""),
        },
        "semantic_similarity_to_seen": round(semantic_similarity, 4),
    }


def validate_dataset(rows: list[dict[str, Any]]) -> None:
    expected_total = sum(int(spec["count"]) for spec in BATCH_SPECS)
    if len(rows) != expected_total:
        raise ValueError("generated dataset has an unexpected row count")
    questions = [normalize_question(str(row["question"])) for row in rows]
    if len(questions) != len(set(questions)):
        raise ValueError("generated dataset contains duplicate questions")
    counts = Counter(str(row["intended_route"]) for row in rows)
    expected_counts = Counter()
    for spec in BATCH_SPECS:
        expected_counts[str(spec["intended_route"])] += int(spec["count"])
    if counts != expected_counts:
        raise ValueError(f"unexpected intended route balance: {counts}")


def validate_checkpoint(rows: list[dict[str, Any]]) -> None:
    specs = {str(spec["name"]): spec for spec in BATCH_SPECS}
    counts = Counter(str(row.get("generation_batch") or "") for row in rows)
    unknown = [name for name in counts if name not in specs]
    if unknown:
        raise ValueError(f"checkpoint contains unknown generation batches: {unknown}; pass --restart")
    oversized = [
        name for name, count in counts.items() if count > int(specs[name]["count"])
    ]
    if oversized:
        raise ValueError(f"checkpoint contains oversized batches: {oversized}; pass --restart")


def max_question_similarity(question: str, others: list[str]) -> float:
    normalized = normalize_question(question)
    if not normalized or not others:
        return 0.0
    return max(
        difflib.SequenceMatcher(None, normalized, normalize_question(other)).ratio()
        for other in others
    )


class SemanticQuestionFilter:
    def __init__(self, questions: list[str], *, model: str, host: str) -> None:
        self.questions = list(questions)
        self.vectors = embed_texts(questions, model, host) if questions else []

    def max_similarity(
        self,
        vector: list[float],
        *,
        extra_vectors: list[list[float]] | None = None,
    ) -> float:
        candidates = self.vectors + list(extra_vectors or [])
        return max((cosine_similarity(vector, candidate) for candidate in candidates), default=0.0)

    def add(self, question: str, vector: list[float]) -> None:
        self.questions.append(question)
        self.vectors.append(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding vectors must be non-empty and have equal length")
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator


def normalize_question(question: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", question.casefold(), flags=re.UNICODE)


def load_existing_questions(paths: list[Path]) -> list[str]:
    questions = []
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            question = str(row.get("question") or "").strip()
            if question:
                questions.append(question)
    return questions


def build_summary(
    rows: list[dict[str, Any]],
    prior_questions: list[str],
    args: argparse.Namespace,
    *,
    batch_attempts: dict[str, int],
    rejected_reasons: Counter[str],
    checkpoint_rows_reused: int,
) -> dict[str, Any]:
    lengths = [len(str(row["question"])) for row in rows]
    first_person_markers = ("我", "我们", "现在", "这种情况", "线上", "项目")
    max_prior_similarity = max(
        max_question_similarity(str(row["question"]), prior_questions) for row in rows
    )
    return {
        "model": args.model,
        "dataset": portable_path(args.output),
        "count": len(rows),
        "intended_route_counts": dict(Counter(str(row["intended_route"]) for row in rows)),
        "stratum_counts": dict(Counter(str(row["stratum"]) for row in rows)),
        "persona_counts": dict(Counter(str(row["persona"]) for row in rows)),
        "topic_counts": dict(Counter(topic for row in rows for topic in row["topics"])),
        "question_length": {
            "min": min(lengths),
            "max": max(lengths),
            "median": statistics.median(lengths),
        },
        "questions_with_context_marker": sum(
            any(marker in str(row["question"]) for marker in first_person_markers) for row in rows
        ),
        "max_similarity_to_prior_datasets": round(max_prior_similarity, 4),
        "similarity_rejection_threshold": args.max_similarity,
        "semantic_similarity_rejection_threshold": args.max_semantic_similarity,
        "max_accepted_semantic_similarity_to_seen": max(
            float(row["semantic_similarity_to_seen"]) for row in rows
        ),
        "batch_attempts": batch_attempts,
        "checkpoint_rows_reused": checkpoint_rows_reused,
        "rejected_reasons": dict(rejected_reasons),
        "boundary": (
            "DeepSeek generated these questions from scenario and topic cards. Intended route is a "
            "development label for route diagnostics, not retrieval-quality ground truth or a holdout result."
        ),
    }


def summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Natural User Query Development Set V1",
        "",
        f"- questions: {summary['count']}",
        f"- generator: {summary['model']}",
        f"- intended routes: {summary['intended_route_counts']}",
        f"- strata: {summary['stratum_counts']}",
        f"- length min/median/max: {summary['question_length']['min']} / "
        f"{summary['question_length']['median']} / {summary['question_length']['max']}",
        f"- questions with conversational context marker: "
        f"{summary['questions_with_context_marker']}/{summary['count']}",
        f"- maximum similarity to prior datasets: {summary['max_similarity_to_prior_datasets']:.3f}",
        f"- maximum accepted semantic similarity to prior/earlier questions: "
        f"{summary['max_accepted_semantic_similarity_to_seen']:.3f}",
        f"- checkpoint rows reused in final invocation: {summary['checkpoint_rows_reused']}",
        "",
        "## Personas",
        "",
        "| persona | count |",
        "|---|---:|",
    ]
    for persona, count in summary["persona_counts"].items():
        lines.append(f"| {persona} | {count} |")
    lines.extend(["", "## Boundary", "", summary["boundary"], ""])
    return "\n".join(lines)


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
