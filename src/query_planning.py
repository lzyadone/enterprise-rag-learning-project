"""Schema-driven query planning for the LLM/RAG knowledge base.

This is a lightweight fallback planner. It does not try to "solve" the user's
question. It only maps the question to knowledge-base categories and creates
retrieval-oriented sub-queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


CATEGORY_HINTS: dict[str, list[str]] = {
    "RAG overview": [
        "rag",
        "retrieval augmented generation",
        "流程",
        "阶段",
        "架构",
        "2-step",
        "agentic",
        "hybrid",
        "知识库",
    ],
    "RAG challenges": [
        "瓶颈",
        "难点",
        "挑战",
        "失败点",
        "局限",
        "限制",
        "failure point",
        "failure points",
        "limitation",
        "challenge",
        "lost in the middle",
    ],
    "document loading": ["loader", "load", "加载", "读取", "解析", "document loader", "pdf", "网页"],
    "chunking": ["chunk", "split", "splitter", "切分", "分块", "固定窗口", "递归切分", "markdown", "标题"],
    "ingestion": ["ingestion", "pipeline", "入库", "摄取", "处理流水线", "transform", "node"],
    "indexing": ["index", "indexing", "索引", "建索引", "index structure"],
    "querying": [
        "query engine",
        "querying",
        "查询",
        "响应合成",
        "response synthesizer",
        "query rewrite",
        "query expansion",
        "query transformation",
        "查询改写",
        "查询扩展",
        "问题改写",
        "子问题",
        "多路查询",
    ],
    "vector db": ["chroma", "vector", "向量库", "metadata", "collection", "filter", "过滤", "where"],
    "local model": ["ollama", "本地模型", "api", "generate", "chat", "qwen", "部署本地"],
    "embedding": ["embedding", "embed", "bge", "bge-m3", "sentence-bert", "向量化", "语义向量", "相似度"],
    "retrieval": [
        "retriever",
        "retrieve",
        "检索",
        "召回",
        "self-query",
        "metadata filter",
        "过滤检索",
        "hybrid search",
        "hybrid retrieval",
        "混合检索",
        "向量相似度",
        "vector search",
        "semantic search",
        "query rewrite",
        "query expansion",
    ],
    "reranking": ["rerank", "reranker", "重排", "colbert", "late interaction", "候选排序"],
    "evaluation": [
        "eval",
        "evaluation",
        "评估",
        "忠实",
        "faithfulness",
        "groundedness",
        "correctness",
        "relevance",
        "命中率",
        "badcase",
    ],
    "RAG paper": ["论文", "paper", "原始论文", "rag paper", "fever", "knowledge-intensive"],
}


CATEGORY_EXPANSIONS: dict[str, str] = {
    "RAG overview": "RAG stages architecture retrieval augmented generation loading indexing querying evaluation",
    "RAG challenges": "RAG failure points limitations challenges Missing Content Missed the Top Ranked Documents Not in Context Not Extracted Incorrect Specificity Incomplete hallucination retrieval errors lost in the middle robustness validation",
    "document loading": "document loaders readers parse files pdf html markdown data sources",
    "chunking": "text splitters chunks recursive splitter markdown structure-based splitting chunk size overlap",
    "ingestion": "ingestion pipeline transformations metadata nodes embeddings vector store",
    "indexing": "indexing vector index nodes embeddings store metadata",
    "querying": "query engine retriever response synthesizer sub query multi step hybrid query query rewrite query expansion query transformation query routing multi query sub-question",
    "vector db": "Chroma vector database collection metadata filter where embeddings document storage",
    "local model": "Ollama API generate chat embed local model Python client",
    "embedding": "embedding model BGE-M3 Sentence-BERT vector representations semantic retrieval",
    "retrieval": "retriever self query metadata filtering vector search hybrid search sparse dense retrieval strategy query rewrite query expansion",
    "reranking": "reranking reranker ColBERT late interaction candidate documents ranking",
    "evaluation": "RAG evaluation correctness relevance groundedness faithfulness retrieval relevance LLM as judge",
    "RAG paper": "Retrieval-Augmented Generation knowledge intensive NLP tasks RAG paper",
}


@dataclass
class QueryAspect:
    name: str
    question: str
    search_query: str = ""
    categories: list[str] = field(default_factory=list)
    required: bool = True

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "question": self.question,
            "search_query": self.search_query,
            "categories": self.categories,
            "required": self.required,
        }


@dataclass
class QueryPlan:
    original_query: str
    rewritten_query: str
    intent: str
    category_filters: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    aspects: list[QueryAspect] = field(default_factory=list)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    planner_version: str = "rules_v2"

    def as_dict(self) -> dict[str, object]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "intent": self.intent,
            "category_filters": self.category_filters,
            "sub_queries": self.sub_queries,
            "aspects": [aspect.as_dict() for aspect in self.aspects],
            "confidence": self.confidence,
            "warnings": self.warnings,
            "planner_version": self.planner_version,
        }


ASPECT_DEFINITIONS: list[dict[str, object]] = [
    {
        "name": "classification",
        "triggers": ["分类", "类别", "类型", "架构", "模式", "路线", "category", "categories", "type", "architecture"],
        "question": "RAG 系统有哪些常见架构、类型或范式？请只总结检索资料能够直接支持的分类。",
        "search_query": "RAG architecture taxonomy paradigm 2-Step RAG Agentic RAG Hybrid RAG naive RAG advanced RAG modular RAG",
        "categories": ["RAG overview", "RAG paper"],
    },
    {
        "name": "workflow",
        "triggers": ["流程", "阶段", "步骤", "怎么做", "实现", "pipeline", "workflow", "stage"],
        "question": "RAG 从文档接入、切分、索引、检索、上下文组装、生成到评估的完整流程是什么？",
        "search_query": "RAG workflow ingestion loading chunking indexing retrieval context assembly response synthesis evaluation",
        "categories": ["RAG overview", "ingestion", "indexing", "querying", "evaluation"],
    },
    {
        "name": "techniques",
        "triggers": [
            "关键技术",
            "技术",
            "组件",
            "模块",
            "能力",
            "technique",
            "component",
            "混合检索",
            "metadata",
            "过滤",
            "重排",
            "向量相似度",
            "hybrid",
            "rerank",
            "filter",
        ],
        "question": "RAG 的关键技术组件有哪些？请覆盖检索资料直接支持的文档处理、检索、重排、上下文组装、向量库和评估等组件。",
        "search_query": "RAG key components document parsing chunking embedding vector database hybrid retrieval reranking context assembly response synthesis evaluation",
        "categories": [
            "document loading",
            "chunking",
            "embedding",
            "vector db",
            "retrieval",
            "reranking",
            "querying",
            "evaluation",
        ],
    },
    {
        "name": "query_optimization",
        "triggers": [
            "query rewrite",
            "query expansion",
            "query transformation",
            "rewrite",
            "expansion",
            "查询改写",
            "查询扩展",
            "问题改写",
            "多路查询",
            "子问题",
        ],
        "question": "RAG 中为什么要做 query rewrite、query expansion 或 query routing？请说明它们如何改善召回、处理复杂或模糊问题，并支持多路检索或子问题拆解。",
        "search_query": "RAG query optimization query rewrite query expansion query transformation query routing multi query sub-question retrieval recall ambiguous complex questions",
        "categories": ["querying", "retrieval", "RAG overview"],
    },
    {
        "name": "bottlenecks",
        "triggers": ["瓶颈", "难点", "挑战", "限制", "缺陷", "痛点", "bottleneck", "challenge", "limitation"],
        "question": "RAG 系统常见瓶颈、挑战和失败点有哪些？请只总结检索资料能够直接支持的内容。",
        "search_query": "RAG bottlenecks challenges failure points Missing Content Missed the Top Ranked Documents Not in Context Not Extracted Incorrect Specificity Incomplete hallucination retrieval errors top K chunking embeddings reranking context token limit rate limit testing monitoring",
        "categories": ["RAG challenges", "retrieval", "chunking", "embedding", "reranking", "evaluation", "vector db", "RAG overview"],
    },
    {
        "name": "evaluation",
        "triggers": ["评估", "指标", "质量", "命中率", "忠实", "groundedness", "faithfulness", "eval", "metric"],
        "question": "RAG 系统应该如何评估？包括检索相关性、忠实性、正确性、引用质量、badcase 分析和 LLM-as-judge。",
        "search_query": "RAG evaluation metrics retrieval relevance faithfulness groundedness correctness citation quality citation recall citation precision badcase failure analysis diagnostic metrics claim-level context precision context recall context utilization hallucination noise sensitivity LLM as judge RAGChecker ALCE Ragas DeepEval Phoenix",
        "categories": ["evaluation", "retrieval"],
    },
    {
        "name": "citation_quality",
        "triggers": ["citation", "cite", "source attribution", "attribution", "verifiability"],
        "question": "RAG 答案的引用质量应该如何评估？请说明引用召回、引用精度、引用支撑性和可验证性。",
        "search_query": "ALCE citation quality citation recall citation precision cited passages supporting evidence verifiability attributable to identified sources NLI entailment citation evaluation",
        "categories": ["evaluation"],
    },
    {
        "name": "badcase_analysis",
        "triggers": ["badcase", "failure analysis", "error analysis", "diagnostic"],
        "question": "RAG 系统的 badcase 应该如何分析？请区分检索错误、生成错误、上下文利用不足、噪声敏感和幻觉等诊断信号。",
        "search_query": "RAGChecker diagnostic metrics failure analysis badcase sources of errors claim-level retriever metrics generator metrics context utilization noise sensitivity hallucination self-knowledge faithfulness",
        "categories": ["evaluation", "RAG challenges"],
    },
]


def plan_query(query: str, max_categories: int = 2) -> QueryPlan:
    normalized = normalize_query(query)
    scores = score_categories(normalized)
    top = [(category, score) for category, score in scores if score > 0][:max_categories]
    categories = [category for category, _ in top]
    aspects = detect_aspects(query, normalized)
    categories = merge_categories(categories, categories_from_aspects(aspects), max_categories=max(2, max_categories))
    confidence = calculate_confidence(scores)
    intent = infer_intent(normalized, categories)
    sub_queries = build_sub_queries(query, categories, aspects)
    warnings: list[str] = []
    if not categories:
        warnings.append("no confident category matched; using broad vector retrieval")
    if aspects:
        warnings.append("multi-aspect retrieval enabled for compound question")

    return QueryPlan(
        original_query=query,
        rewritten_query=query.strip(),
        intent=intent,
        category_filters=categories,
        sub_queries=sub_queries,
        aspects=aspects,
        confidence=confidence,
        warnings=warnings,
    )


def plan_query_v3(query: str, max_categories: int = 2) -> QueryPlan:
    """Build a conservative plan that expands only explicit independent needs."""
    normalized = normalize_query(query)
    scores = score_categories(normalized)
    aspects = detect_conservative_aspects(query, normalized)
    categories = select_conservative_categories(scores, aspects, max_categories)
    confidence = calculate_confidence(scores)
    sub_queries = build_conservative_sub_queries(query, aspects)
    warnings = ["conservative planner v3 preserves original-query entities and constraints"]
    if not categories:
        warnings.append("no confident category matched; using broad vector retrieval")
    if len(aspects) < 2:
        warnings.append("fewer than two independent aspects; expansion disabled")
    else:
        warnings.append(f"bounded expansion enabled for {len(aspects)} explicit aspects")

    return QueryPlan(
        original_query=query,
        rewritten_query=query.strip(),
        intent=infer_intent(normalized, categories),
        category_filters=categories,
        sub_queries=sub_queries,
        aspects=aspects,
        confidence=confidence,
        warnings=warnings,
        planner_version="rules_v3_conservative",
    )


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def score_categories(normalized_query: str) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    for category, hints in CATEGORY_HINTS.items():
        score = 0.0
        for hint in hints:
            hint_norm = hint.lower()
            if hint_norm in normalized_query:
                score += 2.0 if len(hint_norm) > 4 else 1.0
        if category.lower() in normalized_query:
            score += 2.0
        scores.append((category, score))
    return sorted(scores, key=lambda item: item[1], reverse=True)


def calculate_confidence(scores: list[tuple[str, float]]) -> float:
    if not scores or scores[0][1] <= 0:
        return 0.0
    top = scores[0][1]
    second = scores[1][1] if len(scores) > 1 else 0
    confidence = min(1.0, (top + max(0, top - second)) / 8)
    return round(confidence, 2)


def infer_intent(normalized_query: str, categories: list[str]) -> str:
    if any(word in normalized_query for word in ["怎么", "如何", "why", "为什么", "原理"]):
        return "explain"
    if any(word in normalized_query for word in ["比较", "区别", "vs", "对比"]):
        return "compare"
    if any(word in normalized_query for word in ["步骤", "流程", "怎么做", "实现"]):
        return "how_to"
    if "evaluation" in categories or "evaluation" in normalized_query or "评估" in normalized_query:
        return "evaluate"
    return "answer"


def detect_aspects(query: str, normalized_query: str) -> list[QueryAspect]:
    aspects: list[QueryAspect] = []
    for definition in ASPECT_DEFINITIONS:
        triggers = [str(item).lower() for item in definition["triggers"]]  # type: ignore[index]
        if any(trigger in normalized_query for trigger in triggers):
            aspects.append(
                QueryAspect(
                    name=str(definition["name"]),
                    question=str(definition["question"]),
                    search_query=str(definition.get("search_query", "")),
                    categories=[str(item) for item in definition["categories"]],  # type: ignore[index]
                )
            )

    aspect_names = {aspect.name for aspect in aspects}
    if "evaluation" in aspect_names:
        if any(marker in normalized_query for marker in ["badcase", "failure analysis", "error analysis", "diagnostic"]):
            append_aspect_if_missing(aspects, "badcase_analysis")
        if any(
            marker in normalized_query
            for marker in ["citation", "cite", "source attribution", "attribution", "verifiability", "badcase", "metric", "指标"]
        ):
            append_aspect_if_missing(aspects, "citation_quality")

    if is_broad_rag_question(normalized_query) and not aspects:
        for name in ["classification", "techniques", "bottlenecks"]:
            append_aspect_if_missing(aspects, name)

    specialize_aspects_for_query(aspects, normalized_query)
    return aspects


def detect_conservative_aspects(query: str, normalized_query: str) -> list[QueryAspect]:
    """Detect only explicit, independently retrievable answer requirements."""
    aspects: list[QueryAspect] = []
    if contains_any(
        normalized_query,
        ["完整流程", "整体流程", "端到端流程", "rag workflow", "rag pipeline stages"],
    ):
        append_aspect_if_missing(aspects, "workflow")
    if contains_any(
        normalized_query,
        ["分类", "架构类型", "架构模式", "范式", "2-step rag", "agentic rag", "hybrid rag"],
    ):
        append_aspect_if_missing(aspects, "classification")
    if contains_any(
        normalized_query,
        ["关键技术", "核心技术", "技术组件", "系统组件", "核心组件", "由哪些模块", "包含哪些模块"],
    ):
        append_aspect_if_missing(aspects, "techniques")
    if contains_any(
        normalized_query,
        ["query rewrite", "query expansion", "query transformation", "查询改写", "查询扩展", "问题改写", "多路查询", "子问题"],
    ):
        append_aspect_if_missing(aspects, "query_optimization")
    if contains_any(
        normalized_query,
        ["瓶颈", "难点", "挑战", "限制", "缺陷", "痛点", "bottleneck", "challenge", "limitation"],
    ):
        append_aspect_if_missing(aspects, "bottlenecks")
    if contains_any(
        normalized_query,
        ["评估", "指标", "忠实", "groundedness", "faithfulness", "evaluation", "metric"],
    ):
        append_aspect_if_missing(aspects, "evaluation")
    if contains_any(
        normalized_query,
        ["citation", "引用质量", "引用召回", "引用精度", "source attribution", "verifiability"],
    ):
        append_aspect_if_missing(aspects, "citation_quality")
    if contains_any(
        normalized_query,
        ["badcase", "failure analysis", "error analysis", "diagnostic", "错误分析"],
    ):
        append_aspect_if_missing(aspects, "badcase_analysis")

    has_ingestion_scope = contains_any(
        normalized_query,
        ["ingestion", "摄取", "入库", "知识库更新", "增量更新", "向量写入"],
    )
    if has_ingestion_scope and contains_any(
        normalized_query,
        ["幂等", "重复", "去重", "文档id", "document id", "旧向量", "一致性", "过期"],
    ):
        aspects.append(
            QueryAspect(
                name="ingestion_identity",
                question="增量摄取时，文档 ID、去重、旧向量替换和向量写入如何保证幂等与一致性？",
                search_query=(
                    "ingestion pipeline document ID idempotency deduplication update delete stale "
                    "vectors vector store consistency"
                ),
                categories=["ingestion", "indexing", "vector db"],
            )
        )
    if has_ingestion_scope and contains_any(normalized_query, ["缓存", "cache"]):
        aspects.append(
            QueryAspect(
                name="ingestion_cache",
                question="增量摄取中的缓存应如何命中、失效和清理，避免复用过期处理结果？",
                search_query=(
                    "ingestion pipeline cache invalidation document update transformation cache "
                    "clear stale processing results"
                ),
                categories=["ingestion", "indexing"],
            )
        )

    specialize_aspects_for_query(aspects, normalized_query)
    return anchor_aspect_queries(query, dedupe_aspects(aspects))


def select_conservative_categories(
    scores: list[tuple[str, float]],
    aspects: list[QueryAspect],
    max_categories: int,
) -> list[str]:
    if max_categories <= 0:
        return []
    scored = [category for category, score in scores if score > 0]
    specific = [category for category in scored if category not in {"RAG overview", "RAG challenges"}]
    candidates = categories_from_aspects(aspects) + specific + scored
    return dedupe(candidates)[:max_categories]


def build_conservative_sub_queries(query: str, aspects: list[QueryAspect]) -> list[str]:
    if len(aspects) < 2:
        return [query.strip()]
    return dedupe([query.strip()] + [aspect.search_query for aspect in aspects])


def anchor_aspect_queries(query: str, aspects: list[QueryAspect]) -> list[QueryAspect]:
    anchored: list[QueryAspect] = []
    for aspect in aspects:
        focus = aspect.search_query or aspect.question
        anchored.append(
            QueryAspect(
                name=aspect.name,
                question=aspect.question,
                search_query=f"{query.strip()} 检索重点: {focus}",
                categories=list(aspect.categories),
                required=aspect.required,
            )
        )
    return anchored


def dedupe_aspects(aspects: list[QueryAspect]) -> list[QueryAspect]:
    seen: set[str] = set()
    result: list[QueryAspect] = []
    for aspect in aspects:
        if aspect.name in seen:
            continue
        seen.add(aspect.name)
        result.append(aspect)
    return result


def contains_any(text: str, markers: list[str]) -> bool:
    return any(marker in text for marker in markers)


def specialize_aspects_for_query(aspects: list[QueryAspect], normalized_query: str) -> None:
    for aspect in aspects:
        if aspect.name != "techniques":
            continue
        if is_chroma_metadata_question(normalized_query):
            specialize_chroma_metadata_aspect(aspect)
        elif is_reranking_role_question(normalized_query):
            specialize_reranking_aspect(aspect)
        elif is_hybrid_retrieval_strategy_question(normalized_query):
            specialize_hybrid_retrieval_aspect(aspect)


def specialize_hybrid_retrieval_aspect(aspect: QueryAspect) -> None:
    aspect.question = (
        "为什么企业级 RAG 需要混合检索、metadata 过滤和重排，而不是只依赖一次向量相似度搜索？"
        "请围绕召回覆盖、精确过滤、候选排序和单路向量检索的失败风险回答；"
        "只有在检索资料直接支持时，再补充成本、延迟或工程取舍。"
    )
    aspect.search_query = (
        "enterprise RAG hybrid retrieval metadata filtering reranking vector similarity search "
        "BGE-M3 dense sparse retrieval re-ranking failure precision recall"
    )
    aspect.categories = ["retrieval", "vector db", "embedding", "reranking", "RAG challenges"]


def specialize_reranking_aspect(aspect: QueryAspect) -> None:
    aspect.question = (
        "重排在 RAG 中解决什么问题？它和第一阶段向量检索或语义检索是什么关系？"
        "请说明候选召回、二阶段排序、reranker 或 cross-encoder 的作用，以及为什么它是在检索之后进一步筛选候选。"
    )
    aspect.search_query = (
        "RAG reranking reranker candidate documents semantic search dense retrieval "
        "cross encoder relationship vector retrieval"
    )
    aspect.categories = ["reranking", "retrieval", "embedding", "RAG overview"]


def specialize_chroma_metadata_aspect(aspect: QueryAspect) -> None:
    aspect.question = (
        "Chroma 在这个 RAG 项目里承担什么角色？metadata filter 有什么用？"
        "请围绕文档、embedding、metadata 的存储，collection/query，查询时按 metadata 过滤，以及它如何缩小检索范围回答。"
    )
    aspect.search_query = (
        "Chroma vector database collection metadata filter where embeddings document storage retrieval RAG"
    )
    aspect.categories = ["vector db", "retrieval", "ingestion"]


def is_chroma_metadata_question(normalized_query: str) -> bool:
    return "chroma" in normalized_query and any(
        marker in normalized_query for marker in ["metadata", "filter", "where", "过滤"]
    )


def is_reranking_role_question(normalized_query: str) -> bool:
    has_rerank = any(marker in normalized_query for marker in ["rerank", "reranking", "reranker", "重排"])
    has_hybrid_scope = any(marker in normalized_query for marker in ["hybrid", "混合检索", "metadata"])
    return has_rerank and not has_hybrid_scope


def is_hybrid_retrieval_strategy_question(normalized_query: str) -> bool:
    has_hybrid = any(marker in normalized_query for marker in ["hybrid", "混合检索"])
    has_single_vector_contrast = any(
        marker in normalized_query
        for marker in ["only vector", "single vector", "vector similarity", "向量相似度", "只做一次"]
    )
    has_filter_or_rerank = any(marker in normalized_query for marker in ["metadata", "filter", "过滤", "rerank", "重排"])
    return (has_hybrid or has_single_vector_contrast) and has_filter_or_rerank


def append_aspect_if_missing(aspects: list[QueryAspect], name: str) -> None:
    if any(aspect.name == name for aspect in aspects):
        return
    definition = next(item for item in ASPECT_DEFINITIONS if item["name"] == name)
    aspects.append(
        QueryAspect(
            name=str(definition["name"]),
            question=str(definition["question"]),
            search_query=str(definition.get("search_query", "")),
            categories=[str(item) for item in definition["categories"]],  # type: ignore[index]
        )
    )


def is_broad_rag_question(normalized_query: str) -> bool:
    has_rag = "rag" in normalized_query or "检索增强" in normalized_query or "知识库" in normalized_query
    broad_markers = ["介绍", "概览", "系统", "整体", "完整", "有哪些", "是什么", "overview"]
    return has_rag and any(marker in normalized_query for marker in broad_markers)


def categories_from_aspects(aspects: list[QueryAspect]) -> list[str]:
    categories: list[str] = []
    for aspect in aspects:
        categories.extend(aspect.categories)
    return dedupe(categories)


def merge_categories(primary: list[str], secondary: list[str], max_categories: int) -> list[str]:
    merged = dedupe(primary + secondary)
    if len(merged) <= max_categories:
        return merged
    return merged[:max_categories]


def build_sub_queries(query: str, categories: list[str], aspects: list[QueryAspect] | None = None) -> list[str]:
    sub_queries = [query.strip()]
    for aspect in aspects or []:
        sub_queries.append(aspect.question)
        if aspect.search_query:
            sub_queries.append(aspect.search_query)
    for category in categories:
        expansion = CATEGORY_EXPANSIONS.get(category)
        if expansion:
            sub_queries.append(f"{query.strip()} {expansion}")
    return dedupe(sub_queries)


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out
