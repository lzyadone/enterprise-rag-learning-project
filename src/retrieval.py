"""Retrieval utilities for direct and planned RAG search."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

from src.bm25_retrieval import get_bm25_index
from src.cross_encoder_reranking import cross_encoder_rerank
from src.ollama_http import embed_query, unload_embedding_model
from src.query_planning import QueryPlan, plan_query, plan_query_v3
from src.reranking import lexical_rerank


ASPECT_EVIDENCE_TERMS: dict[str, list[str]] = {
    "bottlenecks": [
        "failure points of rag systems",
        "missing content",
        "missed the top ranked documents",
        "not in context",
        "not extracted",
        "incorrect specificity",
        "incomplete",
        "token limit",
        "rate limit",
        "hallucinations",
        "robustness",
        "testing and monitoring",
    ],
    "techniques": [
        "chunking",
        "embedding",
        "retrieval strategy",
        "rerank",
        "metadata",
        "query process",
        "evaluation",
    ],
    "query_optimization": [
        "query rewrite",
        "query expansion",
        "query transformation",
        "query routing",
        "multi query",
        "sub-question",
        "sub query",
        "ambiguous",
        "complex",
        "retrieval validation",
        "query enhancement",
        "recall",
    ],
    "classification": [
        "naive rag",
        "advanced rag",
        "modular rag",
        "2-step rag",
        "agentic rag",
        "hybrid rag",
        "paradigm",
        "architecture",
    ],
    "evaluation": [
        "citation quality",
        "citation recall",
        "citation precision",
        "claim-level",
        "diagnostic metrics",
        "context precision",
        "context recall",
        "context utilization",
        "contextual precision",
        "contextual recall",
        "document relevance",
        "faithfulness",
        "groundedness",
        "hallucination",
        "noise sensitivity",
        "badcase",
        "failure analysis",
        "ragchecker",
        "alce",
    ],
    "citation_quality": [
        "citation quality",
        "citation recall",
        "citation precision",
        "cited passages",
        "supporting evidence",
        "verifiability",
        "attributable",
        "nli model",
        "entails",
        "supported by cited passages",
        "alce",
    ],
    "badcase_analysis": [
        "diagnostic metrics",
        "claim-level",
        "sources of errors",
        "retriever metrics",
        "generator metrics",
        "context utilization",
        "noise sensitivity",
        "hallucination",
        "failure analysis",
        "badcase",
        "ragchecker",
    ],
}

MIN_ASPECT_SOURCES: dict[str, int] = {
    "classification": 2,
    "query_optimization": 2,
    "evaluation": 3,
    "citation_quality": 2,
    "badcase_analysis": 2,
}

DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "llm_rag_docs" / "chunks.jsonl"
PLANNED_ANCHOR_WEIGHT = 2.0
PLANNED_GLOBAL_EXPANSION_WEIGHT = 1.0
PLANNED_FILTERED_EXPANSION_WEIGHT = 0.5
V3_ANCHOR_WEIGHT = 3.0
V3_GLOBAL_EXPANSION_WEIGHT = 0.75
V3_FILTERED_EXPANSION_WEIGHT = 0.25


@dataclass
class RetrievedChunk:
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    distance: float
    score: float
    source_query: str
    category_filter: str | None
    aspect: str | None
    rank: int
    rerank_score: float = 0.0
    rerank_reason: str = ""
    retrieval_channels: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlannedRunSpec:
    query: str
    category: str | None
    aspect: str | None
    group: str


def direct_retrieve(
    collection: chromadb.Collection,
    query: str,
    embedding_model: str,
    ollama_host: str,
    top_k: int = 5,
    candidate_k: int | None = None,
    category: str | None = None,
    rerank_mode: str = "none",
    query_embedding: list[float] | None = None,
    use_embedding_cache: bool = True,
) -> list[RetrievedChunk]:
    retrieval_k = max(top_k, candidate_k or top_k)
    if query_embedding is None:
        query_embedding = embed_query(
            query,
            embedding_model,
            ollama_host,
            use_cache=use_embedding_cache,
        )
    where = {"category": category} if category else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=retrieval_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    candidates = unpack_results(results, query, category)
    prepare_model_residency(rerank_mode, embedding_model, ollama_host)
    return rerank(query, candidates, top_k, rerank_mode)


def bm25_retrieve(
    query: str,
    top_k: int = 5,
    category: str | None = None,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
) -> list[RetrievedChunk]:
    hits = get_bm25_index(chunks_path).search(query, top_k=top_k, category=category)
    return [
        RetrievedChunk(
            chunk_id=hit.chunk_id,
            document=hit.document,
            metadata=hit.metadata,
            distance=1.0 / (1.0 + hit.score),
            score=hit.score,
            source_query=query,
            category_filter=category,
            aspect=None,
            rank=hit.rank,
            retrieval_channels=["bm25"],
        )
        for hit in hits
    ]


def hybrid_retrieve(
    collection: chromadb.Collection,
    query: str,
    embedding_model: str,
    ollama_host: str,
    top_k: int = 5,
    candidate_k: int | None = None,
    category: str | None = None,
    rerank_mode: str = "none",
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    query_embedding: list[float] | None = None,
    use_embedding_cache: bool = True,
) -> list[RetrievedChunk]:
    """Fuse dense Chroma and sparse BM25 candidates with reciprocal-rank fusion."""
    retrieval_k = max(top_k, candidate_k or top_k)
    dense = direct_retrieve(
        collection,
        query,
        embedding_model,
        ollama_host,
        top_k=retrieval_k,
        candidate_k=retrieval_k,
        category=category,
        rerank_mode="none",
        query_embedding=query_embedding,
        use_embedding_cache=use_embedding_cache,
    )
    sparse = bm25_retrieve(query, top_k=retrieval_k, category=category, chunks_path=chunks_path)
    fused = reciprocal_rank_fusion([dense, sparse])
    prepare_model_residency(rerank_mode, embedding_model, ollama_host)
    return rerank(query, fused, top_k, rerank_mode)


def planned_retrieve(
    collection: chromadb.Collection,
    query: str,
    embedding_model: str,
    ollama_host: str,
    top_k: int = 5,
    candidate_k: int = 8,
    max_categories: int = 2,
    manual_category: str | None = None,
    rerank_mode: str = "none",
    retrieval_strategy: str = "dense",
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    reuse_query_embeddings: bool = True,
    query_plan: QueryPlan | None = None,
    fusion_mode: str = "legacy",
) -> tuple[QueryPlan, list[RetrievedChunk]]:
    if fusion_mode in {"anchored", "conservative"}:
        return anchored_planned_retrieve(
            collection,
            query,
            embedding_model,
            ollama_host,
            top_k=top_k,
            candidate_k=candidate_k,
            max_categories=max_categories,
            manual_category=manual_category,
            rerank_mode=rerank_mode,
            retrieval_strategy=retrieval_strategy,
            chunks_path=chunks_path,
            reuse_query_embeddings=reuse_query_embeddings,
            query_plan=query_plan,
            planning_mode="conservative" if fusion_mode == "conservative" else "legacy",
        )
    if fusion_mode != "legacy":
        raise ValueError("fusion_mode must be legacy, anchored, or conservative")

    plan = query_plan or plan_query(query, max_categories=max_categories)
    if manual_category:
        plan.category_filters = [manual_category]
        plan.warnings.append("manual category override applied")
    if plan.aspects:
        adaptive_top_k = min(10, max(top_k, len(plan.aspects) * 2 + 4))
        if adaptive_top_k > top_k:
            plan.warnings.append(f"adaptive top_k expanded from {top_k} to {adaptive_top_k} for aspect coverage")
            top_k = adaptive_top_k
        candidate_k = max(candidate_k, top_k * 2)

    retrieval_queries = []
    retrieval_queries.extend(
        aspect.search_query or aspect.question
        for aspect in plan.aspects
    )
    retrieval_queries.extend(plan.sub_queries)
    unique_queries = list(dict.fromkeys(retrieval_queries))
    embedding_by_query = (
        {
            query_text: embed_query(query_text, embedding_model, ollama_host)
            for query_text in unique_queries
        }
        if reuse_query_embeddings
        else {}
    )

    retrieval_runs: list[list[RetrievedChunk]] = []

    for aspect in plan.aspects:
        aspect_query = aspect.search_query or aspect.question
        aspect_runs: list[list[RetrievedChunk]] = []
        aspect_runs.append(
            retrieve_with_strategy(
                collection,
                aspect_query,
                embedding_model,
                ollama_host,
                top_k=candidate_k,
                category=None,
                rerank_mode="none",
                retrieval_strategy=retrieval_strategy,
                chunks_path=chunks_path,
                query_embedding=embedding_by_query.get(aspect_query),
                use_embedding_cache=reuse_query_embeddings,
            )
        )
        for category in aspect.categories[:8]:
            aspect_runs.append(
                retrieve_with_strategy(
                    collection,
                    aspect_query,
                    embedding_model,
                    ollama_host,
                    top_k=max(3, candidate_k // 2),
                    category=category,
                    rerank_mode="none",
                    retrieval_strategy=retrieval_strategy,
                    chunks_path=chunks_path,
                    query_embedding=embedding_by_query.get(aspect_query),
                    use_embedding_cache=reuse_query_embeddings,
                )
            )
        for run in aspect_runs:
            for item in run:
                item.aspect = aspect.name
        retrieval_runs.extend(aspect_runs)

    for sub_query in plan.sub_queries:
        retrieval_runs.append(
            retrieve_with_strategy(
                collection,
                sub_query,
                embedding_model,
                ollama_host,
                top_k=candidate_k,
                category=None,
                rerank_mode="none",
                retrieval_strategy=retrieval_strategy,
                chunks_path=chunks_path,
                query_embedding=embedding_by_query.get(sub_query),
                use_embedding_cache=reuse_query_embeddings,
            )
        )
        for category in plan.category_filters:
            retrieval_runs.append(
                retrieve_with_strategy(
                    collection,
                    sub_query,
                    embedding_model,
                    ollama_host,
                    top_k=candidate_k,
                    category=category,
                    rerank_mode="none",
                    retrieval_strategy=retrieval_strategy,
                    chunks_path=chunks_path,
                    query_embedding=embedding_by_query.get(sub_query),
                    use_embedding_cache=reuse_query_embeddings,
                )
            )

    fused = reciprocal_rank_fusion(retrieval_runs)
    rerank_pool = fused
    if rerank_mode == "cross_encoder":
        rerank_pool_size = min(len(fused), max(candidate_k, top_k * 3))
        rerank_pool = select_with_plan_coverage(fused, rerank_pool_size, plan)
        if rerank_pool_size < len(fused):
            plan.warnings.append(
                f"cross-encoder pool limited from {len(fused)} to {rerank_pool_size} candidates"
            )
    prepare_model_residency(rerank_mode, embedding_model, ollama_host)
    ranked = rerank(query, rerank_pool, len(rerank_pool), rerank_mode)
    ranked = apply_plan_boosts(ranked, plan)
    return plan, select_with_plan_coverage(ranked, top_k, plan)


def anchored_planned_retrieve(
    collection: chromadb.Collection,
    query: str,
    embedding_model: str,
    ollama_host: str,
    top_k: int = 5,
    candidate_k: int = 8,
    max_categories: int = 2,
    manual_category: str | None = None,
    rerank_mode: str = "none",
    retrieval_strategy: str = "dense",
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    reuse_query_embeddings: bool = True,
    query_plan: QueryPlan | None = None,
    planning_mode: str = "legacy",
) -> tuple[QueryPlan, list[RetrievedChunk]]:
    """Retrieve with an original-query anchor and bounded expansion influence."""
    if planning_mode not in {"legacy", "conservative"}:
        raise ValueError("planning_mode must be legacy or conservative")
    conservative = planning_mode == "conservative"
    plan = query_plan or (
        plan_query_v3(query, max_categories=max_categories)
        if conservative
        else plan_query(query, max_categories=max_categories)
    )
    anchor_candidate_k = candidate_k
    if manual_category:
        plan.category_filters = [manual_category]
        plan.warnings.append("manual category override applied")
    if plan.aspects and (not conservative or len(plan.aspects) >= 2):
        adaptive_top_k = min(10, max(top_k, len(plan.aspects) * 2 + 4))
        if adaptive_top_k > top_k:
            plan.warnings.append(f"adaptive top_k expanded from {top_k} to {adaptive_top_k} for aspect coverage")
            top_k = adaptive_top_k
        candidate_k = max(candidate_k, top_k * 2)

    specs = (
        build_conservative_run_specs(query, plan)
        if conservative
        else build_anchored_run_specs(query, plan)
    )
    unique_queries = list(dict.fromkeys(spec.query for spec in specs))
    embedding_by_query = (
        {
            query_text: embed_query(query_text, embedding_model, ollama_host)
            for query_text in unique_queries
        }
        if reuse_query_embeddings
        else {}
    )
    weighted_runs: list[tuple[list[RetrievedChunk], float]] = []
    weights = conservative_run_weights(specs) if conservative else anchored_run_weights(specs)
    for spec, weight in zip(specs, weights):
        run_top_k = top_k if spec.group == "anchor" else candidate_k
        run_candidate_k = anchor_candidate_k if spec.group == "anchor" else candidate_k
        run = retrieve_with_strategy(
            collection,
            spec.query,
            embedding_model,
            ollama_host,
            top_k=run_top_k,
            candidate_k=run_candidate_k,
            category=spec.category,
            rerank_mode="none",
            retrieval_strategy=retrieval_strategy,
            chunks_path=chunks_path,
            query_embedding=embedding_by_query.get(spec.query),
            use_embedding_cache=reuse_query_embeddings,
        )
        if spec.aspect:
            for item in run:
                item.aspect = spec.aspect
        weighted_runs.append((run, weight))

    fused = reciprocal_rank_fusion(
        [run for run, _ in weighted_runs],
        weights=[weight for _, weight in weighted_runs],
    )
    rerank_pool = fused
    if conservative:
        coverage_slots = min(2, top_k // 4) if len(plan.aspects) >= 2 else 0
    else:
        coverage_slots = max(1, min(2, top_k // 4))
    if rerank_mode == "cross_encoder":
        rerank_pool_size = min(len(fused), max(candidate_k, top_k * 3))
        rerank_pool = select_with_bounded_plan_coverage(
            fused,
            rerank_pool_size,
            plan,
            max_coverage_slots=coverage_slots,
        )
        if rerank_pool_size < len(fused):
            plan.warnings.append(
                f"cross-encoder pool limited from {len(fused)} to {rerank_pool_size} candidates"
            )
    prepare_model_residency(rerank_mode, embedding_model, ollama_host)
    ranked = rerank(query, rerank_pool, len(rerank_pool), rerank_mode)
    ranked = apply_plan_boosts(ranked, plan, max_boost=0.006)
    fusion_label = "conservative v3" if conservative else "anchored"
    plan.warnings.append(
        f"{fusion_label} fusion used {len(specs)} deduplicated runs with "
        f"{coverage_slots} coverage slots"
    )
    return plan, select_with_bounded_plan_coverage(
        ranked,
        top_k,
        plan,
        max_coverage_slots=coverage_slots,
    )


def build_anchored_run_specs(query: str, plan: QueryPlan) -> list[PlannedRunSpec]:
    specs: list[PlannedRunSpec] = []
    seen: set[tuple[str, str | None]] = set()

    def add(query_text: str, category: str | None, aspect: str | None, group: str) -> None:
        normalized = " ".join(query_text.split()).casefold()
        key = (normalized, category)
        if not normalized or key in seen:
            return
        seen.add(key)
        specs.append(
            PlannedRunSpec(
                query=query_text.strip(),
                category=category,
                aspect=aspect,
                group=group,
            )
        )

    add(query, None, None, "anchor")
    for category in plan.category_filters:
        add(query, category, None, "filtered_expansion")
    for aspect in plan.aspects:
        aspect_query = aspect.search_query or aspect.question
        add(aspect_query, None, aspect.name, "global_expansion")
        for category in aspect.categories[:8]:
            add(aspect_query, category, aspect.name, "filtered_expansion")
    for sub_query in plan.sub_queries:
        add(sub_query, None, None, "global_expansion")
        for category in plan.category_filters:
            add(sub_query, category, None, "filtered_expansion")
    return specs


def build_conservative_run_specs(query: str, plan: QueryPlan) -> list[PlannedRunSpec]:
    """Build a small expansion set whose queries retain the original user wording."""
    specs: list[PlannedRunSpec] = []
    seen: set[tuple[str, str | None]] = set()

    def add(query_text: str, category: str | None, aspect: str | None, group: str) -> None:
        normalized = " ".join(query_text.split()).casefold()
        key = (normalized, category)
        if not normalized or key in seen:
            return
        seen.add(key)
        specs.append(PlannedRunSpec(query_text.strip(), category, aspect, group))

    add(query, None, None, "anchor")
    if len(plan.aspects) < 2:
        return specs

    for category in plan.category_filters[:2]:
        add(query, category, None, "filtered_expansion")
    for aspect in plan.aspects[:4]:
        aspect_query = aspect.search_query or f"{query} 检索重点: {aspect.question}"
        add(aspect_query, None, aspect.name, "global_expansion")
        for category in aspect.categories[:1]:
            add(aspect_query, category, aspect.name, "filtered_expansion")
    return specs


def anchored_run_weights(specs: list[PlannedRunSpec]) -> list[float]:
    group_totals = {
        "anchor": PLANNED_ANCHOR_WEIGHT,
        "global_expansion": PLANNED_GLOBAL_EXPANSION_WEIGHT,
        "filtered_expansion": PLANNED_FILTERED_EXPANSION_WEIGHT,
    }
    counts = {
        group: sum(1 for spec in specs if spec.group == group)
        for group in group_totals
    }
    return [group_totals[spec.group] / counts[spec.group] for spec in specs]


def conservative_run_weights(specs: list[PlannedRunSpec]) -> list[float]:
    group_totals = {
        "anchor": V3_ANCHOR_WEIGHT,
        "global_expansion": V3_GLOBAL_EXPANSION_WEIGHT,
        "filtered_expansion": V3_FILTERED_EXPANSION_WEIGHT,
    }
    counts = {group: sum(1 for spec in specs if spec.group == group) for group in group_totals}
    return [group_totals[spec.group] / counts[spec.group] for spec in specs]


def retrieve_with_strategy(
    collection: chromadb.Collection,
    query: str,
    embedding_model: str,
    ollama_host: str,
    top_k: int,
    candidate_k: int | None = None,
    category: str | None = None,
    rerank_mode: str = "none",
    retrieval_strategy: str = "dense",
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    query_embedding: list[float] | None = None,
    use_embedding_cache: bool = True,
) -> list[RetrievedChunk]:
    if retrieval_strategy == "dense":
        return direct_retrieve(
            collection,
            query,
            embedding_model,
            ollama_host,
            top_k=top_k,
            candidate_k=candidate_k,
            category=category,
            rerank_mode=rerank_mode,
            query_embedding=query_embedding,
            use_embedding_cache=use_embedding_cache,
        )
    if retrieval_strategy == "hybrid":
        return hybrid_retrieve(
            collection,
            query,
            embedding_model,
            ollama_host,
            top_k=top_k,
            candidate_k=candidate_k,
            category=category,
            rerank_mode=rerank_mode,
            chunks_path=chunks_path,
            query_embedding=query_embedding,
            use_embedding_cache=use_embedding_cache,
        )
    raise ValueError(f"Unsupported retrieval_strategy: {retrieval_strategy}")


def rerank(query: str, candidates: list[RetrievedChunk], top_k: int, rerank_mode: str) -> list[RetrievedChunk]:
    if rerank_mode == "none":
        return candidates[:top_k]
    if rerank_mode == "lexical":
        return lexical_rerank(query, candidates, top_k=top_k)
    if rerank_mode == "cross_encoder":
        return cross_encoder_rerank(query, candidates, top_k=top_k)
    raise ValueError(f"Unsupported rerank_mode: {rerank_mode}")


def prepare_model_residency(rerank_mode: str, embedding_model: str, ollama_host: str) -> None:
    """Avoid overlapping Ollama embeddings and a local GPU reranker on small machines."""
    if rerank_mode != "cross_encoder":
        return
    enabled = os.getenv("RAG_EXCLUSIVE_MODEL_RESIDENCY", "1").strip().casefold()
    if enabled in {"0", "false", "no", "off"}:
        return
    unload_embedding_model(embedding_model, ollama_host)


def apply_plan_boosts(
    ranked: list[RetrievedChunk],
    plan: QueryPlan,
    *,
    max_boost: float = 0.24,
) -> list[RetrievedChunk]:
    if max_boost < 0:
        raise ValueError("max_boost must be non-negative")
    if not plan.aspects:
        return ranked

    aspect_names = {aspect.name for aspect in plan.aspects}
    for item in ranked:
        boost = 0.0
        searchable = "\n".join(
            [
                str(item.metadata.get("title", "")),
                str(item.metadata.get("heading_path", "")),
                item.document,
            ]
        ).lower()
        for aspect_name in aspect_names:
            if item.aspect != aspect_name:
                continue
            terms = ASPECT_EVIDENCE_TERMS.get(aspect_name, [])
            hits = sum(1 for term in terms if term in searchable)
            if hits:
                boost += min(max_boost, (max_boost / 6) * hits)
        if boost:
            item.score = round(float(item.score or 0.0) + boost, 6)
            item.rerank_score = round(float(item.rerank_score or 0.0) + boost, 6)
            item.rerank_reason = (item.rerank_reason + f"; plan_boost={boost:.3f}").strip("; ")

    return sorted(ranked, key=lambda item: (-item.score, item.distance))


def select_with_category_coverage(
    ranked: list[RetrievedChunk],
    top_k: int,
    required_categories: list[str],
) -> list[RetrievedChunk]:
    """Keep the best item first, then cover planned categories when possible."""
    if top_k <= 0 or not ranked:
        return []
    if not required_categories or top_k == 1:
        return ranked[:top_k]

    selected: list[RetrievedChunk] = [ranked[0]]
    selected_ids = {ranked[0].chunk_id}
    covered = {str(ranked[0].metadata.get("category", ""))}

    for category in required_categories:
        if len(selected) >= top_k:
            break
        if category in covered:
            continue
        match = first_matching_category(ranked, category, selected_ids)
        if match:
            selected.append(match)
            selected_ids.add(match.chunk_id)
            covered.add(category)

    for item in ranked:
        if len(selected) >= top_k:
            break
        if item.chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.chunk_id)

    return selected


def select_with_plan_coverage(
    ranked: list[RetrievedChunk],
    top_k: int,
    plan: QueryPlan,
) -> list[RetrievedChunk]:
    if top_k <= 0 or not ranked:
        return []
    if not plan.aspects:
        return select_with_category_coverage(ranked, top_k, plan.category_filters)

    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()

    for aspect in plan.aspects:
        preferred_categories = preferred_categories_for_aspect(aspect.name, aspect.categories)
        match = first_matching_aspect_category(ranked, aspect.name, preferred_categories, selected_ids)
        if not match:
            match = first_matching_aspect(ranked, aspect.name, selected_ids)
        add_selected(selected, selected_ids, match, top_k)

    for aspect in plan.aspects:
        target_count = MIN_ASPECT_SOURCES.get(aspect.name, 1)
        while count_selected_aspect(selected, aspect.name) < target_count and len(selected) < top_k:
            preferred_categories = preferred_categories_for_aspect(aspect.name, aspect.categories)
            excluded_sources = selected_source_ids_for_aspect(selected, aspect.name)
            match = first_matching_aspect_category(
                ranked,
                aspect.name,
                preferred_categories,
                selected_ids,
                excluded_source_ids=excluded_sources,
            )
            if not match:
                match = first_matching_aspect(ranked, aspect.name, selected_ids, excluded_source_ids=excluded_sources)
            if not add_selected(selected, selected_ids, match, top_k):
                break

    for category in priority_categories_for_plan(plan):
        if len(selected) >= top_k:
            break
        if category in selected_categories(selected):
            continue
        match = first_matching_category(ranked, category, selected_ids)
        add_selected(selected, selected_ids, match, top_k)

    for item in ranked:
        if len(selected) >= top_k:
            break
        add_selected(selected, selected_ids, item, top_k)

    return selected


def select_with_bounded_plan_coverage(
    ranked: list[RetrievedChunk],
    top_k: int,
    plan: QueryPlan,
    *,
    max_coverage_slots: int,
) -> list[RetrievedChunk]:
    """Reserve only a small number of slots for plan coverage, then follow fused relevance."""
    if top_k <= 0 or not ranked:
        return []
    if max_coverage_slots < 0:
        raise ValueError("max_coverage_slots must be non-negative")

    selected: list[RetrievedChunk] = [ranked[0]]
    selected_ids = {ranked[0].chunk_id}
    coverage_added = 0

    for aspect in plan.aspects:
        if coverage_added >= max_coverage_slots or len(selected) >= top_k:
            break
        preferred_categories = preferred_categories_for_aspect(aspect.name, aspect.categories)
        match = first_matching_aspect_category(ranked, aspect.name, preferred_categories, selected_ids)
        if not match:
            match = first_matching_aspect(ranked, aspect.name, selected_ids)
        if add_selected(selected, selected_ids, match, top_k):
            coverage_added += 1

    for category in priority_categories_for_plan(plan):
        if coverage_added >= max_coverage_slots or len(selected) >= top_k:
            break
        if category in selected_categories(selected):
            continue
        match = first_matching_category(ranked, category, selected_ids)
        if add_selected(selected, selected_ids, match, top_k):
            coverage_added += 1

    for item in ranked:
        if len(selected) >= top_k:
            break
        add_selected(selected, selected_ids, item, top_k)

    return selected


def preferred_categories_for_aspect(aspect_name: str, categories: list[str]) -> list[str]:
    if aspect_name == "classification":
        return categories
    specific = [category for category in categories if category not in {"RAG overview", "RAG paper"}]
    return specific or categories


def priority_categories_for_plan(plan: QueryPlan) -> list[str]:
    priority = [
        "RAG challenges",
        "chunking",
        "embedding",
        "retrieval",
        "reranking",
        "evaluation",
        "querying",
        "vector db",
        "document loading",
        "indexing",
        "ingestion",
        "RAG overview",
        "RAG paper",
    ]
    requested = set(plan.category_filters)
    for aspect in plan.aspects:
        requested.update(aspect.categories)
    return [category for category in priority if category in requested]


def add_selected(
    selected: list[RetrievedChunk],
    selected_ids: set[str],
    item: RetrievedChunk | None,
    top_k: int,
) -> bool:
    if item is None or len(selected) >= top_k or item.chunk_id in selected_ids:
        return False
    selected.append(item)
    selected_ids.add(item.chunk_id)
    return True


def selected_categories(selected: list[RetrievedChunk]) -> set[str]:
    return {str(item.metadata.get("category", "")) for item in selected}


def count_selected_aspect(selected: list[RetrievedChunk], aspect: str) -> int:
    return sum(1 for item in selected if item.aspect == aspect)


def selected_source_ids_for_aspect(selected: list[RetrievedChunk], aspect: str) -> set[str]:
    return {str(item.metadata.get("source_id", "")) for item in selected if item.aspect == aspect}


def select_with_aspect_coverage(
    ranked: list[RetrievedChunk],
    top_k: int,
    required_aspects: list[str],
) -> list[RetrievedChunk]:
    """Keep high-ranked chunks while reserving slots for each planned answer aspect."""
    if top_k <= 0 or not ranked:
        return []
    if not required_aspects or top_k == 1:
        return ranked[:top_k]

    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()

    for aspect in required_aspects:
        if len(selected) >= top_k:
            break
        match = first_matching_aspect(ranked, aspect, selected_ids)
        if match:
            selected.append(match)
            selected_ids.add(match.chunk_id)

    for item in ranked:
        if len(selected) >= top_k:
            break
        if item.chunk_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.chunk_id)

    return sorted(selected, key=lambda item: (-item.score, item.distance))


def first_matching_aspect_category(
    ranked: list[RetrievedChunk],
    aspect: str,
    categories: list[str],
    excluded_ids: set[str],
    excluded_source_ids: set[str] | None = None,
) -> RetrievedChunk | None:
    category_set = set(categories)
    excluded_source_ids = excluded_source_ids or set()
    for item in ranked:
        if item.chunk_id in excluded_ids:
            continue
        if str(item.metadata.get("source_id", "")) in excluded_source_ids:
            continue
        if item.aspect == aspect and str(item.metadata.get("category", "")) in category_set:
            return item
    return None


def first_matching_aspect(
    ranked: list[RetrievedChunk],
    aspect: str,
    excluded_ids: set[str],
    excluded_source_ids: set[str] | None = None,
) -> RetrievedChunk | None:
    excluded_source_ids = excluded_source_ids or set()
    for item in ranked:
        if item.chunk_id in excluded_ids:
            continue
        if str(item.metadata.get("source_id", "")) in excluded_source_ids:
            continue
        if item.aspect == aspect:
            return item
    return None


def first_matching_category(
    ranked: list[RetrievedChunk],
    category: str,
    excluded_ids: set[str],
) -> RetrievedChunk | None:
    for item in ranked:
        if item.chunk_id in excluded_ids:
            continue
        if str(item.metadata.get("category", "")) == category:
            return item
    return None


def unpack_results(results: dict[str, Any], query: str, category: str | None) -> list[RetrievedChunk]:
    rows: list[RetrievedChunk] = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1,
    ):
        rows.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                document=str(document),
                metadata=dict(metadata or {}),
                distance=float(distance),
                score=0.0,
                source_query=query,
                category_filter=category,
                aspect=None,
                rank=rank,
                retrieval_channels=["dense"],
            )
        )
    return rows


def reciprocal_rank_fusion(
    runs: list[list[RetrievedChunk]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[RetrievedChunk]:
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(runs)
    if len(weights) != len(runs):
        raise ValueError("weights must have the same length as runs")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")

    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}
    for run, weight in zip(runs, weights):
        for rank, item in enumerate(run, start=1):
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + weight / (k + rank)
            previous = fused.get(item.chunk_id)
            previous_aspect = previous.aspect if previous else None
            merged_channels = sorted(set((previous.retrieval_channels if previous else []) + item.retrieval_channels))
            if item.chunk_id not in fused or item.distance < fused[item.chunk_id].distance:
                if previous_aspect and not item.aspect:
                    item.aspect = previous_aspect
                item.retrieval_channels = merged_channels
                fused[item.chunk_id] = item
            elif item.aspect and not fused[item.chunk_id].aspect:
                fused[item.chunk_id].aspect = item.aspect
            if item.chunk_id in fused:
                fused[item.chunk_id].retrieval_channels = merged_channels

    for chunk_id, item in fused.items():
        item.score = scores.get(chunk_id, 0.0)

    return sorted(fused.values(), key=lambda item: (-item.score, item.distance))
