"""Lazy-loaded cross-encoder reranking with CPU and multilingual GPU backends."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKEND = "transformers"
DEFAULT_FASTEMBED_MODEL = "BAAI/bge-reranker-base"
DEFAULT_TRANSFORMERS_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BATCH_SIZE = 2
DEFAULT_MAX_LENGTH = 512


class CrossEncoder(Protocol):
    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 64, **kwargs: Any) -> Iterable[float]: ...


class TransformersCrossEncoder:
    """Minimal Transformers adapter for encoder-only sequence classifiers."""

    def __init__(self, model_name: str, cache_dir: Path, device: str, max_length: int) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The transformers GPU reranker is not installed; install requirements-reranker-gpu.txt"
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"RAG_RERANKER_DEVICE={device}, but PyTorch cannot access CUDA")

        cache_dir.mkdir(parents=True, exist_ok=True)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._torch = torch
        self._inference_lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            dtype=dtype,
        ).to(device)
        self.model.eval()

    def rerank(self, query: str, documents: Iterable[str], batch_size: int = 2, **_: Any) -> list[float]:
        document_list = list(documents)
        scores: list[float] = []
        try:
            # Serialize requests so two web clients cannot over-commit a 4 GB GPU.
            with self._inference_lock, self._torch.inference_mode():
                for start in range(0, len(document_list), batch_size):
                    pairs = [[query, document] for document in document_list[start : start + batch_size]]
                    inputs = self.tokenizer(
                        pairs,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    logits = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu().tolist()
                    scores.extend(float(value) for value in logits)
        except RuntimeError as exc:
            if "out of memory" in str(exc).casefold() and self.device.startswith("cuda"):
                self._torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA ran out of memory while reranking; reduce RAG_RERANKER_BATCH_SIZE or "
                    "RAG_RERANKER_MAX_LENGTH"
                ) from exc
            raise
        return scores


_MODEL_CACHE: dict[tuple[str, str, str, str, int], CrossEncoder] = {}
_MODEL_LOCK = threading.Lock()


def cross_encoder_rerank(
    query: str,
    chunks: Sequence[Any],
    top_k: int,
    *,
    encoder: CrossEncoder | None = None,
    model_name: str | None = None,
    cache_dir: Path | None = None,
    batch_size: int | None = None,
    backend: str | None = None,
    retrieval_weight: float = 0.0,
) -> list[Any]:
    """Score each query-document pair jointly and return the highest logits."""
    if top_k <= 0 or not chunks:
        return []
    if not 0.0 <= retrieval_weight <= 1.0:
        raise ValueError("retrieval_weight must be between 0 and 1")

    resolved_backend = backend or configured_backend()
    resolved_model_name = model_name or configured_model_name(resolved_backend)
    resolved_batch_size = batch_size or configured_batch_size()
    active_encoder = encoder or get_cross_encoder(
        resolved_model_name,
        cache_dir=cache_dir,
        backend=resolved_backend,
    )
    documents = [build_reranker_document(item) for item in chunks]
    try:
        scores = [float(score) for score in active_encoder.rerank(query, documents, batch_size=resolved_batch_size)]
    except Exception as exc:  # noqa: BLE001 - add actionable context without hiding the original failure.
        raise RuntimeError(f"cross-encoder reranking failed for {resolved_model_name}: {type(exc).__name__}: {exc}") from exc

    if len(scores) != len(chunks):
        raise RuntimeError(f"cross-encoder returned {len(scores)} scores for {len(chunks)} candidates")

    model_order = sorted(range(len(scores)), key=lambda index: (-scores[index], chunks[index].distance))
    model_ranks = {candidate_index: rank for rank, candidate_index in enumerate(model_order, start=1)}
    scored: list[tuple[float, Any]] = []
    for base_rank, (item, model_score) in enumerate(zip(chunks, scores), start=1):
        base_score = float(getattr(item, "score", 0.0) or 0.0)
        if retrieval_weight:
            model_rank = model_ranks[base_rank - 1]
            final_score = retrieval_weight / (60 + base_rank) + (1.0 - retrieval_weight) / (60 + model_rank)
            score_reason = (
                f"rank_fusion={final_score:.6f}; retrieval_rank={base_rank}; "
                f"model_rank={model_rank}; retrieval_weight={retrieval_weight:.2f}"
            )
        else:
            final_score = model_score
            score_reason = f"cross_encoder={model_score:.6f}"
        setattr(item, "rerank_score", model_score)
        setattr(
            item,
            "rerank_reason",
            f"{score_reason}; model_score={model_score:.6f}; base={base_score:.6f}; model={resolved_model_name}",
        )
        item.score = final_score
        scored.append((final_score, item))

    return [item for _, item in sorted(scored, key=lambda pair: (-pair[0], pair[1].distance))[:top_k]]


def get_cross_encoder(
    model_name: str | None = None,
    cache_dir: Path | None = None,
    *,
    backend: str | None = None,
) -> CrossEncoder:
    """Load one model per process and reuse it across requests."""
    resolved_backend = backend or configured_backend()
    resolved_model_name = model_name or configured_model_name(resolved_backend)
    resolved_cache_dir = (cache_dir or configured_cache_dir(resolved_backend)).resolve()
    device = configured_device(resolved_backend)
    max_length = configured_max_length()
    cache_key = (resolved_backend, resolved_model_name, str(resolved_cache_dir), device, max_length)

    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if resolved_backend == "fastembed":
            model = load_fastembed_encoder(resolved_model_name, resolved_cache_dir)
        elif resolved_backend == "transformers":
            model = TransformersCrossEncoder(
                resolved_model_name,
                resolved_cache_dir,
                device=device,
                max_length=max_length,
            )
        else:
            raise ValueError(f"Unsupported RAG_RERANKER_BACKEND: {resolved_backend}")
        _MODEL_CACHE[cache_key] = model
        return model


def load_fastembed_encoder(model_name: str, cache_dir: Path) -> CrossEncoder:
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
    except ImportError as exc:
        raise RuntimeError("fastembed is required for the fastembed reranker; install requirements.txt") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    return TextCrossEncoder(
        model_name=model_name,
        cache_dir=str(cache_dir),
        threads=configured_threads(),
        cuda=False,
    )


def preload_cross_encoder(
    model_name: str | None = None,
    cache_dir: Path | None = None,
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    resolved_backend = backend or configured_backend()
    resolved_model_name = model_name or configured_model_name(resolved_backend)
    resolved_cache_dir = cache_dir or configured_cache_dir(resolved_backend)
    get_cross_encoder(resolved_model_name, resolved_cache_dir, backend=resolved_backend)
    return runtime_config(resolved_model_name, resolved_cache_dir, backend=resolved_backend)


def runtime_config(
    model_name: str | None = None,
    cache_dir: Path | None = None,
    *,
    backend: str | None = None,
) -> dict[str, Any]:
    resolved_backend = backend or configured_backend()
    device = configured_device(resolved_backend)
    return {
        "model": model_name or configured_model_name(resolved_backend),
        "backend": resolved_backend,
        "device": device,
        "dtype": "float16" if device.startswith("cuda") else "float32",
        "cache_dir": str((cache_dir or configured_cache_dir(resolved_backend)).resolve()),
        "threads": configured_threads() if resolved_backend == "fastembed" else None,
        "batch_size": configured_batch_size(),
        "max_length": configured_max_length(),
    }


def build_reranker_document(item: Any) -> str:
    metadata = getattr(item, "metadata", {}) or {}
    return "\n".join(
        [
            f"Title: {metadata.get('title', '')}",
            f"Category: {metadata.get('category', '')}",
            f"Section: {metadata.get('heading_path', '')}",
            str(getattr(item, "document", "")),
        ]
    ).strip()


def configured_backend() -> str:
    value = os.getenv("RAG_RERANKER_BACKEND", DEFAULT_BACKEND).strip().casefold()
    if value not in {"fastembed", "transformers"}:
        raise ValueError(f"RAG_RERANKER_BACKEND must be fastembed or transformers, got {value!r}")
    return value


def configured_model_name(backend: str | None = None) -> str:
    resolved_backend = backend or configured_backend()
    default = DEFAULT_FASTEMBED_MODEL if resolved_backend == "fastembed" else DEFAULT_TRANSFORMERS_MODEL
    return os.getenv("RAG_RERANKER_MODEL", default).strip() or default


def configured_cache_dir(backend: str | None = None) -> Path:
    value = os.getenv("RAG_RERANKER_CACHE_DIR", "").strip()
    if value:
        return Path(value)
    resolved_backend = backend or configured_backend()
    dirname = "fastembed_cache" if resolved_backend == "fastembed" else "huggingface"
    return PROJECT_ROOT / "data" / "runtime" / dirname


def configured_device(backend: str | None = None) -> str:
    resolved_backend = backend or configured_backend()
    if resolved_backend == "fastembed":
        return "cpu"
    value = os.getenv("RAG_RERANKER_DEVICE", "auto").strip().casefold() or "auto"
    if value != "auto":
        return value
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def configured_threads() -> int:
    default = min(4, max(1, os.cpu_count() or 1))
    return positive_int_env("RAG_RERANKER_THREADS", default)


def configured_batch_size() -> int:
    return positive_int_env("RAG_RERANKER_BATCH_SIZE", DEFAULT_BATCH_SIZE)


def configured_max_length() -> int:
    return positive_int_env("RAG_RERANKER_MAX_LENGTH", DEFAULT_MAX_LENGTH)


def positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed
