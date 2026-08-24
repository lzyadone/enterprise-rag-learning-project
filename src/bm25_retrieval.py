"""Local BM25 retrieval over the processed knowledge-base chunks."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from rank_bm25 import BM25Okapi as RankBM25Okapi
except ImportError:  # pragma: no cover - covered by fallback behavior.
    RankBM25Okapi = None

from src.reranking import TERM_ALIASES


ENGLISH_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]{1,}")
CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class BM25Hit:
    chunk_id: str
    document: str
    metadata: dict[str, Any]
    score: float
    rank: int


class BM25ChunkIndex:
    """In-memory BM25 index built from the same chunks used by Chroma."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError("BM25 index requires at least one chunk")
        self.rows = rows
        corpus = [tokenize(build_index_text(row)) for row in rows]
        self.model = RankBM25Okapi(corpus) if RankBM25Okapi is not None else SimpleBM25Okapi(corpus)

    @classmethod
    def from_jsonl(cls, path: Path) -> BM25ChunkIndex:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not row.get("chunk_id") or not row.get("text"):
                    raise ValueError(f"{path}:{line_no} must contain chunk_id and text")
                rows.append(row)
        return cls(rows)

    def search(self, query: str, top_k: int, category: str | None = None) -> list[BM25Hit]:
        if top_k <= 0:
            return []
        query_tokens = expand_query_tokens(query)
        if not query_tokens:
            return []

        scores = self.model.get_scores(query_tokens)
        ranked_indexes = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
        hits: list[BM25Hit] = []
        for row_index in ranked_indexes:
            score = float(scores[row_index])
            if score <= 0:
                break
            row = self.rows[row_index]
            if category and str(row.get("category", "")) != category:
                continue
            metadata = {key: value for key, value in row.items() if key not in {"chunk_id", "text"}}
            hits.append(
                BM25Hit(
                    chunk_id=str(row["chunk_id"]),
                    document=str(row["text"]),
                    metadata=metadata,
                    score=score,
                    rank=len(hits) + 1,
                )
            )
            if len(hits) >= top_k:
                break
        return hits


def get_bm25_index(path: Path) -> BM25ChunkIndex:
    """Cache the index until the source JSONL file changes."""
    resolved = path.resolve()
    stat = resolved.stat()
    return _cached_bm25_index(str(resolved), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _cached_bm25_index(path: str, _mtime_ns: int, _size: int) -> BM25ChunkIndex:
    return BM25ChunkIndex.from_jsonl(Path(path))


def build_index_text(row: dict[str, Any]) -> str:
    """Give titles and structural metadata extra weight without changing source text."""
    title = str(row.get("title", ""))
    category = str(row.get("category", ""))
    heading = str(row.get("heading_path", ""))
    text = str(row.get("text", ""))
    return "\n".join([title, title, category, category, heading, heading, text])


def expand_query_tokens(query: str) -> list[str]:
    tokens = tokenize(query)
    normalized_query = query.casefold()
    for key, aliases in TERM_ALIASES.items():
        if key.casefold() not in normalized_query:
            continue
        for alias in aliases:
            tokens.extend(tokenize(alias))

    # Repeating query terms can accidentally overpower the BM25 score after
    # alias expansion, so keep each term once while preserving stable order.
    return list(Counter(tokens).keys())


def tokenize(text: str) -> list[str]:
    """Tokenize English terms plus Chinese bi/tri-grams for exact-term recall."""
    normalized = text.casefold()
    tokens = ENGLISH_TOKEN_RE.findall(normalized)
    for span in CJK_SPAN_RE.findall(normalized):
        if len(span) == 1:
            tokens.append(span)
            continue
        tokens.extend(span[index : index + 2] for index in range(len(span) - 1))
        if len(span) >= 3:
            tokens.extend(span[index : index + 3] for index in range(len(span) - 2))
    return tokens


class SimpleBM25Okapi:
    """Small BM25 fallback used when rank-bm25 is not installed."""

    def __init__(self, corpus: list[list[str]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(document) for document in corpus]
        self.doc_lens = [len(document) for document in corpus]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        import math

        doc_count = len(self.corpus)
        document_frequency: Counter[str] = Counter()
        for document in self.corpus:
            document_frequency.update(set(document))
        return {
            term: math.log(1 + (doc_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for term_counts, doc_len in zip(self.doc_freqs, self.doc_lens):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * doc_len / max(1.0, self.avgdl))
            for token in query_tokens:
                frequency = term_counts.get(token, 0)
                if not frequency:
                    continue
                score += self.idf.get(token, 0.0) * (frequency * (self.k1 + 1)) / (frequency + norm)
            scores.append(score)
        return scores
