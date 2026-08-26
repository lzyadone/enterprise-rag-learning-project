"""Thread-safe bounded caches for planned retrieval candidates and reranking."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from threading import RLock
from typing import Any, Hashable


CANDIDATE_CACHE_MAX_SIZE = 512
RERANK_CACHE_MAX_SIZE = 128


class BoundedResultCache:
    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.max_size = max_size
        self._values: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Hashable) -> tuple[bool, Any]:
        with self._lock:
            if key not in self._values:
                self._misses += 1
                return False, None
            self._hits += 1
            self._values.move_to_end(key)
            return True, deepcopy(self._values[key])

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._values[key] = deepcopy(value)
            self._values.move_to_end(key)
            while len(self._values) > self.max_size:
                self._values.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0

    def info(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._values),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }


_candidate_cache = BoundedResultCache(CANDIDATE_CACHE_MAX_SIZE)
_rerank_cache = BoundedResultCache(RERANK_CACHE_MAX_SIZE)


def get_cached_candidates(key: Hashable) -> tuple[bool, Any]:
    return _candidate_cache.get(key)


def cache_candidates(key: Hashable, value: Any) -> None:
    _candidate_cache.put(key, value)


def get_cached_rerank(key: Hashable) -> tuple[bool, Any]:
    return _rerank_cache.get(key)


def cache_rerank(key: Hashable, value: Any) -> None:
    _rerank_cache.put(key, value)


def clear_retrieval_caches() -> None:
    _candidate_cache.clear()
    _rerank_cache.clear()


def retrieval_cache_info() -> dict[str, dict[str, int]]:
    return {
        "candidates": _candidate_cache.info(),
        "rerank": _rerank_cache.info(),
    }
