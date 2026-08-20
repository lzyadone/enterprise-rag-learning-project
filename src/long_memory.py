"""Persistent long-term memory for the local RAG workbench.

Long memory is deliberately separated from the factual RAG knowledge base.
It helps interpret user preferences, goals, project history, and follow-up
intent, but it must not be cited as source evidence in answers.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb

from src.answer_audit import parse_json_object
from src.deepseek_client import DEFAULT_BASE_URL, chat_completion, get_deepseek_api_key
from src.ollama_http import embed_query


DEFAULT_NAMESPACE = "local_user"
DEFAULT_COLLECTION = "rag_long_memory"
SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{12,}"),
    re.compile(r"(?i)(api[_ -]?key|secret|token)\s*[:=]\s*\S+"),
]


@dataclass
class MemoryRecord:
    id: str
    namespace: str
    kind: str
    content: str
    importance: float
    source_turn_id: str
    created_at: float
    updated_at: float
    last_accessed_at: float
    access_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    distance: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "kind": self.kind,
            "content": self.content,
            "importance": self.importance,
            "source_turn_id": self.source_turn_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "access_count": self.access_count,
            "metadata": self.metadata,
            "score": self.score,
            "distance": self.distance,
        }


@dataclass
class MemoryCandidate:
    kind: str
    content: str
    importance: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


class LongMemoryStore:
    def __init__(
        self,
        sqlite_path: Path,
        chroma_dir: Path,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.sqlite_path = sqlite_path
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(sqlite_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.setup()

    def setup(self) -> None:
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_titles_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                namespace TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL NOT NULL,
                source_turn_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_accessed_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                metadata_json TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                UNIQUE(namespace, content_hash)
            )
            """
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
        self.db.commit()

    def add_turn(
        self,
        namespace: str,
        session_id: str,
        user: str,
        answer: str,
        sources: list[dict[str, Any]],
        embedding_model: str,
        ollama_host: str,
        use_llm_extraction: bool = True,
    ) -> list[MemoryRecord]:
        turn_id = str(uuid.uuid4())
        now = time.time()
        source_titles = [str(source.get("title", "")) for source in sources[:5] if source.get("title")]
        self.db.execute(
            """
            INSERT INTO turns(id, namespace, session_id, user, answer, source_titles_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (turn_id, namespace, session_id, user, answer, json.dumps(source_titles, ensure_ascii=False), now),
        )
        self.db.commit()

        candidates = extract_memory_candidates(user, answer, source_titles, use_llm=use_llm_extraction)
        stored: list[MemoryRecord] = []
        for candidate in candidates:
            record = self.add_memory(namespace, candidate, turn_id, embedding_model, ollama_host)
            if record:
                stored.append(record)
        return stored

    def add_memory(
        self,
        namespace: str,
        candidate: MemoryCandidate,
        source_turn_id: str,
        embedding_model: str,
        ollama_host: str,
    ) -> MemoryRecord | None:
        content = sanitize_memory(candidate.content)
        if not should_store_memory(content):
            return None

        now = time.time()
        content_hash = stable_hash(f"{candidate.kind}\n{content}")
        existing = self.db.execute(
            "SELECT * FROM memories WHERE namespace = ? AND content_hash = ?",
            (namespace, content_hash),
        ).fetchone()
        if existing:
            self.db.execute(
                """
                UPDATE memories
                SET updated_at = ?, importance = MAX(importance, ?)
                WHERE id = ?
                """,
                (now, float(candidate.importance), existing["id"]),
            )
            self.db.commit()
            record = self.row_to_record(existing)
            record.updated_at = now
            record.importance = max(record.importance, float(candidate.importance))
            return record

        memory_id = str(uuid.uuid4())
        metadata = dict(candidate.metadata or {})
        metadata["content_hash"] = content_hash
        self.db.execute(
            """
            INSERT INTO memories(
                id, namespace, kind, content, importance, source_turn_id,
                created_at, updated_at, last_accessed_at, access_count,
                metadata_json, content_hash
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                namespace,
                candidate.kind,
                content,
                float(candidate.importance),
                source_turn_id,
                now,
                now,
                0.0,
                0,
                json.dumps(metadata, ensure_ascii=False),
                content_hash,
            ),
        )
        self.db.commit()

        embedding = embed_query(content, embedding_model, ollama_host)
        self.collection.upsert(
            ids=[memory_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[
                {
                    "namespace": namespace,
                    "kind": candidate.kind,
                    "importance": float(candidate.importance),
                    "created_at": now,
                }
            ],
        )
        return self.get(memory_id)

    def search(
        self,
        query: str,
        namespace: str,
        embedding_model: str,
        ollama_host: str,
        top_k: int = 5,
    ) -> list[MemoryRecord]:
        if self.count(namespace) == 0:
            return []

        query_embedding = embed_query(query, embedding_model, ollama_host)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(top_k * 3, top_k),
            where={"namespace": namespace},
            include=["documents", "metadatas", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        records: list[MemoryRecord] = []
        for memory_id, distance in zip(ids, distances):
            record = self.get(str(memory_id))
            if not record:
                continue
            record.distance = float(distance)
            record.score = memory_score(record, float(distance))
            records.append(record)

        records = sorted(records, key=lambda item: (-item.score, item.distance))[:top_k]
        self.mark_accessed([record.id for record in records])
        return records

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self.row_to_record(row) if row else None

    def mark_accessed(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = time.time()
        self.db.executemany(
            "UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1 WHERE id = ?",
            [(now, memory_id) for memory_id in memory_ids],
        )
        self.db.commit()

    def recent(self, namespace: str, limit: int = 8) -> list[MemoryRecord]:
        rows = self.db.execute(
            """
            SELECT * FROM memories
            WHERE namespace = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (namespace, limit),
        ).fetchall()
        return [self.row_to_record(row) for row in rows]

    def count(self, namespace: str | None = None) -> int:
        if namespace:
            row = self.db.execute("SELECT COUNT(*) AS n FROM memories WHERE namespace = ?", (namespace,)).fetchone()
        else:
            row = self.db.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        return int(row["n"] if row else 0)

    def stats(self, namespace: str = DEFAULT_NAMESPACE) -> dict[str, Any]:
        rows = self.db.execute(
            """
            SELECT kind, COUNT(*) AS n
            FROM memories
            WHERE namespace = ?
            GROUP BY kind
            ORDER BY n DESC
            """,
            (namespace,),
        ).fetchall()
        return {
            "namespace": namespace,
            "memory_count": self.count(namespace),
            "turn_count": self.turn_count(namespace),
            "by_kind": {str(row["kind"]): int(row["n"]) for row in rows},
        }

    def turn_count(self, namespace: str) -> int:
        row = self.db.execute("SELECT COUNT(*) AS n FROM turns WHERE namespace = ?", (namespace,)).fetchone()
        return int(row["n"] if row else 0)

    def clear(self, namespace: str = DEFAULT_NAMESPACE) -> None:
        ids = [row["id"] for row in self.db.execute("SELECT id FROM memories WHERE namespace = ?", (namespace,)).fetchall()]
        if ids:
            self.collection.delete(ids=ids)
        self.db.execute("DELETE FROM memories WHERE namespace = ?", (namespace,))
        self.db.execute("DELETE FROM turns WHERE namespace = ?", (namespace,))
        self.db.commit()

    def row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        metadata = json.loads(row["metadata_json"] or "{}")
        return MemoryRecord(
            id=str(row["id"]),
            namespace=str(row["namespace"]),
            kind=str(row["kind"]),
            content=str(row["content"]),
            importance=float(row["importance"]),
            source_turn_id=str(row["source_turn_id"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_accessed_at=float(row["last_accessed_at"]),
            access_count=int(row["access_count"]),
            metadata=metadata,
        )


def extract_memory_candidates(
    user: str,
    answer: str,
    source_titles: list[str],
    use_llm: bool = True,
) -> list[MemoryCandidate]:
    if use_llm and get_deepseek_api_key():
        try:
            candidates = extract_with_deepseek(user, answer, source_titles)
            if candidates:
                return candidates
        except Exception:
            pass
    return extract_with_rules(user, answer, source_titles)


def extract_with_deepseek(user: str, answer: str, source_titles: list[str]) -> list[MemoryCandidate]:
    prompt = f"""请从这轮对话中抽取长期记忆，返回 JSON 对象。

只保存对后续对话长期有用的信息，例如：
- 用户长期偏好、学习方式、约束
- 用户当前长期目标
- 项目已经完成的阶段或重要决策
- 后续需要继续处理的开放问题

不要保存：
- API key、token、密码等秘密
- 检索资料里的普通知识点
- 一次性临时问题
- 太长的答案原文

返回格式：
{{
  "memories": [
    {{"kind": "preference|goal|project_state|decision|open_question|episode", "content": "...", "importance": 0.0到1.0}}
  ]
}}

用户：
{user}

助手答案摘要：
{" ".join(answer.split())[:1200]}

本轮来源标题：
{", ".join(source_titles)}
"""
    raw = chat_completion(
        [
            {"role": "system", "content": "你是严谨的长期记忆抽取器。只返回 JSON。"},
            {"role": "user", "content": prompt},
        ],
        model="deepseek-chat",
        base_url=DEFAULT_BASE_URL,
        temperature=0.0,
        max_tokens=900,
        response_format=None,
    )
    parsed = parse_json_object(raw)
    values = parsed.get("memories", [])
    if not isinstance(values, list):
        return []
    candidates: list[MemoryCandidate] = []
    for item in values[:6]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        kind = normalize_kind(str(item.get("kind", "episode")))
        importance = clamp_float(item.get("importance", 0.5), low=0.0, high=1.0)
        if content:
            candidates.append(MemoryCandidate(kind=kind, content=content, importance=importance, metadata={"extractor": "deepseek"}))
    return candidates


def extract_with_rules(user: str, answer: str, source_titles: list[str]) -> list[MemoryCandidate]:
    candidates: list[MemoryCandidate] = []
    user_clean = " ".join(user.split())
    answer_preview = " ".join(answer.split())[:280]

    if re.search(r"(我希望|我想|我需要|不要|不想|每天|后面|以后|作品集|github|项目|学习)", user_clean, flags=re.I):
        kind = "preference" if re.search(r"(我希望|不要|不想|每天|学习)", user_clean) else "goal"
        candidates.append(
            MemoryCandidate(
                kind=kind,
                content=f"用户长期偏好或目标：{user_clean[:260]}",
                importance=0.75,
                metadata={"extractor": "rules"},
            )
        )

    if any(marker in user_clean for marker in ["继续", "下一步", "现在", "项目", "RAG", "网页", "长记忆"]):
        source_text = f"；相关来源：{', '.join(source_titles[:3])}" if source_titles else ""
        candidates.append(
            MemoryCandidate(
                kind="project_state",
                content=f"项目对话进展：用户问题是“{user_clean[:180]}”；助手输出摘要：{answer_preview}{source_text}",
                importance=0.55,
                metadata={"extractor": "rules"},
            )
        )

    return candidates


def format_long_memory_context(records: list[MemoryRecord]) -> str:
    if not records:
        return ""
    lines = [
        "长期记忆仅用于理解用户偏好、项目状态和追问指代，不得作为事实来源引用。",
    ]
    for idx, record in enumerate(records, start=1):
        lines.append(
            f"{idx}. [{record.kind}, importance={record.importance:.2f}, score={record.score:.3f}] {record.content}"
        )
    return "\n".join(lines)


def sanitize_memory(content: str) -> str:
    content = " ".join(content.strip().split())
    for pattern in SECRET_PATTERNS:
        content = pattern.sub("[REDACTED_SECRET]", content)
    return content[:600]


def should_store_memory(content: str) -> bool:
    if len(content) < 8:
        return False
    if "[REDACTED_SECRET]" in content and len(content) < 40:
        return False
    return True


def normalize_kind(kind: str) -> str:
    allowed = {"preference", "goal", "project_state", "decision", "open_question", "episode"}
    kind = kind.strip().lower()
    return kind if kind in allowed else "episode"


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def memory_score(record: MemoryRecord, distance: float) -> float:
    semantic = 1.0 / (1.0 + max(0.0, distance))
    recency = 0.0
    if record.updated_at:
        age_days = max(0.0, (time.time() - record.updated_at) / 86400)
        recency = 1.0 / (1.0 + age_days)
    return round((0.72 * semantic) + (0.18 * record.importance) + (0.10 * recency), 6)


def clamp_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))
