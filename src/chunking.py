"""Structure-aware Markdown chunking for RAG ingestion.

The splitter treats Markdown headings as the primary structure. Character
limits are used only as safety caps for very long sections, not as the main
chunking strategy.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FRONTMATTER_RE = re.compile(r"(?s)^---\n.*?\n---\n*")


@dataclass
class MarkdownSection:
    heading_path: str
    heading_level: int
    body: str
    section_index: int


@dataclass
class Chunk:
    text: str
    heading_path: str
    heading_level: int
    section_index: int
    chunk_in_section: int
    char_count: int
    token_estimate: int
    text_hash: str


def strip_frontmatter(text: str) -> str:
    return FRONTMATTER_RE.sub("", text.strip(), count=1).strip()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def markdown_sections(text: str, default_title: str) -> list[MarkdownSection]:
    """Split Markdown into sections using headings while respecting code fences."""
    text = clean_metadata_lines(normalize_text(strip_frontmatter(text)))
    lines = text.splitlines()
    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = [(1, default_title)]
    current_lines: list[str] = []
    current_level = 1
    section_index = 0
    in_code = False

    def current_path() -> str:
        return " > ".join(title for _, title in heading_stack if title)

    def flush() -> None:
        nonlocal section_index, current_lines
        body = normalize_text("\n".join(current_lines))
        if body:
            sections.append(
                MarkdownSection(
                    heading_path=current_path(),
                    heading_level=current_level,
                    body=body,
                    section_index=section_index,
                )
            )
            section_index += 1
        current_lines = []

    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            current_lines.append(line)
            continue

        match = HEADING_RE.match(line) if not in_code else None
        if match:
            flush()
            level = len(match.group(1))
            title = clean_heading(match.group(2))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_level = level
            continue

        current_lines.append(line)

    flush()
    if not sections and text:
        sections.append(
            MarkdownSection(
                heading_path=default_title,
                heading_level=1,
                body=text,
                section_index=0,
            )
        )
    return sections


def clean_heading(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip(" #")


def clean_metadata_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if re.match(r"^Source:\s+https?://", line.strip(), flags=re.IGNORECASE):
            continue
        lines.append(line)
    return normalize_text("\n".join(lines))


def split_markdown_document(
    text: str,
    title: str,
    soft_max_chars: int = 1800,
    hard_max_chars: int = 3500,
    min_chars: int = 280,
) -> list[Chunk]:
    sections = markdown_sections(text, default_title=title)
    chunks: list[Chunk] = []

    for section in sections:
        parts = split_section_body(section.body, soft_max_chars, hard_max_chars)
        for part_index, part in enumerate(parts):
            chunk_text = format_chunk_text(title, section.heading_path, part)
            chunks.append(make_chunk(chunk_text, section, part_index))

    return merge_tiny_chunks(chunks, merge_max_chars=hard_max_chars, min_chars=min_chars)


def split_section_body(body: str, soft_max_chars: int, hard_max_chars: int) -> list[str]:
    body = normalize_text(body)
    if len(body) <= hard_max_chars:
        return [body]

    paragraphs = split_paragraphs(body)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > hard_max_chars:
            if current:
                chunks.append(normalize_text("\n\n".join(current)))
                current = []
                current_len = 0
            chunks.extend(split_long_paragraph(paragraph, hard_max_chars))
            continue

        next_len = current_len + len(paragraph) + (2 if current else 0)
        if current and next_len > soft_max_chars:
            chunks.append(normalize_text("\n\n".join(current)))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len = next_len

    if current:
        chunks.append(normalize_text("\n\n".join(current)))
    return [chunk for chunk in chunks if chunk]


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [normalize_text(part) for part in re.split(r"\n\s*\n", text)]
    return [paragraph for paragraph in paragraphs if paragraph]


def split_long_paragraph(paragraph: str, hard_max_chars: int) -> list[str]:
    pieces = re.split(r"(?<=[.!?。！？])\s+", paragraph)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > hard_max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(piece[i : i + hard_max_chars] for i in range(0, len(piece), hard_max_chars))
            continue
        if current and len(current) + 1 + len(piece) > hard_max_chars:
            chunks.append(current.strip())
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def format_chunk_text(title: str, heading_path: str, body: str) -> str:
    if heading_path and heading_path != title:
        return normalize_text(f"Document: {title}\nSection: {heading_path}\n\n{body}")
    return normalize_text(f"Document: {title}\n\n{body}")


def make_chunk(text: str, section: MarkdownSection, chunk_in_section: int) -> Chunk:
    text = normalize_text(text)
    return Chunk(
        text=text,
        heading_path=section.heading_path,
        heading_level=section.heading_level,
        section_index=section.section_index,
        chunk_in_section=chunk_in_section,
        char_count=len(text),
        token_estimate=estimate_tokens(text),
        text_hash=hash_text(text),
    )


def merge_tiny_chunks(chunks: list[Chunk], merge_max_chars: int, min_chars: int) -> list[Chunk]:
    if not chunks:
        return []

    merged: list[Chunk] = []
    for chunk in chunks:
        if (
            merged
            and (chunk.char_count < min_chars or merged[-1].char_count < min_chars)
            and merged[-1].char_count + chunk.char_count + 2 <= merge_max_chars
        ):
            previous = merged.pop()
            combined_text = normalize_text(previous.text + "\n\n" + chunk.text)
            combined = Chunk(
                text=combined_text,
                heading_path=previous.heading_path,
                heading_level=previous.heading_level,
                section_index=previous.section_index,
                chunk_in_section=previous.chunk_in_section,
                char_count=len(combined_text),
                token_estimate=estimate_tokens(combined_text),
                text_hash=hash_text(combined_text),
            )
            merged.append(combined)
        else:
            merged.append(chunk)
    return merged


def estimate_tokens(text: str) -> int:
    # A lightweight estimate good enough for chunk statistics.
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    other = max(0, len(text) - cjk_chars - sum(len(w) for w in re.findall(r"[A-Za-z0-9_]+", text)))
    return int(cjk_chars * 1.2 + latin_words * 1.3 + other / 4)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def chunks_to_records(
    chunks: Iterable[Chunk],
    doc_metadata: dict[str, str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    doc_id = str(doc_metadata["doc_id"])
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}::chunk_{idx:04d}"
        records.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "source_id": doc_metadata["source_id"],
                "title": doc_metadata["title"],
                "category": doc_metadata["category"],
                "priority": doc_metadata["priority"],
                "source_type": doc_metadata["source_type"],
                "url": doc_metadata["url"],
                "heading_path": chunk.heading_path,
                "heading_level": chunk.heading_level,
                "section_index": chunk.section_index,
                "chunk_index": idx,
                "chunk_in_section": chunk.chunk_in_section,
                "char_count": chunk.char_count,
                "token_estimate": chunk.token_estimate,
                "text_hash": chunk.text_hash,
                "text": chunk.text,
            }
        )
    return records
