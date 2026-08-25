"""Fetch curated LLM/RAG sources and convert them into local Markdown docs.

This script is intentionally manifest-driven:

1. Read high-quality sources from data/source_manifests/llm_rag_sources.csv.
2. Download only selected rows, usually P0 + ingest_first=yes.
3. Convert local files to Markdown through MarkItDown when possible.
4. Write one normalized documents.jsonl file for the future chunking/index stage.

The goal is to make the knowledge-base data layer reproducible instead of
manually copying pages into the project.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "source_manifests" / "llm_rag_sources.csv"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "llm_rag_docs"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "llm_rag_docs"
DEFAULT_MARKITDOWN = Path.home() / ".codex" / "skills" / "markitdown" / "scripts" / "convert.py"

USER_AGENT = (
    "enterprise-rag-learning-project/0.1 "
    "(learning portfolio; contact: local-user)"
)


@dataclass(frozen=True)
class Source:
    source_id: str
    priority: str
    category: str
    title: str
    source_type: str
    url: str
    ingest_first: str
    why: str
    notes: str
    extract_symbol: str = ""


class BasicHTMLToMarkdown(HTMLParser):
    """Small stdlib fallback for HTML pages when MarkItDown is unavailable."""

    BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "footer",
        "li",
        "ul",
        "ol",
        "table",
        "tr",
        "blockquote",
        "pre",
        "br",
    }

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0
        self.heading_level: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n\n" + "#" * int(tag[1]) + " ")
            self.heading_level = int(tag[1])
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n\n")
            self.heading_level = None
        elif tag in self.BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        text = html.unescape(data)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            self.parts.append(text + " ")

    def markdown(self) -> str:
        text = "".join(self.parts)
        return normalize_markdown(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and normalize LLM/RAG source docs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--priority", default="P0", help="Comma-separated priorities, e.g. P0 or P0,P1.")
    parser.add_argument("--ingest-first-only", action="store_true", default=True)
    parser.add_argument("--include-all", action="store_true", help="Ignore ingest_first filter.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source-id", action="append", default=None, help="Fetch only selected source_id. Can repeat.")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="Re-download and re-convert existing files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--convert-timeout", type=int, default=90)
    parser.add_argument("--sleep", type=float, default=0.5, help="Polite delay between downloads.")
    parser.add_argument("--markitdown-convert", type=Path, default=DEFAULT_MARKITDOWN)
    parser.add_argument("--no-convert", action="store_true")
    return parser.parse_args()


def read_sources(manifest: Path) -> list[Source]:
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = set(Source.__dataclass_fields__.keys()) - {"extract_symbol"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    sources = []
    for row in rows:
        values = {field: row[field].strip() for field in required}
        values["extract_symbol"] = (row.get("extract_symbol") or "").strip()
        sources.append(Source(**values))
    return sources


def select_eligible_sources(sources: Iterable[Source], args: argparse.Namespace) -> list[Source]:
    priorities = {item.strip() for item in args.priority.split(",") if item.strip()}
    selected = [source for source in sources if source.priority in priorities]
    if not args.include_all and args.ingest_first_only:
        selected = [source for source in selected if source.ingest_first.lower() == "yes"]
    return selected


def select_sources(sources: Iterable[Source], args: argparse.Namespace) -> list[Source]:
    selected = select_eligible_sources(sources, args)
    if args.source_id:
        wanted = set(args.source_id)
        selected = [source for source in selected if source.source_id in wanted]
    if args.limit:
        selected = selected[: args.limit]
    return selected


def candidate_urls(source: Source) -> list[str]:
    url = source.url
    parsed = urlparse(url)
    candidates = [url]

    if parsed.netloc == "arxiv.org" and parsed.path.startswith("/abs/"):
        arxiv_id = parsed.path.removeprefix("/abs/")
        candidates.insert(0, f"https://arxiv.org/pdf/{arxiv_id}")
        candidates.insert(0, f"https://arxiv.org/html/{arxiv_id}")

    if parsed.netloc == "docs.llamaindex.ai":
        raw_path = parsed.path
        raw_path = raw_path.replace("/en/stable/", "/python/framework/")
        raw_path = raw_path.rstrip("/") + "/index.md"
        candidates.insert(0, f"https://developers.llamaindex.ai{raw_path}")

    if parsed.netloc == "developers.llamaindex.ai" and not parsed.path.endswith("/index.md"):
        candidates.insert(0, url.rstrip("/") + "/index.md")

    if parsed.netloc == "docs.langchain.com":
        candidates.insert(0, url.rstrip("/") + ".md")

    if parsed.netloc == "github.com":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 5 and parts[2] == "blob":
            owner, repo, _, branch = parts[:4]
            rest = "/".join(parts[4:])
            candidates.insert(0, f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{rest}")
        elif len(parts) >= 2:
            owner, repo = parts[:2]
            candidates.extend(
                [
                    f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md",
                    f"https://raw.githubusercontent.com/{owner}/{repo}/master/README.md",
                ]
            )
    return dedupe(candidates)


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extension_for(url: str, content_type: str | None) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return ".pdf"
    if path.endswith(".md"):
        return ".md"
    if path.endswith(".txt"):
        return ".txt"
    if content_type:
        content_type = content_type.lower()
        if "pdf" in content_type:
            return ".pdf"
        if "markdown" in content_type:
            return ".md"
        if "text/plain" in content_type:
            return ".txt"
        if "html" in content_type:
            return ".html"
    return ".html"


def download_source(source: Source, source_dir: Path, timeout: int, force: bool) -> tuple[Path, str]:
    existing = list(source_dir.glob("raw.*"))
    if existing and not force:
        return preferred_raw_file(existing), "cached"

    source_dir.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for url in candidate_urls(source):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                content = response.read()
                content_type = response.headers.get("content-type")
            ext = extension_for(url, content_type)
            raw_path = source_dir / f"raw{ext}"
            raw_path.write_bytes(content)
            (source_dir / "download_url.txt").write_text(url, encoding="utf-8")
            return raw_path, "downloaded"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
    raise RuntimeError(last_error or "unknown download error")


def preferred_raw_file(paths: Iterable[Path]) -> Path:
    order = {".md": 0, ".txt": 1, ".html": 2, ".htm": 3, ".pdf": 4}
    return sorted(paths, key=lambda path: (order.get(path.suffix.lower(), 99), path.name))[0]


def convert_to_markdown(
    source: Source,
    raw_path: Path,
    markdown_path: Path,
    markitdown_convert: Path,
    no_convert: bool,
    force: bool,
    convert_timeout: int,
) -> tuple[bool, str]:
    if markdown_path.exists() and not force:
        return True, "cached"

    if raw_path.suffix.lower() in {".md", ".txt"}:
        body = raw_path.read_text(encoding="utf-8", errors="replace")
        markdown_path.write_text(wrap_markdown(source, body), encoding="utf-8")
        return True, "copied"

    if raw_path.suffix.lower() in {".html", ".htm"}:
        parser = BasicHTMLToMarkdown()
        parser.feed(raw_path.read_text(encoding="utf-8", errors="replace"))
        markdown_path.write_text(wrap_markdown(source, parser.markdown()), encoding="utf-8")
        return True, "html_fallback"

    if not no_convert and markitdown_convert.exists():
        try:
            result = subprocess.run(
                [sys.executable, str(markitdown_convert), str(raw_path), "-o", str(markdown_path)],
                text=True,
                capture_output=True,
                check=False,
                timeout=convert_timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"markitdown timeout after {convert_timeout}s"
        if result.returncode == 0 and markdown_path.exists():
            body = markdown_path.read_text(encoding="utf-8", errors="replace")
            markdown_path.write_text(wrap_markdown(source, body), encoding="utf-8")
            return True, "markitdown"
        markitdown_error = (result.stderr or result.stdout).strip()
    else:
        markitdown_error = "markitdown disabled or missing"

    return False, markitdown_error


def wrap_markdown(source: Source, body: str) -> str:
    if source.extract_symbol:
        body = extract_python_symbol(body, source.extract_symbol)
    body = clean_document_body(source, body)
    metadata = {
        "source_id": source.source_id,
        "title": source.title,
        "category": source.category,
        "priority": source.priority,
        "source_type": source.source_type,
        "url": source.url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    header = ["---"]
    for key, value in metadata.items():
        safe = str(value).replace("\n", " ").replace('"', "'")
        header.append(f'{key}: "{safe}"')
    header.extend(["---", "", f"# {source.title}", "", f"Source: {source.url}", ""])
    return "\n".join(header) + body + "\n"


def extract_python_symbol(body: str, symbol: str) -> str:
    """Return selected Python definitions from a pinned source file."""
    try:
        module = ast.parse(body)
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse Python source while extracting {symbol}: {exc}") from exc

    segments = []
    for selector in (item.strip() for item in symbol.split(",")):
        if not selector:
            continue
        nodes = module.body
        node = None
        for part in selector.split("."):
            node = next(
                (
                    candidate
                    for candidate in nodes
                    if isinstance(candidate, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and candidate.name == part
                ),
                None,
            )
            if node is None:
                raise ValueError(f"Python symbol not found: {selector}")
            nodes = getattr(node, "body", [])

        assert node is not None
        start_line = min(
            [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
        )
        end_line = node.end_lineno
        if end_line is None:
            raise ValueError(f"Python parser did not report an end line for {selector}")
        definition = "\n".join(body.splitlines()[start_line - 1 : end_line]).strip()
        segments.append(f"# Official symbol: {selector}\n{definition}")

    if not segments:
        raise ValueError("At least one Python symbol is required")
    return "\n\n".join(segments)


def clean_document_body(source: Source, body: str) -> str:
    body = normalize_markdown(body)
    body = strip_embedded_frontmatter(body)
    body = strip_before_first_content_heading(source, body)
    body = strip_tail_boilerplate(body)
    body = clean_boilerplate_lines(body)
    body = clean_heading_anchors(body)
    body = strip_leading_h1(body)
    return normalize_markdown(body)


def strip_embedded_frontmatter(body: str) -> str:
    return re.sub(r"(?s)^---\n.*?\n---\n+", "", body, count=1).strip()


def strip_before_first_content_heading(source: Source, body: str) -> str:
    if source.source_type == "paper" and "\n# " not in body:
        return body

    match = re.search(r"(?m)^#\s+(.+?)\s*$", body)
    if not match:
        return body

    heading = match.group(1).strip().lower()
    noisy_headings = {
        "documentation index",
        "docs by langchain",
        "developer documentation",
    }
    if heading in noisy_headings:
        next_match = re.search(r"(?m)^#\s+(.+?)\s*$", body[match.end() :])
        if next_match:
            offset = match.end()
            return body[offset + next_match.start() :]
        return body
    return body[match.start() :]


def strip_tail_boilerplate(body: str) -> str:
    markers = [
        "\n[Previous",
        "\nPrevious\n",
        "\n## On this page",
        "\nWas this page helpful?",
        "\n[Edit this page",
        "\nEdit this page",
        "\nPowered by",
        "\nNote for AI agents:",
    ]
    cutoff = len(body)
    for marker in markers:
        idx = body.find(marker)
        if idx != -1 and idx > len(body) * 0.35:
            cutoff = min(cutoff, idx)
    return body[:cutoff]


def clean_boilerplate_lines(body: str) -> str:
    drop_patterns = [
        r"^\[Skip to .*?\]\(#.*?\)$",
        r"^Search\.\.\.$",
        r"^Search `?CtrlK`?$",
        r"^⌘K$",
        r"^Auto$",
        r"^Light$",
        r"^Dark$",
        r"^Copy page(\s*Copy page)?\s*$",
        r"^Copy Markdown$",
        r"^On this page\s*$",
        r"^\*\*Copy Markdown\*\*$",
        r"^\*\*View as Markdown\*\*$",
        r"^Open in \*\*.*?\*\*$",
        r"^Install in .*$",
        r"^Copy Claude Code command$",
        r"^Copy Codex config$",
        r"^Documentation search MCP$",
        r"^Copy MCP URL$",
        r"^Learn more$",
        r"^Report GitHub Issue$",
        r"^\[Button: .*?\]$",
        r"^!\[.*?\]\(.*?\)$",
        r"^\[Section titled .*?\]\(#.*?\)$",
        r"^</?(Note|Tip|Warning|Info|Card|CardGroup).*>$",
        r"^(Tip|Note|Warning|Info)$",
    ]
    compiled = [re.compile(pattern, flags=re.IGNORECASE) for pattern in drop_patterns]
    kept: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if any(pattern.search(stripped) for pattern in compiled):
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_heading_anchors(body: str) -> str:
    body = re.sub(r"(?m)^(#{1,6})\s+\[.*?\]\(#.*?\)\s*", r"\1 ", body)
    body = re.sub(r"(?m)^\[.*?†(.+?)\]\s*$", r"\1", body)
    return body


def strip_leading_h1(body: str) -> str:
    return re.sub(r"(?s)^#\s+.+?\n+", "", body, count=1).strip()


def normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def write_metadata(source: Source, source_dir: Path, raw_path: Path | None, status: dict[str, str]) -> None:
    payload = {
        "source_id": source.source_id,
        "priority": source.priority,
        "category": source.category,
        "title": source.title,
        "source_type": source.source_type,
        "url": source.url,
        "ingest_first": source.ingest_first,
        "why": source.why,
        "notes": source.notes,
        "extract_symbol": source.extract_symbol,
        "raw_file": str(raw_path) if raw_path else None,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (source_dir / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_documents_jsonl(
    raw_dir: Path,
    processed_dir: Path,
    selected: list[Source],
    *,
    require_complete: bool = False,
) -> int:
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "documents.jsonl"
    temp_path = processed_dir / "documents.jsonl.tmp"
    records: list[dict[str, str]] = []
    missing: list[str] = []
    for source in selected:
        markdown_path = raw_dir / source.source_id / "document.md"
        if not markdown_path.exists():
            missing.append(source.source_id)
            continue
        text = markdown_path.read_text(encoding="utf-8", errors="replace")
        records.append(
            {
                "doc_id": source.source_id,
                "source_id": source.source_id,
                "title": source.title,
                "category": source.category,
                "priority": source.priority,
                "source_type": source.source_type,
                "url": source.url,
                "local_path": str(markdown_path),
                "text_hash": text_hash(text),
                "text": text,
            }
        )

    if require_complete and missing:
        raise FileNotFoundError(
            "Refusing to replace documents.jsonl because normalized sources are missing: "
            + ", ".join(missing)
        )

    try:
        with temp_path.open("w", encoding="utf-8") as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        temp_path.replace(out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return len(records)


def main() -> None:
    args = parse_args()
    if args.force:
        args.skip_existing = False

    sources = read_sources(args.manifest)
    eligible_sources = select_eligible_sources(sources, args)
    selected = select_sources(sources, args)

    if args.source_id and not selected:
        raise SystemExit("No requested source ids matched the active priority and ingest filters")

    print(f"manifest: {args.manifest}", flush=True)
    print(f"selected sources: {len(selected)}", flush=True)
    for source in selected:
        print(f"- {source.source_id} [{source.priority}/{source.category}] {source.title}", flush=True)

    if args.dry_run:
        print("dry-run: no files downloaded or converted", flush=True)
        return

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    report: list[dict[str, str]] = []
    for idx, source in enumerate(selected, start=1):
        source_dir = args.raw_dir / source.source_id
        markdown_path = source_dir / "document.md"
        raw_path: Path | None = None
        status: dict[str, str] = {"download": "pending", "convert": "pending"}
        try:
            print(f"[{idx}/{len(selected)}] {source.source_id}: downloading", flush=True)
            raw_path, download_status = download_source(source, source_dir, args.timeout, args.force)
            status["download"] = download_status
            print(f"[{idx}/{len(selected)}] {source.source_id}: converting {raw_path.name}", flush=True)
            ok, convert_status = convert_to_markdown(
                source,
                raw_path,
                markdown_path,
                args.markitdown_convert,
                args.no_convert,
                args.force,
                args.convert_timeout,
            )
            status["convert"] = convert_status if ok else f"failed: {convert_status}"
            write_metadata(source, source_dir, raw_path, status)
            print(f"[{idx}/{len(selected)}] {source.source_id}: {status['download']}, {status['convert']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report per-source failures and continue.
            status["error"] = f"{type(exc).__name__}: {exc}"
            source_dir.mkdir(parents=True, exist_ok=True)
            write_metadata(source, source_dir, raw_path, status)
            print(f"[{idx}/{len(selected)}] {source.source_id}: failed: {status['error']}", flush=True)
        report.append({"source_id": source.source_id, **status})
        if idx < len(selected):
            time.sleep(args.sleep)

    document_sources = eligible_sources if args.source_id else selected
    docs_count = build_documents_jsonl(
        args.raw_dir,
        args.processed_dir,
        document_sources,
        require_complete=bool(args.source_id),
    )
    report_path = args.processed_dir / "fetch_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"documents: {docs_count}", flush=True)
    print(f"documents_jsonl: {args.processed_dir / 'documents.jsonl'}", flush=True)
    print(f"report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
