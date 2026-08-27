"""Context assembly for the RAG web workbench.

Conversation memory helps interpret follow-up questions, but retrieved evidence
is the only factual source that the answer may cite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.retrieval import RetrievedChunk
from src.rag_security import assess_evidence_security, security_prompt_rules


@dataclass
class ContextSection:
    name: str
    role: str
    char_count: int
    content: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "char_count": self.char_count,
            "content": self.content,
        }


@dataclass
class AssembledContext:
    prompt_context: str
    evidence_context: str
    sections: list[ContextSection]
    used_chars: int
    max_chars: int
    security: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_context": self.prompt_context,
            "evidence_context": self.evidence_context,
            "sections": [section.as_dict() for section in self.sections],
            "used_chars": self.used_chars,
            "max_chars": self.max_chars,
            "security": self.security,
        }


def assemble_context(
    query: str,
    retrieved: list[RetrievedChunk],
    memory_context: str = "",
    max_chars: int = 9000,
) -> AssembledContext:
    sections: list[ContextSection] = []
    used = 0

    memory_context = memory_context.strip()
    if memory_context:
        memory_block = (
            "对话记忆仅用于理解用户当前问题中的指代、偏好和上下文，不得作为事实来源引用。\n"
            f"{memory_context}"
        )
        sections.append(
            ContextSection(
                name="conversation_memory",
                role="memory",
                char_count=len(memory_block),
                content=memory_block,
            )
        )
        used += len(memory_block)

    evidence_blocks: list[str] = []
    included_retrieved: list[RetrievedChunk] = []
    for idx, item in enumerate(retrieved, start=1):
        metadata = item.metadata
        header = (
            f"[{idx}] title={metadata.get('title')} | "
            f"section={metadata.get('heading_path')} | "
            f"url={metadata.get('url')} | "
            f"aspect={item.aspect or 'general'} | "
            f"chunk_id={item.chunk_id}"
        )
        block = f'<retrieved_source id="{idx}">\n{header}\n{item.document.strip()}\n</retrieved_source>'
        if used + len(block) > max_chars and evidence_blocks:
            break
        evidence_blocks.append(block)
        included_retrieved.append(item)
        sections.append(
            ContextSection(
                name=f"source_{idx}",
                role="retrieved_evidence",
                char_count=len(block),
                content=block,
            )
        )
        used += len(block)

    evidence_context = "\n\n---\n\n".join(evidence_blocks)
    prompt_parts = []
    if memory_context:
        prompt_parts.append("【对话记忆】\n" + sections[0].content)
    prompt_parts.append("【检索资料】\n" + evidence_context)
    return AssembledContext(
        prompt_context="\n\n".join(prompt_parts),
        evidence_context=evidence_context,
        sections=sections,
        used_chars=used,
        max_chars=max_chars,
        security=assess_evidence_security(included_retrieved),
    )


def build_answer_prompt(
    query: str,
    assembled: AssembledContext,
    answer_requirements: list[str] | None = None,
) -> str:
    requirements_text = build_requirements_text(answer_requirements or [])
    security_rules = security_prompt_rules(assembled.security)
    return f"""你是一个大模型工程知识库助手。请只根据“检索资料”回答问题。

领域规则：
- 本项目中，RAG 固定指 Retrieval-Augmented Generation，即检索增强生成。
- 对话记忆只能用于理解用户当前问题，不得作为事实来源。
- 事实判断必须来自检索资料，并标注检索资料编号。
- 如果检索资料没有支持某个判断，不要写这个判断。
{requirements_text}

{security_rules}

强制输出格式：
结论：
- 一句话直接回答用户问题，并在句末标注来源编号，例如 [1]。

要点：
1. 一个关键点。必须在句末标注来源编号，例如 [1]。
2. 一个关键点。必须在句末标注来源编号，例如 [2]。

来源：
[1] 标题 - URL
[2] 标题 - URL

要求：
1. 用中文回答。
2. 每个关键结论和每个要点都必须标注来源编号。
3. 来源编号只能使用检索资料中已有的 [1]、[2]、[3] 等编号。
4. 如果资料不足，不要编造，直接说“根据当前知识库资料不足以回答”。
5. 不要引用资料外的事实。
6. 不要复制资料里的 Markdown 锚点、HTML 标签、代码装饰或页面导航文本。
7. 每个要点尽量说明“为什么有用/解决什么问题”。
8. 最后必须输出“来源：”列表。

用户问题：
{query}

上下文：
{assembled.prompt_context}

请给出答案：
"""


def build_requirements_text(answer_requirements: list[str]) -> str:
    if not answer_requirements:
        return ""
    lines = ["", "本轮问题需要覆盖这些子问题："]
    for idx, requirement in enumerate(answer_requirements, start=1):
        lines.append(f"{idx}. {requirement}")
    lines.append("如果某个子问题没有足够检索资料支持，请在答案中明确说明资料不足。")
    return "\n".join(lines)
