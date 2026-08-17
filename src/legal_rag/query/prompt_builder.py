"""Builds a citation-aware, grounded generation prompt (FR-004).

Two requirements from FR-004 are enforced directly in the prompt template:
  - the answer must be a *concise* summary, not a re-dump of the chunks
  - every substantive claim must be traceable to a numbered source, and
    the model is told to say so explicitly when evidence is insufficient
    (this also protects against hallucinated citations).
"""
from __future__ import annotations

from legal_rag.query.models import Citation, RerankedChunk

_SYSTEM_AR = (
    "أنت مساعد قانوني يلخّص فقط بالاعتماد على المقاطع المرجعية المرفقة أدناه. "
    "لا تستخدم أي معلومة من خارج هذه المقاطع. اذكر رقم المصدر [n] بعد كل ادّعاء تقدّمه. "
    "اجعل الإجابة موجزة ومباشرة. إذا لم تكن المقاطع كافية للإجابة، صرّح بذلك بوضوح "
    "بدلاً من التخمين."
)
_SYSTEM_EN = (
    "You are a legal assistant. Answer strictly using the numbered source "
    "excerpts provided below and no outside knowledge. Cite the source "
    "number [n] after every claim you make. Keep the answer concise. If the "
    "excerpts do not contain enough information to answer, say so "
    "explicitly instead of guessing."
)
_SYSTEM_MIXED = _SYSTEM_EN + " Respond in the same language(s) as the question."


def _system_prompt(language: str) -> str:
    return {"ar": _SYSTEM_AR, "en": _SYSTEM_EN}.get(language, _SYSTEM_MIXED)


def build_grounded_prompt(
    query: str,
    chunks: list[RerankedChunk],
    language: str = "mixed",
) -> tuple[str, list[Citation]]:
    """Returns (prompt_text, citations).

    citations[i].marker == f"[{i+1}]" matches the numbering used inside the
    prompt, so the generation output and any citation-rendering UI stay in
    sync without re-parsing the model's answer.
    """
    citations: list[Citation] = []
    evidence_blocks: list[str] = []

    for i, chunk in enumerate(chunks, start=1):
        marker = f"[{i}]"
        citations.append(
            Citation(
                marker=marker,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source_file=chunk.source_file,
                section_title=chunk.section_title,
                page=chunk.page,
            )
        )
        locator = chunk.section_title or (f"page {chunk.page}" if chunk.page else "")
        header = f"{marker} Source: {chunk.source_file or chunk.document_id}"
        if locator:
            header += f" ({locator})"
        evidence_blocks.append(f"{header}\n{chunk.text.strip()}")

    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "(no relevant excerpts found)"

    prompt = (
        f"{_system_prompt(language)}\n\n"
        f"### Question / السؤال\n{query.strip()}\n\n"
        f"### Source excerpts / المقاطع المرجعية\n{evidence_text}\n\n"
        f"### Answer / الإجابة\n"
    )
    return prompt, citations
