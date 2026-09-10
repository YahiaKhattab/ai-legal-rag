from __future__ import annotations

from legal_rag.chat_context.models import ChatMessage
from legal_rag.query.ollama_client import OllamaGenerationClient


class QueryContextualizer:
    """Rewrite follow-up questions into standalone legal retrieval queries."""

    def __init__(self, client: OllamaGenerationClient) -> None:
        self._client = client

    def contextualize(
        self,
        query: str,
        history: list[ChatMessage],
    ) -> str:
        """Return a standalone version of the current query."""

        query = query.strip()

        if not query:
            raise ValueError("query must not be empty")

        if not history:
            return query

        previous_user_questions = [
            message.content.strip()
            for message in history
            if message.role == "user" and message.content.strip()
        ][-3:]

        if not previous_user_questions:
            return query

        # The oldest question in the recent window establishes the topic.
        # This prevents repeated follow-up questions from becoming the topic
        # themselves.
        topic_question = previous_user_questions[0]

        prompt = f"""
You are a legal search query rewriter.

The user is having a conversation about a legal topic.

Original topic question:
{topic_question}

Recent user questions:
{chr(10).join(previous_user_questions)}

Current user question:
{query}

Task:
Rewrite the current user question into one standalone legal retrieval query.

Rules:
- Resolve follow-up references using the original topic question.
- Preserve the legal subject from the original topic question when the
  current question depends on it.
- "وما الحد الأقصى؟" is a follow-up question, not a new topic.
- Do not answer the question.
- Do not invent legal facts.
- Do not invent a duration, number, article, law, date, limit, condition,
  or other legal detail that was not stated.
- Keep the meaning of the current question unchanged.
- Return ONLY the rewritten question.

Example:

Original topic question:
ما هي شروط عقد الإيجار؟

Current user question:
وما الحد الأقصى؟

Correct rewritten question:
ما هو الحد الأقصى المتعلق بعقد الإيجار؟

Return ONLY the rewritten question.
""".strip()

        rewritten = self._client.generate(
            prompt,
            temperature=0.0,
            max_tokens=64,
        ).strip()

        if not rewritten:
            return query

        return rewritten