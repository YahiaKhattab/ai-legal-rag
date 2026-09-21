from __future__ import annotations

from legal_rag.chat_context.models import ChatMessage, ConversationTopic
from legal_rag.query.ollama_client import OllamaGenerationClient


class TopicTracker:
    """Extract and maintain the active conversation topic."""

    def __init__(self, client: OllamaGenerationClient) -> None:
        self._client = client

    def extract_topic(
        self,
        *,
        query: str,
        previous_topic: ConversationTopic | None = None,
        previous_messages: list[ChatMessage] | None = None,
    ) -> str:
        """Extract a concise but context-preserving legal topic."""

        query = query.strip()

        if not query:
            raise ValueError("query must not be empty")

        previous_messages = previous_messages or []

        previous_topic_text = (
            previous_topic.topic
            if previous_topic is not None
            else "None"
        )

        recent_context = self._build_recent_context(previous_messages)

        prompt = f"""
You are a legal conversation topic tracker.

Your ONLY task is to identify the CURRENT legal topic of the user's
question.

Do NOT answer the question.

CURRENT USER QUESTION:
{query}

CURRENT PREVIOUS TOPIC:
{previous_topic_text}

RECENT CONVERSATION:
{recent_context}

CRITICAL TOPIC RULE:

If the current question is a follow-up question, DO NOT replace the
previous topic with only the short phrase from the current question.

Instead, preserve the previous legal subject and combine it with the
specific aspect being asked about.

For example:

Previous topic:
نسبة القبول المجاني من نسبة إشغال المؤسسة لرعاية المسن

Current question:
وما الحد الأدنى؟

Correct topic:
الحد الأدنى المرتبط بنسبة القبول المجاني من نسبة إشغال المؤسسة لرعاية المسن

INCORRECT:
الحد الأدنى

Another example:

Previous topic:
شروط تقديم الخدمات الصحية داخل المؤسسة لرعاية المسن

Current question:
وما الجهة المختصة؟

Correct topic:
الجهة المختصة المتعلقة بتقديم الخدمات الصحية داخل المؤسسة لرعاية المسن

INCORRECT:
الجهة المختصة

GENERAL RULES:

1. Identify the specific legal subject currently being discussed.

2. Prefer the current question together with the immediately previous
   topic.

3. For independent questions, create a new topic.

4. For follow-up questions, preserve the previous topic.

5. Never allow a short follow-up phrase such as:
   - الحد الأدنى
   - الحد الأقصى
   - الجهة المختصة
   - المدة
   - النسبة
   - الشرط
   - المبلغ
   - هذا
   - هذه
   - ذلك
   - تلك

   to become the complete topic by itself when a previous topic exists.

6. If the previous topic contains an important legal identifier,
   preserve it when relevant, including:
   - percentages
   - amounts
   - durations
   - authorities
   - services
   - requirements
   - conditions
   - legal obligations

7. Keep the topic concise, but make it understandable without the
   rest of the conversation.

8. Do not invent legal facts.

9. Do not answer the question.

10. Return ONLY the topic.

11. Do not return:
    - explanations
    - analysis
    - answers
    - JSON
    - bullets
    - quotation marks
    - citation markers

EXAMPLES:

Previous topic:
شروط تقديم الخدمات الصحية داخل المؤسسة لرعاية المسن

Current question:
وما الجهة المختصة؟

Topic:
الجهة المختصة المتعلقة بتقديم الخدمات الصحية داخل المؤسسة لرعاية المسن

Previous topic:
نسبة القبول المجاني من نسبة إشغال المؤسسة لرعاية المسن

Current question:
وما الحد الأدنى؟

Topic:
الحد الأدنى المرتبط بنسبة القبول المجاني من نسبة إشغال المؤسسة لرعاية المسن

Previous topic:
نسبة القبول المجاني من نسبة إشغال المؤسسة لرعاية المسن

Current question:
وما النسبة؟

Topic:
النسبة المرتبطة بالقبول المجاني من نسبة إشغال المؤسسة لرعاية المسن

Previous topic:
شروط تقديم الخدمات الصحية داخل المؤسسة لرعاية المسن

Current question:
هل يشترط الحصول على موافقة؟

Topic:
شروط الحصول على موافقة لتقديم الخدمات الصحية داخل المؤسسة لرعاية المسن

Return ONLY the topic.
""".strip()

        topic = self._client.generate(
            prompt,
            temperature=0.0,
            max_tokens=80,
        ).strip()

        if not topic:
            return self._fallback_topic(
                query=query,
                previous_topic=previous_topic,
            )

        if len(topic) > 300:
            return self._fallback_topic(
                query=query,
                previous_topic=previous_topic,
            )

        return topic

    @staticmethod
    def _build_recent_context(
        messages: list[ChatMessage],
    ) -> str:
        if not messages:
            return "None"

        parts: list[str] = []

        for message in messages[-6:]:
            role = (
                "USER"
                if message.role == "user"
                else "ASSISTANT"
            )

            parts.append(
                f"{role}:\n{message.content.strip()}"
            )

        return "\n\n".join(parts)

    @staticmethod
    def _fallback_topic(
        *,
        query: str,
        previous_topic: ConversationTopic | None,
    ) -> str:
        """
        Preserve the previous topic when the LLM fails to produce
        a usable topic.
        """

        if previous_topic is not None:
            return previous_topic.topic

        return query