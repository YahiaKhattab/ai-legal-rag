from __future__ import annotations

import re

from legal_rag.chat_context.models import ChatMessage
from legal_rag.observability.tracing import attributes, traced
from legal_rag.query.ollama_client import OllamaGenerationClient


class QueryContextualizer:
    """Rewrite follow-up questions into standalone legal retrieval queries."""

    _FOLLOW_UP_PATTERNS = (
        r"^وما\b",
        r"^وما هو\b",
        r"^وما هي\b",
        r"^وماذا\b",
        r"^وهل\b",
        r"^ومن\b",
        r"^وأين\b",
        r"^ومتى\b",
        r"^وكيف\b",
        r"^وكم\b",
        r"^وما الحد\b",
        r"^ما الحد\b",
        r"^وهذا\b",
        r"^وهذه\b",
        r"^ذلك\b",
        r"^تلك\b",
        r"^هذه\b",
        r"^هذا\b",
        r"^المذكور\b",
        r"^المذكورة\b",
        r"^نفس\b",
        r"^هل يجوز\b",
        r"^هل يمكن\b",
        r"^وماذا عن\b",
    )

    _REFERENCE_PATTERNS = (
        r"\bذلك\b",
        r"\bتلك\b",
        r"\bهذا\b",
        r"\bهذه\b",
        r"\bهذه النسبة\b",
        r"\bهذا الشرط\b",
        r"\bهذه المدة\b",
        r"\bهذا المبلغ\b",
        r"\bهذه المادة\b",
        r"\bالمذكور\b",
        r"\bالمذكورة\b",
        r"\bالمذكور أعلاه\b",
        r"\bالمذكورة أعلاه\b",
        r"\bنفس\b",
    )

    _ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"

    _DIGIT_TRANSLATION = str.maketrans(
        _ARABIC_DIGITS,
        "0123456789",
    )

    def __init__(self, client: OllamaGenerationClient) -> None:
        self._client = client

    @traced("query.contextualize")
    def contextualize(
        self,
        query: str,
        history: list[ChatMessage],
        current_topic: str | None = None,
    ) -> str:
        """Return a standalone version of a follow-up query."""

        query = query.strip()
        attributes(**{
            "rag.history_count": len(history),
            "rag.input_characters": len(query),
            "rag.query_changed": False,
        })

        if not query:
            raise ValueError("query must not be empty")

        current_topic = (
            current_topic.strip()
            if current_topic is not None
            else None
        )

        if not history and not current_topic:
            return query

        if not self._looks_like_follow_up(query, current_topic):
            return query

        previous_user_question, previous_assistant_answer = (
            self._get_immediate_previous_turn(history)
        )

        if not previous_user_question and not current_topic:
            return query

        earlier_messages = self._get_earlier_context(
            history=history,
            previous_user_question=previous_user_question,
        )

        context = self._build_context(
            current_topic=current_topic,
            previous_user_question=previous_user_question,
            previous_assistant_answer=previous_assistant_answer,
            earlier_messages=earlier_messages,
        )

        protected_identifiers = self._extract_legal_identifiers(
            current_topic=current_topic,
            previous_user_question=previous_user_question,
            previous_assistant_answer=previous_assistant_answer,
        )

        protected_identifier_text = (
            "\n".join(
                f"- {identifier}"
                for identifier in protected_identifiers
            )
            if protected_identifiers
            else "None"
        )

        prompt = f"""
You are a legal search query rewriter.

Your ONLY task is to rewrite a follow-up legal question into ONE
standalone legal retrieval query.

The conversation history is context only.
It is NOT legal evidence.
Do not answer the question.

CURRENT USER QUESTION:
{query}

CONVERSATION CONTEXT:

{context}

PROTECTED LEGAL IDENTIFIERS:

{protected_identifier_text}

CRITICAL TOPIC RULE:

The CURRENT ACTIVE TOPIC is the PRIMARY topic for the current
follow-up question.

If an active topic is provided, use it as the main subject of the
rewritten query.

Do NOT switch to an older topic from the conversation when the current
active topic clearly establishes the subject.

CRITICAL FOLLOW-UP RULE:

Resolve a follow-up using this priority:

1. CURRENT ACTIVE TOPIC
2. IMMEDIATELY PREVIOUS USER QUESTION
3. IMMEDIATELY PREVIOUS ASSISTANT ANSWER
4. EARLIER CONVERSATION

Do NOT use an older conversation topic when the current active topic
clearly establishes the subject.

CRITICAL LEGAL IDENTIFIER RULE:

Protected legal identifiers must NEVER be changed.

If the context contains ٤٪, do not produce ٣٪ or ٥٪.

If the current question refers to a protected identifier, preserve it
exactly in the rewritten query.

EXAMPLES:

Current topic:
نسبة القبول المجاني من نسبة إشغال المؤسسة لرعاية المسن

Protected identifier:
٤٪

Current question:
وما الحد الأدنى؟

Correct:
ما الحد الأدنى المرتبط بنسبة ٤٪ من نسبة إشغال المؤسسة بالمجان؟

Current question:
وهل توجد حالات استثناء؟

Correct:
هل توجد حالات استثناء مرتبطة بنسبة ٤٪ من نسبة إشغال المؤسسة بالمجان؟

Incorrect:
هل توجد حالات استثناء مرتبطة بنسبة ٣٪؟

Current topic:
الجهة المختصة بتقديم الخدمات الصحية داخل المؤسسة لرعاية المسن

Current question:
ما هي الجهة المختصة؟

Correct:
ما هي الجهة المختصة بتقديم الخدمات الصحية داخل المؤسسة لرعاية المسن؟

GENERAL RULES:

1. Keep the current legal topic.
2. Preserve important legal values and terms.
3. Never invent legal facts.
4. Never change a protected number or percentage.
5. Do not answer the question.
6. Keep the rewritten query concise.
7. The result must remain a QUESTION.
8. Return ONLY the rewritten query.
9. Do not return explanations, JSON, bullets, or quotation marks.

Return ONLY the rewritten query.
""".strip()

        rewritten = self._client.generate(
            prompt,
            temperature=0.0,
            max_tokens=96,
        ).strip()

        validated = self._validate_rewrite(
            original_query=query,
            rewritten_query=rewritten,
        )

        if self._contains_conflicting_identifier(
            rewritten_query=validated,
            original_query=query,
            protected_identifiers=protected_identifiers,
        ):
            return self._build_safe_fallback_query(
                query=query,
                current_topic=current_topic,
                protected_identifiers=protected_identifiers,
            )

        if (
            protected_identifiers
            and self._looks_like_identifier_dependent_follow_up(query)
            and not self._contains_any_protected_identifier(
                validated,
                protected_identifiers,
            )
        ):
            return self._build_safe_fallback_query(
                query=query,
                current_topic=current_topic,
                protected_identifiers=protected_identifiers,
            )

        return validated

    @classmethod
    def _looks_like_follow_up(
        cls,
        query: str,
        current_topic: str | None = None,
    ) -> bool:
        normalized = " ".join(query.split())

        for pattern in cls._FOLLOW_UP_PATTERNS:
            if re.search(pattern, normalized):
                return True

        for pattern in cls._REFERENCE_PATTERNS:
            if re.search(pattern, normalized):
                return True

        # A short generic question can be a follow-up even when it does
        # not contain an explicit reference word such as "هذا" or "ما الحد".
        #
        # When an active topic exists, topic-dependent wording is a strong
        # signal that the question depends on the current legal subject.
        if current_topic:
            topic_dependent_patterns = (
                r"\bالجهة\b",
                r"\bالسلطة\b",
                r"\bالحد الأدنى\b",
                r"\bالحد الأقصى\b",
                r"\bالنسبة\b",
                r"\bالمبلغ\b",
                r"\bالمدة\b",
                r"\bالشرط\b",
                r"\bالشروط\b",
                r"\bالاستثناء\b",
                r"\bالاستثناءات\b",
                r"\bالحكم\b",
                r"\bالعقوبة\b",
            )

            if len(normalized.split()) <= 8 and any(
                re.search(pattern, normalized)
                for pattern in topic_dependent_patterns
            ):
                return True

        return False

    @staticmethod
    def _looks_like_identifier_dependent_follow_up(
        query: str,
    ) -> bool:
        normalized = " ".join(query.split())

        patterns = (
            r"\bالحد الأدنى\b",
            r"\bالحد الأقصى\b",
            r"\bالنسبة\b",
            r"\bالمبلغ\b",
            r"\bالمدة\b",
            r"\bاستثناء\b",
            r"\bاستثناءات\b",
            r"\bالشرط\b",
            r"\bالشروط\b",
            r"\bالجهة\b",
            r"\bالسلطة\b",
        )

        return any(
            re.search(pattern, normalized)
            for pattern in patterns
        )

    @staticmethod
    def _get_immediate_previous_turn(
        history: list[ChatMessage],
    ) -> tuple[str | None, str | None]:
        messages = [
            message
            for message in history
            if message.content.strip()
        ]

        if not messages:
            return None, None

        previous_user_index: int | None = None

        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                previous_user_index = index
                break

        if previous_user_index is None:
            return None, None

        previous_user_question = (
            messages[previous_user_index].content.strip()
        )

        previous_assistant_answer: str | None = None

        if previous_user_index + 1 < len(messages):
            next_message = messages[previous_user_index + 1]

            if next_message.role == "assistant":
                previous_assistant_answer = (
                    next_message.content.strip()
                )

        return (
            previous_user_question,
            previous_assistant_answer,
        )

    @staticmethod
    def _get_earlier_context(
        *,
        history: list[ChatMessage],
        previous_user_question: str | None,
    ) -> list[ChatMessage]:
        if not previous_user_question:
            return []

        messages = [
            message
            for message in history
            if message.content.strip()
        ]

        previous_user_index: int | None = None

        for index in range(len(messages) - 1, -1, -1):
            if (
                messages[index].role == "user"
                and messages[index].content.strip()
                == previous_user_question
            ):
                previous_user_index = index
                break

        if previous_user_index is None:
            return []

        return messages[
            max(0, previous_user_index - 4):
            previous_user_index
        ]

    @staticmethod
    def _build_context(
        *,
        current_topic: str | None,
        previous_user_question: str | None,
        previous_assistant_answer: str | None,
        earlier_messages: list[ChatMessage],
    ) -> str:
        parts: list[str] = []

        if current_topic:
            parts.append(
                "CURRENT ACTIVE TOPIC:\n"
                f"{current_topic}"
            )

        if previous_user_question:
            parts.append(
                "IMMEDIATELY PREVIOUS USER QUESTION:\n"
                f"{previous_user_question}"
            )

        if previous_assistant_answer:
            parts.append(
                "IMMEDIATELY PREVIOUS ASSISTANT ANSWER:\n"
                f"{previous_assistant_answer}"
            )

        if earlier_messages:
            earlier_parts: list[str] = []

            for message in earlier_messages:
                role = (
                    "USER"
                    if message.role == "user"
                    else "ASSISTANT"
                )

                earlier_parts.append(
                    f"{role}:\n"
                    f"{message.content.strip()}"
                )

            parts.append(
                "EARLIER CONVERSATION:\n"
                + "\n\n".join(earlier_parts)
            )

        return "\n\n".join(parts) if parts else "None"

    @classmethod
    def _extract_legal_identifiers(
        cls,
        *,
        current_topic: str | None,
        previous_user_question: str | None,
        previous_assistant_answer: str | None,
    ) -> list[str]:
        sources = [
            current_topic or "",
            previous_user_question or "",
            previous_assistant_answer or "",
        ]

        combined_text = "\n".join(sources)

        # Citation markers such as [1], [2], [3] are retrieval metadata,
        # not legal identifiers. Remove them before extracting numbers.
        combined_text = re.sub(
            r"\[\s*\d+\s*\]",
            " ",
            combined_text,
        )

        identifiers: list[str] = []

        percentage_patterns = (
            r"(?<!\w)\d+(?:[.,]\d+)?\s*%",
            r"(?<!\w)[٠-٩]+(?:[.,][٠-٩]+)?\s*٪",
            r"(?<!\w)\d+(?:[.,]\d+)?\s*بالمائة",
            r"(?<!\w)[٠-٩]+(?:[.,][٠-٩]+)?\s*بالمائة",
        )

        for pattern in percentage_patterns:
            for match in re.finditer(
                pattern,
                combined_text,
            ):
                value = match.group(0).strip()

                if value not in identifiers:
                    identifiers.append(value)

        numeric_patterns = (
            r"\b\d+(?:[.,]\d+)?\b",
            r"[٠-٩]+(?:[.,][٠-٩]+)?",
        )

        for pattern in numeric_patterns:
            for match in re.finditer(
                pattern,
                combined_text,
            ):
                value = match.group(0).strip()

                if value not in identifiers:
                    identifiers.append(value)

        return identifiers

    @classmethod
    def _contains_conflicting_identifier(
        cls,
        *,
        rewritten_query: str,
        original_query: str,
        protected_identifiers: list[str],
    ) -> bool:
        if not protected_identifiers:
            return False

        rewritten_numbers = cls._extract_numeric_tokens(
            rewritten_query
        )

        original_numbers = cls._extract_numeric_tokens(
            original_query
        )

        protected_numbers = {
            cls._normalize_digits(identifier)
            for identifier in protected_identifiers
        }

        new_numbers = rewritten_numbers - original_numbers

        return any(
            number not in protected_numbers
            for number in new_numbers
        )

    @classmethod
    def _contains_any_protected_identifier(
        cls,
        text: str,
        protected_identifiers: list[str],
    ) -> bool:
        text_numbers = cls._extract_numeric_tokens(text)

        protected_numbers = {
            cls._normalize_digits(identifier)
            for identifier in protected_identifiers
        }

        return bool(
            text_numbers.intersection(protected_numbers)
        )

    @classmethod
    def _build_safe_fallback_query(
        cls,
        *,
        query: str,
        current_topic: str | None,
        protected_identifiers: list[str],
    ) -> str:
        if not current_topic:
            return query

        identifier = cls._select_best_identifier(
            protected_identifiers
        )

        if identifier:
            return (
                f"{query} المرتبط بـ {identifier} "
                f"في موضوع {current_topic}"
            )

        return f"{query} المرتبط بموضوع {current_topic}"

    @staticmethod
    def _select_best_identifier(
        identifiers: list[str],
    ) -> str | None:
        for identifier in identifiers:
            if "%" in identifier or "٪" in identifier:
                return identifier

        for identifier in identifiers:
            if "بالمائة" in identifier:
                return identifier

        return identifiers[0] if identifiers else None

    @classmethod
    def _extract_numeric_tokens(
        cls,
        text: str,
    ) -> set[str]:
        tokens: set[str] = set()

        for match in re.finditer(
            r"(?<!\w)(?:\d+(?:[.,]\d+)?|[٠-٩]+(?:[.,][٠-٩]+)?)(?!\w)",
            text,
        ):
            tokens.add(
                cls._normalize_digits(match.group(0))
            )

        return tokens

    @classmethod
    def _normalize_digits(
        cls,
        value: str,
    ) -> str:
        return value.translate(cls._DIGIT_TRANSLATION)

    @staticmethod
    def _validate_rewrite(
        *,
        original_query: str,
        rewritten_query: str,
    ) -> str:
        if not rewritten_query:
            return original_query

        rewritten_query = rewritten_query.strip()

        if len(rewritten_query) > 1000:
            return original_query

        forbidden_markers = (
            "```",
            "{",
            "}",
            '"answer"',
            '"evidence_ids"',
            "Here is",
            "Here’s",
            "الإجابة:",
            "الإجابة هي:",
            "الجواب:",
        )

        if any(
            marker.lower() in rewritten_query.lower()
            for marker in forbidden_markers
        ):
            return original_query

        if rewritten_query == original_query:
            return original_query
          
        attributes(**{
            "rag.query_changed": rewritten != query,
            "rag.output_characters": len(rewritten),
        })

        return rewritten_query
