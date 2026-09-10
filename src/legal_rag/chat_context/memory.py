from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from qdrant_client import QdrantClient, models

from legal_rag.chat_context.models import ChatMessage


class ChatMemoryStore:
    """Persistent conversation memory backed by Qdrant."""

    COLLECTION_NAME = "chat_memory"
    VECTOR_SIZE = 1

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ) -> None:
        self._client = QdrantClient(
            url=url,
            api_key=api_key,
        )

    def ensure_collection(self) -> None:
        """Create the chat memory collection if it does not exist."""
        collections = self._client.get_collections().collections
        existing_names = {collection.name for collection in collections}

        if self.COLLECTION_NAME in existing_names:
            return

        self._client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=self.VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

        self._client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name="session_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        self._client.create_payload_index(
            collection_name=self.COLLECTION_NAME,
            field_name="timestamp",
            field_schema=models.PayloadSchemaType.DATETIME,
        )

    def save_message(
        self,
        session_id: UUID,
        role: str,
        content: str,
    ) -> ChatMessage:
        """Persist one conversation message."""
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")

        content = content.strip()
        if not content:
            raise ValueError("content must not be empty")

        message = ChatMessage(
            session_id=session_id,
            message_id=uuid4(),
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc),
        )

        self._client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(message.message_id),
                    vector=[0.0],
                    payload=message.to_payload(),
                )
            ],
        )

        return message

    def get_recent_messages(
        self,
        session_id: UUID,
        limit: int = 10,
    ) -> list[ChatMessage]:
        """Return the most recent messages for a session."""
        if limit < 1:
            raise ValueError("limit must be greater than zero")

        points, _ = self._client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(
                            value=str(session_id),
                        ),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        messages: list[ChatMessage] = []

        for point in points:
            payload = point.payload or {}

            messages.append(
                ChatMessage(
                    session_id=UUID(str(payload["session_id"])),
                    message_id=UUID(str(payload["message_id"])),
                    role=payload["role"],
                    content=str(payload["content"]),
                    timestamp=datetime.fromisoformat(
                        str(payload["timestamp"])
                    ),
                )
            )

        messages.sort(key=lambda message: message.timestamp)

        return messages[-limit:]