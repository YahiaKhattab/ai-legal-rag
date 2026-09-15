from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from qdrant_client import QdrantClient, models

from legal_rag.chat_context.models import (
    ChatMessage,
    Conversation,
    ConversationTopic,
)


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
        collections = self._client.get_collections().collections
        existing_names = {
            collection.name
            for collection in collections
        }

        if self.COLLECTION_NAME not in existing_names:
            self._client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )

        self._ensure_payload_index(
            field_name="conversation_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._ensure_payload_index(
            field_name="timestamp",
            field_schema=models.PayloadSchemaType.DATETIME,
        )
        self._ensure_payload_index(
            field_name="created_at",
            field_schema=models.PayloadSchemaType.DATETIME,
        )
        self._ensure_payload_index(
            field_name="updated_at",
            field_schema=models.PayloadSchemaType.DATETIME,
        )
        self._ensure_payload_index(
            field_name="memory_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def _ensure_payload_index(
        self,
        *,
        field_name: str,
        field_schema: models.PayloadSchemaType,
    ) -> None:
        try:
            self._client.create_payload_index(
                collection_name=self.COLLECTION_NAME,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception:
            # The index may already exist.
            pass

    def create_conversation(
        self,
        title: str = "New conversation",
    ) -> Conversation:
        """Create a new independent conversation."""

        title = title.strip()

        if not title:
            title = "New conversation"

        now = datetime.now(timezone.utc)

        conversation = Conversation(
            conversation_id=uuid4(),
            title=title,
            created_at=now,
            updated_at=now,
        )

        self._client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(conversation.conversation_id),
                    vector=[0.0],
                    payload=conversation.to_payload(),
                )
            ],
        )

        return conversation

    def get_conversation(
        self,
        conversation_id: UUID,
    ) -> Conversation | None:
        """Return one conversation by ID."""

        try:
            point = self._client.retrieve(
                collection_name=self.COLLECTION_NAME,
                ids=[str(conversation_id)],
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return None

        if not point:
            return None

        payload = point[0].payload or {}

        if payload.get("memory_type") != "conversation":
            return None

        try:
            return Conversation(
                conversation_id=UUID(
                    str(payload["conversation_id"])
                ),
                title=str(payload["title"]),
                created_at=datetime.fromisoformat(
                    str(payload["created_at"])
                ),
                updated_at=datetime.fromisoformat(
                    str(payload["updated_at"])
                ),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def rename_conversation(
        self,
        conversation_id: UUID,
        title: str,
    ) -> Conversation | None:
        """Rename an existing conversation."""

        title = title.strip()

        if not title:
            raise ValueError(
                "title must not be empty"
            )

        conversation = self.get_conversation(
            conversation_id,
        )

        if conversation is None:
            return None

        renamed_conversation = Conversation(
            conversation_id=conversation.conversation_id,
            title=title,
            created_at=conversation.created_at,
            updated_at=datetime.now(timezone.utc),
        )

        self._client.set_payload(
            collection_name=self.COLLECTION_NAME,
            payload=renamed_conversation.to_payload(),
            points=[str(conversation_id)],
        )

        return renamed_conversation

    def delete_conversation(
        self,
        conversation_id: UUID,
    ) -> bool:
        """Delete a conversation and all of its messages and topics."""

        conversation = self.get_conversation(
            conversation_id,
        )

        if conversation is None:
            return False

        conversation_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="conversation_id",
                    match=models.MatchValue(
                        value=str(conversation_id)
                    ),
                )
            ]
        )

        self._client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=conversation_filter,
            ),
        )

        self._client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=models.PointIdsList(
                points=[str(conversation_id)],
            ),
        )

        return True

    def list_conversations(
        self,
        limit: int = 50,
    ) -> list[Conversation]:
        """Return conversations ordered by most recently updated."""

        if limit < 1:
            raise ValueError(
                "limit must be greater than zero"
            )

        points, _ = self._client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="memory_type",
                        match=models.MatchValue(
                            value="conversation"
                        ),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        conversations: list[Conversation] = []

        for point in points:
            payload = point.payload or {}

            try:
                conversations.append(
                    Conversation(
                        conversation_id=UUID(
                            str(payload["conversation_id"])
                        ),
                        title=str(payload["title"]),
                        created_at=datetime.fromisoformat(
                            str(payload["created_at"])
                        ),
                        updated_at=datetime.fromisoformat(
                            str(payload["updated_at"])
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        conversations.sort(
            key=lambda conversation: conversation.updated_at,
            reverse=True,
        )

        return conversations[:limit]

    def touch_conversation(
        self,
        conversation_id: UUID,
    ) -> Conversation | None:
        """Update a conversation's updated_at timestamp."""

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            return None

        updated_conversation = Conversation(
            conversation_id=conversation.conversation_id,
            title=conversation.title,
            created_at=conversation.created_at,
            updated_at=datetime.now(timezone.utc),
        )

        self._client.set_payload(
            collection_name=self.COLLECTION_NAME,
            payload=updated_conversation.to_payload(),
            points=[str(conversation_id)],
        )

        return updated_conversation

    def save_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
    ) -> ChatMessage:
        if role not in {"user", "assistant"}:
            raise ValueError(
                "role must be 'user' or 'assistant'"
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "content must not be empty"
            )

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            raise ValueError(
                "conversation does not exist"
            )

        message = ChatMessage(
            conversation_id=conversation_id,
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

        self.touch_conversation(
            conversation_id
        )

        return message

    def get_recent_messages(
        self,
        conversation_id: UUID,
        limit: int = 10,
    ) -> list[ChatMessage]:
        if limit < 1:
            raise ValueError(
                "limit must be greater than zero"
            )

        points, _ = self._client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="conversation_id",
                        match=models.MatchValue(
                            value=str(conversation_id)
                        ),
                    ),
                    models.FieldCondition(
                        key="memory_type",
                        match=models.MatchValue(
                            value="message"
                        ),
                    ),
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        messages: list[ChatMessage] = []

        for point in points:
            payload = point.payload or {}

            try:
                messages.append(
                    ChatMessage(
                        conversation_id=UUID(
                            str(payload["conversation_id"])
                        ),
                        message_id=UUID(
                            str(payload["message_id"])
                        ),
                        role=payload["role"],
                        content=str(
                            payload["content"]
                        ),
                        timestamp=datetime.fromisoformat(
                            str(payload["timestamp"])
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        messages.sort(
            key=lambda message: message.timestamp
        )

        return messages[-limit:]

    def save_topic(
        self,
        conversation_id: UUID,
        topic: str,
        source_message_id: UUID,
    ) -> ConversationTopic:
        topic = topic.strip()

        if not topic:
            raise ValueError(
                "topic must not be empty"
            )

        conversation = self.get_conversation(
            conversation_id
        )

        if conversation is None:
            raise ValueError(
                "conversation does not exist"
            )

        conversation_topic = ConversationTopic(
            conversation_id=conversation_id,
            topic=topic,
            source_message_id=source_message_id,
            timestamp=datetime.now(timezone.utc),
        )

        self._client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=str(uuid4()),
                    vector=[0.0],
                    payload=conversation_topic.to_payload(),
                )
            ],
        )

        self.touch_conversation(
            conversation_id
        )

        return conversation_topic

    def get_current_topic(
        self,
        conversation_id: UUID,
    ) -> ConversationTopic | None:
        points, _ = self._client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="conversation_id",
                        match=models.MatchValue(
                            value=str(conversation_id)
                        ),
                    ),
                    models.FieldCondition(
                        key="memory_type",
                        match=models.MatchValue(
                            value="topic"
                        ),
                    ),
                ]
            ),
            limit=20,
            with_payload=True,
            with_vectors=False,
        )

        if not points:
            return None

        topics: list[ConversationTopic] = []

        for point in points:
            payload = point.payload or {}

            try:
                topics.append(
                    ConversationTopic(
                        conversation_id=UUID(
                            str(payload["conversation_id"])
                        ),
                        topic=str(payload["topic"]),
                        source_message_id=UUID(
                            str(
                                payload[
                                    "source_message_id"
                                ]
                            )
                        ),
                        timestamp=datetime.fromisoformat(
                            str(payload["timestamp"])
                        ),
                    )
                )
            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                continue

        if not topics:
            return None

        topics.sort(
            key=lambda topic: topic.timestamp
        )

        return topics[-1]