from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID


ChatRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """A single message belonging to a conversation."""

    conversation_id: UUID
    message_id: UUID
    role: ChatRole
    content: str
    timestamp: datetime

    def to_payload(self) -> dict[str, object]:
        return {
            "conversation_id": str(self.conversation_id),
            "message_id": str(self.message_id),
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "memory_type": "message",
        }


@dataclass(frozen=True)
class ConversationTopic:
    """The currently active topic of a conversation."""

    conversation_id: UUID
    topic: str
    source_message_id: UUID
    timestamp: datetime

    def to_payload(self) -> dict[str, object]:
        return {
            "conversation_id": str(self.conversation_id),
            "topic": self.topic,
            "source_message_id": str(self.source_message_id),
            "timestamp": self.timestamp.isoformat(),
            "memory_type": "topic",
        }


@dataclass(frozen=True)
class Conversation:
    """Represents an independent chat conversation."""

    conversation_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    def to_payload(self) -> dict[str, object]:
        return {
            "conversation_id": str(self.conversation_id),
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "memory_type": "conversation",
        }