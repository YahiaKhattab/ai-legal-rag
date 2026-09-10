from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID


ChatRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """A single message stored in conversation memory."""

    session_id: UUID
    message_id: UUID
    role: ChatRole
    content: str
    timestamp: datetime

    def to_payload(self) -> dict[str, object]:
        """Convert the message to a Qdrant payload."""
        return {
            "session_id": str(self.session_id),
            "message_id": str(self.message_id),
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }