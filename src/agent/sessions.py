import asyncio
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class ConversationSession:
    session_id: str

    messages: list[dict[str, object]] = field(default_factory=list)

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[
            str,
            ConversationSession,
        ] = {}

        self._lock = asyncio.Lock()

    async def create(self) -> ConversationSession:
        session = ConversationSession(
            session_id=str(uuid4()),
        )

        async with self._lock:
            self._sessions[session.session_id] = session

        return session

    async def get(
        self,
        session_id: str,
    ) -> ConversationSession | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete(
        self,
        session_id: str,
    ) -> bool:
        async with self._lock:
            return (
                self._sessions.pop(
                    session_id,
                    None,
                )
                is not None
            )
