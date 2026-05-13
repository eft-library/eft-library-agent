import json
import logging
import os

from db.connection import get_pool
from schemas.models_v3 import ChatMessageV3, Lang

log = logging.getLogger(__name__)


async def save_message_v3(
    session_id: str,
    role: str,
    content: str,
    lang: Lang = "ko",
    source_docs: list[dict] | None = None,
) -> dict:
    if role not in ("user", "assistant", "system"):
        raise ValueError(
            f"role은 'user', 'assistant', 'system'만 가능합니다. 입력값: {role}"
        )

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO chat_messages_v3 (session_id, role, content, lang, source_docs)
            VALUES ($1::uuid, $2, $3, $4, $5)
            RETURNING id, created_at
            """,
            session_id,
            role,
            content,
            lang,
            json.dumps(source_docs or [], ensure_ascii=False),
        )

    log.info("[history_v3] save session=%s role=%s id=%s", session_id, role, row["id"])
    return {"id": row["id"], "created_at": row["created_at"].isoformat()}


async def get_history_v3(
    session_id: str,
    limit: int = int(os.getenv("RAG_LIMIT", "10")),
) -> list[ChatMessageV3]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content FROM (
                SELECT role, content, created_at
                FROM chat_messages_v3
                WHERE session_id = $1::uuid
                ORDER BY created_at DESC
                LIMIT $2
            ) sub
            ORDER BY created_at ASC
            """,
            session_id,
            limit,
        )

    messages = [ChatMessageV3(role=row["role"], content=row["content"]) for row in rows]
    log.info("[history_v3] get session=%s messages=%s", session_id, len(messages))
    return messages
