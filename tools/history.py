import json
import logging
from db.connection import get_pool
from schemas.models import ChatMessage

log = logging.getLogger(__name__)


async def save_message(
    session_id: str,
    role: str,
    content: str,
    lang: str = "ko",
    source_docs: list[dict] | None = None,
) -> dict:
    if role not in ("user", "assistant"):
        raise ValueError(f"role은 'user' 또는 'assistant'만 가능합니다. 입력값: {role}")

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO chat_messages (session_id, role, content, lang, source_docs)
            VALUES ($1::uuid, $2, $3, $4, $5)
            RETURNING id, created_at
        """,
            session_id,
            role,
            content,
            lang,
            json.dumps(source_docs or [], ensure_ascii=False),
        )
        # 세션 updated_at 갱신
        await conn.execute(
            """
            UPDATE chat_sessions
            SET updated_at = NOW()
            WHERE session_id = $1::uuid
        """,
            session_id,
        )

    log.info(f"[history] save session={session_id} role={role} id={row['id']}")
    return {"id": row["id"], "created_at": row["created_at"].isoformat()}


async def get_history(
    session_id: str,
    limit: int = 10,
) -> list[ChatMessage]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content FROM (
                SELECT role, content, created_at
                FROM chat_messages
                WHERE session_id = $1::uuid
                ORDER BY created_at DESC
                LIMIT $2
            ) sub
            ORDER BY created_at ASC
        """,
            session_id,
            limit,
        )

    messages = [ChatMessage(role=row["role"], content=row["content"]) for row in rows]
    log.info(f"[history] get session={session_id} messages={len(messages)}")
    return messages
