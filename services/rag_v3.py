import json
import logging
import os

from schemas.models_v3 import ChatMessageV3, Domain, Lang, RagDocumentV3
from tools.history_v3 import get_history_v3, save_message_v3
from tools.llm_v3 import chat_llm_stream_v3
from tools.retriever_v3 import search_rag_v3

log = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = int(os.getenv("RAG_V3_MAX_CONTEXT_CHARS", "18000"))


def build_context_v3(
    docs: list[RagDocumentV3],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    if not docs:
        return ""

    parts: list[str] = []
    used_chars = 0
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        entity_name = metadata.get("entity_name") or metadata.get("title") or doc.entity_id
        url = metadata.get("url") or metadata.get("link")
        url_line = f"\nURL: {url}" if url else ""
        scores = (
            f"vector={doc.similarity}, lexical={doc.lexical_score}, "
            f"trigram={doc.trigram_score}, fused={doc.fused_score}"
        )
        text = (
            f"[문서 {i}]\n"
            f"domain: {doc.domain}\n"
            f"entity: {entity_name} ({doc.entity_id})\n"
            f"chunk_type: {doc.chunk_type}\n"
            f"scores: {scores}"
            f"{url_line}\n"
            f"{doc.content.strip()}"
        )
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[: max(0, remaining - 20)].rstrip() + "\n... [truncated]"
        parts.append(text)
        used_chars += len(text) + 2

    return "\n\n".join(parts)


def source_docs_v3(docs: list[RagDocumentV3]) -> list[dict]:
    sources: list[dict] = []
    seen = set()
    for doc in docs:
        metadata = doc.metadata or {}
        key = (doc.domain, doc.entity_id, doc.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "domain": doc.domain,
                "entity_id": doc.entity_id,
                "chunk_id": doc.chunk_id,
                "chunk_type": doc.chunk_type,
                "entity_name": metadata.get("entity_name"),
                "url": metadata.get("url") or metadata.get("link"),
                "similarity": doc.similarity,
                "lexical_score": doc.lexical_score,
                "trigram_score": doc.trigram_score,
                "fused_score": doc.fused_score,
            }
        )
    return sources


async def run_rag_pipeline_stream_v3(
    session_id: str,
    user_query: str,
    lang: Lang = "ko",
    rag_limit: int = int(os.getenv("RAG_LIMIT", "3")),
    history_limit: int = int(os.getenv("RAG_LIMIT", "3")),
    domain: Domain | None = None,
):
    history = await get_history_v3(session_id, limit=history_limit)
    await save_message_v3(session_id, "user", user_query, lang)

    docs = await search_rag_v3(
        query=user_query,
        lang=lang,
        limit=rag_limit,
        domain=domain,
    )
    sources = source_docs_v3(docs)
    context = build_context_v3(docs)

    yield f"data: {json.dumps({'type': 'docs', 'docs': sources}, ensure_ascii=False)}\n\n"

    messages = [*history, ChatMessageV3(role="user", content=user_query)]
    full_answer = ""
    async for token in chat_llm_stream_v3(messages=messages, context=context, lang=lang):
        full_answer += token
        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    await save_message_v3(session_id, "assistant", full_answer, lang, sources)
    log.info("[rag_pipeline_v3] session=%s docs=%s", session_id, len(docs))
