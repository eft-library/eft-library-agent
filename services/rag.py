import logging
from tools.retriever import search_rag
from tools.llm import chat_llm, chat_llm_stream
from tools.history import save_message, get_history
from schemas.models import ChatMessage, RagDocument

log = logging.getLogger(__name__)


def build_context(docs: list[RagDocument]) -> str:
    """검색된 문서들을 LLM context 문자열로 조합"""
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[문서 {i}] (출처: {doc.source_table}, 유사도: {doc.similarity})\n{doc.content}"
        )
    return "\n\n".join(parts)


async def run_rag_pipeline(
    session_id: str,
    user_query: str,
    lang: str = "ko",
    rag_limit: int = 5,
    history_limit: int = 10,
    source_table: str | None = None,
) -> dict:
    """
    RAG 전체 파이프라인
    1. 히스토리 조회
    2. RAG 검색
    3. LLM 호출
    4. 메시지 저장 (user + assistant)
    """

    # 1. 히스토리 조회
    history = await get_history(session_id, limit=history_limit)

    # 2. 사용자 메시지 저장
    await save_message(session_id, "user", user_query, lang)

    # 3. RAG 검색
    docs = await search_rag(
        query=user_query,
        lang=lang,
        limit=rag_limit,
        source_table=source_table,
    )
    context = build_context(docs)

    # 4. LLM 호출 (히스토리 + 새 질문)
    messages = [*history, ChatMessage(role="user", content=user_query)]
    answer = await chat_llm(messages=messages, context=context)

    # 5. 어시스턴트 메시지 저장
    source_docs = [
        {
            "source_table": d.source_table,
            "source_id": d.source_id,
            "similarity": d.similarity,
        }
        for d in docs
    ]
    await save_message(session_id, "assistant", answer, lang, source_docs)

    log.info(f"[rag_pipeline] session={session_id} docs={len(docs)}")
    return {
        "answer": answer,
        "docs": [d.model_dump() for d in docs],
    }


async def run_rag_pipeline_stream(
    session_id: str,
    user_query: str,
    lang: str = "ko",
    rag_limit: int = 5,
    history_limit: int = 10,
    source_table: str | None = None,
):
    """
    스트리밍 RAG 파이프라인
    - 히스토리/RAG 검색은 동일
    - LLM 토큰을 yield로 흘려보냄
    - 완료 후 DB 저장
    """

    # 1. 히스토리 조회
    history = await get_history(session_id, limit=history_limit)

    # 2. 사용자 메시지 저장
    await save_message(session_id, "user", user_query, lang)

    # 3. RAG 검색
    docs = await search_rag(
        query=user_query,
        lang=lang,
        limit=rag_limit,
        source_table=source_table,
    )
    context = build_context(docs)

    # 4. docs 메타정보 먼저 yield (Next.js에서 출처 표시용)
    import json

    source_docs = [
        {
            "source_table": d.source_table,
            "source_id": d.source_id,
            "similarity": d.similarity,
        }
        for d in docs
    ]
    yield f"data: {json.dumps({'type': 'docs', 'docs': source_docs}, ensure_ascii=False)}\n\n"

    # 5. LLM 스트리밍
    messages = [*history, ChatMessage(role="user", content=user_query)]
    full_answer = ""
    async for token in chat_llm_stream(messages=messages, context=context):
        full_answer += token
        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

    # 6. 완료 시그널
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # 7. 어시스턴트 메시지 저장
    await save_message(session_id, "assistant", full_answer, lang, source_docs)
    log.info(f"[rag_pipeline_stream] session={session_id} docs={len(docs)}")
