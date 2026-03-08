import json
import logging
from tools.retriever import search_rag
from tools.llm import chat_llm_stream
from tools.history import save_message, get_history
from tools.price import get_item_prices
from schemas.models import ChatMessage, RagDocument
import os

from tools.router import rule_based_routing

log = logging.getLogger(__name__)


async def build_context(docs: list[RagDocument], lang: str = "ko") -> str:
    """
    검색된 문서들을 LLM context 문자열로 조합
    item_i18n 문서가 있으면 ITEM_PRICE_I18N에서 시세 정보도 추가 (hybrid)
    ref_id 기준으로 item_id 수집 (source_id는 _spec/_craft 등 suffix가 붙을 수 있음)
    """
    if not docs:
        return ""

    # ref_id 기준으로 item_id 수집 (source_id 대신)
    item_ids = list(
        {
            doc.metadata.get("item_id")
            for doc in docs
            if doc.source_table == "item_i18n" and doc.metadata.get("item_id")
        }
    )

    # 시세 조회 (있을 때만)
    price_map: dict[str, str] = {}
    if item_ids:
        price_map = await get_item_prices(item_ids, lang)
        log.info(f"[hybrid] item_ids={item_ids} price_found={list(price_map.keys())}")

    parts = []
    for i, doc in enumerate(docs, 1):
        url = doc.metadata.get("url", "")
        url_line = f"\n출처 URL: {url}" if url else ""
        doc_text = f"[문서 {i}] (출처: {doc.source_table}, 유사도: {doc.similarity}){url_line}\n{doc.content}"

        # 아이템 문서에 시세 정보 추가 (item_id로 매핑)
        item_id = doc.metadata.get("item_id")
        if doc.source_table == "item_i18n" and item_id and item_id in price_map:
            doc_text += f"\n\n{price_map[item_id]}"

        parts.append(doc_text)

    return "\n\n".join(parts)


async def run_rag_pipeline_stream(
    session_id: str,
    user_query: str,
    lang: str = "ko",
    rag_limit: int = int(os.getenv("RAG_LIMIT")),
    history_limit: int = int(os.getenv("RAG_LIMIT")),
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

    if source_table is None:
        source_table = rule_based_routing(user_query)

    # 3. RAG 검색
    docs = await search_rag(
        query=user_query,
        lang=lang,
        limit=int(os.getenv("RAG_LIMIT")),
        source_table=source_table,
    )

    # 4. context 조합 (hybrid: 아이템이면 시세 추가)
    context = await build_context(docs, lang)

    # 5. docs 메타정보 먼저 yield (Next.js에서 출처 표시용)
    source_docs = [
        {
            "source_table": d.source_table,
            "source_id": d.source_id,
            "similarity": d.similarity,
        }
        for d in docs
    ]
    yield f"data: {json.dumps({'type': 'docs', 'docs': source_docs}, ensure_ascii=False)}\n\n"

    # 6. LLM 스트리밍
    messages = [*history, ChatMessage(role="user", content=user_query)]
    full_answer = ""
    async for token in chat_llm_stream(messages=messages, context=context, lang=lang):
        full_answer += token
        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

    # 7. 완료 시그널
    yield f"data: {json.dumps({'type': 'done'})}\n\n"

    # 8. 어시스턴트 메시지 저장
    await save_message(session_id, "assistant", full_answer, lang, source_docs)
    log.info(f"[rag_pipeline_stream] session={session_id} docs={len(docs)}")
