import json
import logging
import os

from schemas.models_v3 import ChatMessageV3, Domain, Lang, RagDocumentV3
from tools.answerability_v3 import assess_answerability_v3
from tools.history_v3 import get_history_v3, save_message_v3
from tools.llm_v3 import chat_llm_stream_v3
from tools.retriever_v3 import search_rag_v3
from tools.web_fallback_v3 import WebFallbackResult, search_web_fallback_v3

log = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = int(os.getenv("RAG_V3_MAX_CONTEXT_CHARS", "18000"))
ANSWERABILITY_CHECK_ENABLED = (
    os.getenv("RAG_V3_ANSWERABILITY_CHECK", "true").lower() == "true"
)


def build_context_v3(
    docs: list[RagDocumentV3],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    if not docs:
        return ""

    parts: list[str] = [
        "[관계 섹션 해석 규칙]",
        "- [이 아이템 제작에 필요한 재료]: 이 아이템은 제작 가능하며, 각 항목 괄호의 시설/Lv가 제작 위치입니다.",
        "- [이 아이템을 재료로 제작 가능한 아이템]: 이 아이템을 재료로 써서 만들 수 있는 결과물입니다.",
        "- [이 아이템 교환에 필요한 재료]: 이 아이템을 바터로 받기 위해 필요한 재료입니다.",
        "- [이 아이템을 재료로 교환 가능한 아이템]: 이 아이템을 내고 받을 수 있는 바터 결과물입니다.",
    ]
    used_chars = sum(len(part) + 1 for part in parts)
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


def build_web_context_v3(results: list[WebFallbackResult]) -> str:
    if not results:
        return ""

    parts = [
        "[웹 fallback 참고 문서]",
        "로컬 RAG에서 충분한 문서를 찾지 못해 웹 검색 결과를 참고합니다.",
        "커뮤니티 사이트 결과는 패치 버전, 작성 시점, 작성자 경험에 따라 부정확할 수 있습니다.",
    ]
    for i, result in enumerate(results, 1):
        community = "yes" if result.is_community else "no"
        parts.append(
            "\n".join(
                [
                    f"[웹 문서 {i}]",
                    f"title: {result.title}",
                    f"url: {result.url}",
                    f"source: {result.source}",
                    f"community: {community}",
                    result.snippet,
                ]
            )
        )
    return "\n\n".join(parts)


def source_web_docs_v3(results: list[WebFallbackResult]) -> list[dict]:
    return [
        {
            "origin": "web",
            "title": result.title,
            "url": result.url,
            "source": result.source,
            "is_community": result.is_community,
        }
        for result in results
    ]


async def should_use_web_fallback_v3(
    query: str,
    docs: list[RagDocumentV3],
    context: str,
    lang: Lang,
) -> tuple[bool, str]:
    if not docs:
        return True, "no_local_docs"

    if not ANSWERABILITY_CHECK_ENABLED:
        return False, "answerability_check_disabled"

    result = await assess_answerability_v3(query=query, context=context, lang=lang)
    if not result.answerable:
        return True, f"not_answerable:{result.reason}"
    return False, f"answerable:{result.reason}"


async def run_rag_pipeline_stream_v3(
    session_id: str,
    user_query: str,
    lang: Lang = "ko",
    rag_limit: int = int(os.getenv("RAG_LIMIT", "3")),
    history_limit: int = int(os.getenv("RAG_LIMIT", "3")),
    domain: Domain | None = None,
):
    full_answer = ""
    sources: list[dict] = []
    try:
        if domain in ("", "string"):
            domain = None
        if rag_limit <= 0:
            rag_limit = int(os.getenv("RAG_LIMIT", "3"))
        if history_limit <= 0:
            history_limit = int(os.getenv("RAG_LIMIT", "3"))

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
        use_web, fallback_reason = await should_use_web_fallback_v3(
            query=user_query,
            docs=docs,
            context=context,
            lang=lang,
        )
        log.info(
            "[rag_pipeline_v3] fallback_check use_web=%s reason=%s",
            use_web,
            fallback_reason,
        )
        if use_web:
            web_results = await search_web_fallback_v3(user_query)
            if web_results:
                sources = source_web_docs_v3(web_results)
                context = build_web_context_v3(web_results)
            else:
                sources = []
                context = ""

        yield f"data: {json.dumps({'type': 'docs', 'docs': sources}, ensure_ascii=False)}\n\n"

        messages = [*history, ChatMessageV3(role="user", content=user_query)]
        async for token in chat_llm_stream_v3(messages=messages, context=context, lang=lang):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

        await save_message_v3(session_id, "assistant", full_answer, lang, sources)
        log.info("[rag_pipeline_v3] session=%s docs=%s", session_id, len(docs))
    except Exception as exc:
        log.exception("[rag_pipeline_v3] stream error session=%s", session_id)
        error_payload = {
            "type": "error",
            "message": "RAG 응답 생성 중 오류가 발생했습니다.",
            "detail": str(exc),
        }
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
        if full_answer:
            try:
                await save_message_v3(session_id, "assistant", full_answer, lang, sources)
            except Exception:
                log.exception("[rag_pipeline_v3] failed to save partial answer")
