import json
import logging
from db.connection import get_pool
from tools.embedder import get_embedding
from schemas.models import RagDocument
import os
import httpx

log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL")


CONDITION_KEYWORDS = [
    # 추천/조건
    "추천",
    "추천해",
    "뭐가 좋",
    "어떤 거",
    "어떤게",
    "레벨",
    "lv",
    "level",
    "쉬운",
    "어려운",
    "초보",
    "빠른",
    "효율",
    # 영어
    "recommend",
    "best",
    "cheap",
    "easy",
    "beginner",
    # 일본어
    "おすすめ",
    "レベル",
    "安い",
    "簡単",
]

ENTITY_KEYWORDS = [
    # 특정 이름을 묻는 패턴
    "어디서",
    "어디에",
    "위치",
    "어떻게 쓰",
    "뭐야",
    "뭔가요",
    "무엇",
    "정보",
    "스펙",
    "능력치",
    "스탯",
]


def classify_by_keyword(query: str) -> str | None:
    """
    키워드로 확실히 분류 가능하면 반환, 불확실하면 None
    """
    query_lower = query.lower()

    if any(kw in query_lower for kw in CONDITION_KEYWORDS):
        return "condition"

    if any(kw in query_lower for kw in ENTITY_KEYWORDS):
        return "entity"

    return None  # 불확실 → LLM으로


async def classify_query(query: str) -> str:
    # 1. 키워드로 먼저 시도
    keyword_result = classify_by_keyword(query)
    if keyword_result:
        log.info(
            f"[retriever] 쿼리 분류 (키워드): {keyword_result} query={query[:30]}..."
        )
        return keyword_result

    # 2. 불확실한 경우만 LLM 호출
    try:
        prompt = f"""다음 쿼리가 어떤 유형인지 분류하세요.

- entity: 특정 보스, 아이템, 퀘스트, 맵 등의 이름을 찾는 질문
- condition: 레벨, 가격, 효과, 추천, 조건 등으로 검색하는 질문

쿼리: {query}

반드시 JSON만 응답하세요: {{"type": "entity"}} 또는 {{"type": "condition"}}"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": CHAT_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=5.0,  # 짧게 잡아서 느리면 바로 fallback
            )
            response.raise_for_status()
            text = response.json()["message"]["content"].strip()
            result = json.loads(text)
            query_type = result.get("type", "entity")
            if query_type not in ("entity", "condition"):
                query_type = "entity"

        log.info(f"[retriever] 쿼리 분류 (LLM): {query_type} query={query[:30]}...")
        return query_type

    except Exception as e:
        # 3. LLM 실패 시 entity fallback (기존 방식 유지)
        log.warning(f"[retriever] 쿼리 분류 실패, entity fallback: {e}")
        return "entity"


async def search_rag(
    query: str,
    lang: str = "ko",
    limit: int = int(os.getenv("RAG_LIMIT")),
    source_table: str | None = None,
) -> list[RagDocument]:
    """
    2-step RAG 검색 (쿼리 의도 분류 포함)
    - entity:    identifier 벡터 검색 → ref_id 추출 → content 조회
    - condition: content 직접 벡터 검색
    """
    embedding = await get_embedding(query)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    query_type = await classify_query(query)

    pool = await get_pool()
    async with pool.acquire() as conn:

        if query_type == "condition":
            # ── condition: content 직접 벡터 검색 ──────────────────────────
            table_filter = "AND source_table = $3" if source_table else ""
            params = [embedding_str, lang]
            if source_table:
                params.append(source_table)
            params.append(limit)

            content_rows = await conn.fetch(
                f"""
                SELECT
                    source_table, source_id, lang, content, metadata,
                    ref_type, ref_id,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM rag_documents
                WHERE lang = $2
                  AND chunk_type = 'content'
                  {table_filter}
                ORDER BY embedding <=> $1::vector
                LIMIT ${len(params)}
                """,
                *params,
            )

            results = []
            for row in content_rows:
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                results.append(
                    RagDocument(
                        source_table=row["source_table"],
                        source_id=row["source_id"],
                        lang=row["lang"],
                        content=row["content"],
                        metadata=metadata,
                        similarity=round(float(row["similarity"]), 4),
                    )
                )

            log.info(
                f"[retriever] condition 검색 완료: {len(results)}개 query={query[:30]}..."
            )

        else:
            # ── entity: 기존 2-step 검색 ────────────────────────────────────
            if source_table:
                identifier_rows = await conn.fetch(
                    """
                    SELECT
                        source_table, ref_type, ref_id,
                        1 - (embedding <=> $1::vector) AS similarity
                    FROM rag_documents
                    WHERE lang = $2
                      AND source_table = $3
                      AND chunk_type = 'identifier'
                    ORDER BY embedding <=> $1::vector
                    LIMIT $4
                    """,
                    embedding_str,
                    lang,
                    source_table,
                    limit,
                )
            else:
                identifier_rows = await conn.fetch(
                    """
                    SELECT
                        source_table, ref_type, ref_id,
                        1 - (embedding <=> $1::vector) AS similarity
                    FROM rag_documents
                    WHERE lang = $2
                      AND chunk_type = 'identifier'
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    embedding_str,
                    lang,
                    limit,
                )

            if not identifier_rows:
                log.info(f"[retriever] identifier 검색 결과 없음 query={query[:30]}...")
                return []

            similarity_map: dict[tuple[str, str], float] = {
                (row["source_table"], row["ref_id"]): round(float(row["similarity"]), 4)
                for row in identifier_rows
            }
            source_tables = [row["source_table"] for row in identifier_rows]
            ref_ids = [row["ref_id"] for row in identifier_rows]

            log.info(
                f"[retriever] 1단계 완료: {len(ref_ids)}개 ref_id 추출 query={query[:30]}... lang={lang}"
            )

            content_rows = await conn.fetch(
                """
                SELECT
                    source_table, source_id, lang, content, metadata,
                    ref_type, ref_id
                FROM rag_documents
                WHERE lang = $1
                  AND chunk_type = 'content'
                  AND (source_table, ref_id) = ANY(
                      SELECT * FROM UNNEST($2::text[], $3::text[])
                  )
                ORDER BY source_table, ref_id, source_id
                """,
                lang,
                source_tables,
                ref_ids,
            )

            results = []
            for row in content_rows:
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                similarity = similarity_map.get(
                    (row["source_table"], row["ref_id"]), 0.0
                )
                results.append(
                    RagDocument(
                        source_table=row["source_table"],
                        source_id=row["source_id"],
                        lang=row["lang"],
                        content=row["content"],
                        metadata=metadata,
                        similarity=similarity,
                    )
                )

            results.sort(key=lambda r: r.similarity, reverse=True)
            log.info(
                f"[retriever] 2단계 완료: content {len(results)}개 조회 query={query[:30]}... lang={lang}"
            )

    for r in results:
        name = (
            r.metadata.get("quest_name")
            or r.metadata.get("boss_name")
            or r.metadata.get("item_name")
            or r.metadata.get("hideout_name")
            or r.metadata.get("story_name")
            or r.metadata.get("map_name")
            or ""
        )
        if isinstance(name, dict):
            name = name.get(lang, "")
        log.info(
            f"  {r.source_table} | {r.source_id} | {name} | similarity={r.similarity}"
        )

    return results
