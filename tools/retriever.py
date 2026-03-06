import json
import logging
from db.connection import get_pool
from tools.embedder import get_embedding
from schemas.models import RagDocument
import os

log = logging.getLogger(__name__)


async def search_rag(
    query: str,
    lang: str = "ko",
    limit: int = int(os.getenv("RAG_LIMIT")),
    source_table: str | None = None,
) -> list[RagDocument]:
    """
    2-step RAG 검색
    1단계: identifier 청크에서 벡터 검색 → ref_id 추출
    2단계: 추출된 ref_id의 content 청크 전체 조회
    """
    embedding = await get_embedding(query)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    pool = await get_pool()
    async with pool.acquire() as conn:

        # ── 1단계: identifier 청크에서 유사도 검색 ──────────────────────────
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

        # ref_id별 similarity 보존 (content 조회 후 매핑용)
        similarity_map: dict[tuple[str, str], float] = {
            (row["source_table"], row["ref_id"]): round(float(row["similarity"]), 4)
            for row in identifier_rows
        }

        # (source_table, ref_id) 쌍 목록
        source_tables = [row["source_table"] for row in identifier_rows]
        ref_ids = [row["ref_id"] for row in identifier_rows]

        log.info(
            f"[retriever] 1단계 완료: {len(ref_ids)}개 ref_id 추출 "
            f"query={query[:30]}... lang={lang}"
        )

        # ── 2단계: content 청크 전체 조회 ──────────────────────────────────
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

    # ── 결과 조합 ────────────────────────────────────────────────────────────
    results = []
    for row in content_rows:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        similarity = similarity_map.get((row["source_table"], row["ref_id"]), 0.0)

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

    # identifier 유사도 기준으로 정렬
    results.sort(key=lambda r: r.similarity, reverse=True)

    log.info(
        f"[retriever] 2단계 완료: content {len(results)}개 조회 "
        f"query={query[:30]}... lang={lang}"
    )
    for r in results:
        name = (
            r.metadata.get("quest_name")
            or r.metadata.get("boss_name")
            or r.metadata.get("item_name")
            or r.metadata.get("hideout_name")
            or r.metadata.get("story_name")
            or r.metadata.get("map_name")
            or r.metadata.get("item_name")
            or ""
        )
        if isinstance(name, dict):
            name = name.get(lang, "")
        log.info(
            f"  {r.source_table} | {r.source_id} | {name} | similarity={r.similarity}"
        )

    return results
