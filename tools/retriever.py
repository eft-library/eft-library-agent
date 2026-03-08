import json
import logging
from db.connection import get_pool
from tools.embedder import get_embedding
from schemas.models import RagDocument
import os

log = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD"))
TRGM_THRESHOLD = float(os.getenv("RAG_TRGM_THRESHOLD"))
RRF_K = int(os.getenv("RAG_RRF_K"))


def _reciprocal_rank_fusion(
    vector_rows: list,
    trgm_rows: list,
    k: int = 60,
) -> dict[tuple[str, str], float]:
    """RRF로 벡터 검색 + trgm 검색 결과 합산"""
    scores: dict[tuple[str, str], float] = {}

    for rank, row in enumerate(vector_rows):
        key = (row["source_table"], row["ref_id"])
        scores[key] = scores.get(key, 0.0) + 1 / (k + rank + 1)

    for rank, row in enumerate(trgm_rows):
        key = (row["source_table"], row["ref_id"])
        scores[key] = scores.get(key, 0.0) + 1 / (k + rank + 1)

    return scores


async def search_rag(
    query: str,
    lang: str = "ko",
    limit: int = int(os.getenv("RAG_LIMIT", "10")),
    source_table: str | None = None,
) -> list[RagDocument]:
    """
    2-step 하이브리드 RAG 검색
    1단계: identifier 청크에서 벡터 검색 + trgm 검색 → RRF 합산 → ref_id 추출
    2단계: 추출된 ref_id의 content 청크 전체 조회
    """
    embedding = await get_embedding(query)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    pool = await get_pool()
    async with pool.acquire() as conn:

        # ── 1단계-A: 벡터 검색 ──────────────────────────────────────────────
        if source_table:
            vector_rows = await conn.fetch(
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
                limit * 2,
            )
        else:
            vector_rows = await conn.fetch(
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
                limit * 2,
            )

        # ── 1단계-B: trgm 검색 ──────────────────────────────────────────────
        if source_table:
            trgm_rows = await conn.fetch(
                """
                SELECT
                    source_table, ref_type, ref_id,
                    similarity(content, $1) AS similarity
                FROM rag_documents
                WHERE lang = $2
                  AND source_table = $3
                  AND chunk_type = 'identifier'
                  AND similarity(content, $1) > $4
                ORDER BY similarity DESC
                LIMIT $5
                """,
                query,
                lang,
                source_table,
                TRGM_THRESHOLD,
                limit * 2,
            )
        else:
            trgm_rows = await conn.fetch(
                """
                SELECT
                    source_table, ref_type, ref_id,
                    similarity(content, $1) AS similarity
                FROM rag_documents
                WHERE lang = $2
                  AND chunk_type = 'identifier'
                  AND similarity(content, $1) > $3
                ORDER BY similarity DESC
                LIMIT $4
                """,
                query,
                lang,
                TRGM_THRESHOLD,
                limit * 2,
            )

        log.info(
            f"[retriever] 검색 완료: vector={len(vector_rows)}개 "
            f"trgm={len(trgm_rows)}개 query={query[:30]}..."
        )

        # ── RRF 합산 ────────────────────────────────────────────────────────
        rrf_scores = _reciprocal_rank_fusion(vector_rows, trgm_rows, k=RRF_K)

        if not rrf_scores:
            log.info(f"[retriever] 검색 결과 없음 query={query[:30]}...")
            return []

        # 벡터 similarity 맵 (임계값 필터링 + 로그용)
        vector_similarity_map: dict[tuple[str, str], float] = {
            (row["source_table"], row["ref_id"]): round(float(row["similarity"]), 4)
            for row in vector_rows
        }

        # ── 임계값 필터링 ────────────────────────────────────────────────────
        # trgm에서만 히트한 경우(벡터 similarity 없음)는 trgm 결과를 신뢰해서 통과
        filtered_keys = [
            key
            for key, rrf_score in sorted(
                rrf_scores.items(), key=lambda x: x[1], reverse=True
            )
            if vector_similarity_map.get(key, 0.0) >= SIMILARITY_THRESHOLD
            or key not in vector_similarity_map  # trgm에서만 찾힌 경우
        ][:limit]

        if not filtered_keys:
            max_sim = max(vector_similarity_map.values(), default=0.0)
            log.info(
                f"[retriever] 임계값({SIMILARITY_THRESHOLD}) 미달로 결과 없음 "
                f"max_vector_similarity={max_sim:.4f} "
                f"query={query[:30]}..."
            )
            return []

        source_tables = [k[0] for k in filtered_keys]
        ref_ids = [k[1] for k in filtered_keys]

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

        similarity = vector_similarity_map.get(
            (row["source_table"], row["ref_id"]), 0.0
        )

        results.append(
            RagDocument(
                source_table=row["source_table"],
                source_id=row["source_id"],
                lang=row["lang"],
                ref_id=row["ref_id"],
                content=row["content"],
                metadata=metadata,
                similarity=similarity,
            )
        )

    # RRF 점수 기준으로 정렬
    rrf_key_order = {key: rank for rank, key in enumerate(filtered_keys)}
    results.sort(key=lambda r: rrf_key_order.get((r.source_table, r.ref_id), 999))

    for row in content_rows:
        log.info(f"content row: source_id={row['source_id']} ref_id={row['ref_id']}")

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
            or ""
        )
        if isinstance(name, dict):
            name = name.get(lang, "")
        log.info(
            f"  {r.source_table} | {r.source_id} | {name} | similarity={r.similarity}"
        )

    return results
