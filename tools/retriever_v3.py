import json
import logging
import os
import argparse
import asyncio
from collections.abc import Iterable
from dotenv import load_dotenv

from db.connection import get_pool
from db.connection import close_pool
from schemas.models_v3 import Domain, Lang, RagDocumentV3
from tools.embedder import get_embedding

load_dotenv()

log = logging.getLogger(__name__)

VECTOR_THRESHOLD = float(os.getenv("RAG_V3_VECTOR_THRESHOLD", "0.45"))
TRGM_THRESHOLD = float(os.getenv("RAG_TRGM_THRESHOLD", "0.08"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))
ANSWER_CHUNK_LIMIT = int(os.getenv("RAG_ANSWER_CHUNK_LIMIT", "12"))

SEARCHABLE_CHUNK_TYPES = ("identifier", "retrieval", "summary", "relation")
ANSWER_CHUNK_TYPES = ("content", "relation", "guide", "summary", "retrieval")


def _entity_key(row) -> tuple[str, str]:
    return row["domain"], row["entity_id"]


def _reciprocal_rank_fusion(
    ranked_lists: Iterable[list],
    k: int = 60,
) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for rows in ranked_lists:
        deduped_rows = []
        seen = set()
        for row in rows:
            key = _entity_key(row)
            if key in seen:
                continue
            seen.add(key)
            deduped_rows.append(row)

        for rank, row in enumerate(deduped_rows):
            key = _entity_key(row)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank + 1)
    return scores


def _max_score_by_entity(rows: list) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for row in rows:
        key = _entity_key(row)
        score = round(float(row["score"]), 4)
        scores[key] = max(scores.get(key, 0.0), score)
    return scores


async def search_rag_v3(
    query: str,
    lang: Lang = "ko",
    limit: int = int(os.getenv("RAG_LIMIT", "10")),
    domain: Domain | None = None,
) -> list[RagDocumentV3]:
    embedding = await get_embedding(query)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    candidate_limit = max(limit * 4, 20)

    pool = await get_pool()
    async with pool.acquire() as conn:
        vector_params = [
            embedding_str,
            lang,
            list(SEARCHABLE_CHUNK_TYPES),
            candidate_limit,
        ]
        lexical_params = [
            query,
            lang,
            list(SEARCHABLE_CHUNK_TYPES),
            candidate_limit,
        ]
        trigram_params = [
            query,
            lang,
            list(SEARCHABLE_CHUNK_TYPES),
            candidate_limit,
            TRGM_THRESHOLD,
        ]
        vector_domain_clause = ""
        lexical_domain_clause = ""
        trigram_domain_clause = ""
        if domain:
            vector_domain_clause = "AND domain = $5"
            lexical_domain_clause = "AND domain = $5"
            trigram_domain_clause = "AND domain = $6"
            vector_params.append(domain)
            lexical_params.append(domain)
            trigram_params.append(domain)

        vector_rows = await conn.fetch(
            f"""
            SELECT
                domain, entity_id, chunk_id, chunk_type,
                1 - (embedding <=> $1::vector) AS score
            FROM rag_documents_v3
            WHERE lang = $2
              AND is_active
              AND searchable
              AND chunk_type = ANY($3::text[])
              {vector_domain_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT $4
            """,
            *vector_params,
        )

        lexical_rows = await conn.fetch(
            f"""
            SELECT
                domain, entity_id, chunk_id, chunk_type,
                ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) AS score
            FROM rag_documents_v3
            WHERE lang = $2
              AND is_active
              AND searchable
              AND chunk_type = ANY($3::text[])
              AND search_vector @@ plainto_tsquery('simple', $1)
              {lexical_domain_clause}
            ORDER BY score DESC
            LIMIT $4
            """,
            *lexical_params,
        )

        trigram_rows = await conn.fetch(
            f"""
            SELECT
                domain, entity_id, chunk_id, chunk_type,
                similarity(content, $1) AS score
            FROM rag_documents_v3
            WHERE lang = $2
              AND is_active
              AND searchable
              AND chunk_type = ANY($3::text[])
              AND similarity(content, $1) > $5
              {trigram_domain_clause}
            ORDER BY score DESC
            LIMIT $4
            """,
            *trigram_params,
        )

        rrf_scores = _reciprocal_rank_fusion(
            [list(vector_rows), list(lexical_rows), list(trigram_rows)],
            k=RRF_K,
        )

        if not rrf_scores:
            log.info("[retriever_v3] no candidates query=%s", query[:40])
            return []

        vector_score_map = _max_score_by_entity(list(vector_rows))
        lexical_score_map = _max_score_by_entity(list(lexical_rows))
        trigram_score_map = _max_score_by_entity(list(trigram_rows))

        filtered_keys = [
            key
            for key, _score in sorted(
                rrf_scores.items(), key=lambda item: item[1], reverse=True
            )
            if vector_score_map.get(key, 0.0) >= VECTOR_THRESHOLD
            or lexical_score_map.get(key, 0.0) > 0.0
            or trigram_score_map.get(key, 0.0) > 0.0
        ][:limit]

        if not filtered_keys:
            log.info("[retriever_v3] low confidence candidates query=%s", query[:40])
            return []

        domains = [key[0] for key in filtered_keys]
        entity_ids = [key[1] for key in filtered_keys]

        rows = await conn.fetch(
            """
            WITH selected(domain, entity_id, ord) AS (
                SELECT * FROM unnest($1::text[], $2::text[]) WITH ORDINALITY
                    AS t(domain, entity_id, ord)
            )
            SELECT
                d.domain, d.entity_id, d.chunk_id, d.chunk_type, d.lang,
                d.content, d.metadata, selected.ord
            FROM rag_documents_v3 d
            JOIN selected
              ON selected.domain = d.domain
             AND selected.entity_id = d.entity_id
            WHERE d.lang = $3
              AND d.is_active
              AND d.chunk_type = ANY($4::text[])
            ORDER BY
                selected.ord,
                CASE d.chunk_type
                    WHEN 'relation' THEN 1
                    WHEN 'content' THEN 2
                    WHEN 'guide' THEN 3
                    WHEN 'summary' THEN 4
                    WHEN 'retrieval' THEN 5
                    ELSE 9
                END,
                d.chunk_id
            LIMIT $5
            """,
            domains,
            entity_ids,
            lang,
            list(ANSWER_CHUNK_TYPES),
            ANSWER_CHUNK_LIMIT,
        )

    results: list[RagDocumentV3] = []
    for row in rows:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        key = (row["domain"], row["entity_id"])
        results.append(
            RagDocumentV3(
                domain=row["domain"],
                entity_id=row["entity_id"],
                chunk_id=row["chunk_id"],
                chunk_type=row["chunk_type"],
                lang=row["lang"],
                content=row["content"],
                metadata=metadata,
                similarity=vector_score_map.get(key),
                lexical_score=lexical_score_map.get(key),
                trigram_score=trigram_score_map.get(key),
                fused_score=round(rrf_scores.get(key, 0.0), 6),
            )
        )

    log.info(
        "[retriever_v3] query=%s vector=%s lexical=%s trigram=%s entities=%s chunks=%s",
        query[:40],
        len(vector_rows),
        len(lexical_rows),
        len(trigram_rows),
        len(filtered_keys),
        len(results),
    )
    return results


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Search V3 RAG documents.")
    parser.add_argument("query")
    parser.add_argument("--lang", default="ko", choices=["ko", "en", "ja"])
    parser.add_argument("--domain", choices=["item", "quest", "map", "boss", "hideout", "trader", "information", "story"])
    parser.add_argument("--limit", type=int, default=int(os.getenv("RAG_LIMIT", "10")))
    args = parser.parse_args()

    try:
        docs = await search_rag_v3(
            query=args.query,
            lang=args.lang,
            limit=args.limit,
            domain=args.domain,
        )
        for i, doc in enumerate(docs, 1):
            name = doc.metadata.get("entity_name", "")
            print(
                f"\n[{i}] {doc.domain}/{doc.entity_id} "
                f"{doc.chunk_type}:{doc.chunk_id} "
                f"vector={doc.similarity} lexical={doc.lexical_score} "
                f"trgm={doc.trigram_score} fused={doc.fused_score}"
            )
            if name:
                print(f"name: {name}")
            print(doc.content[:1200])
    finally:
        await close_pool()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(_main())
