import json
import logging
from db.connection import get_pool
from tools.embedder import get_embedding
from schemas.models import RagDocument

log = logging.getLogger(__name__)


async def search_rag(
    query: str,
    lang: str = "ko",
    limit: int = 5,
    source_table: str | None = None,
) -> list[RagDocument]:
    embedding = await get_embedding(query)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    pool = await get_pool()
    async with pool.acquire() as conn:
        if source_table:
            rows = await conn.fetch(
                """
                SELECT
                    source_table, source_id, lang, content, metadata,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM rag_documents
                WHERE lang = $2
                  AND source_table = $3
                ORDER BY embedding <=> $1::vector
                LIMIT $4
            """,
                embedding_str,
                lang,
                source_table,
                limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT
                    source_table, source_id, lang, content, metadata,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM rag_documents
                WHERE lang = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
            """,
                embedding_str,
                lang,
                limit,
            )

    results = []
    for row in rows:
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

    log.info(f"[retriever] query={query[:30]}... lang={lang} results={len(results)}")
    return results
