import json
import logging
from collections.abc import Iterable
from datetime import datetime

import asyncpg
import httpx

from schemas.models_v3 import Lang, RagChunkV3
from tools.embedder import get_embedding

log = logging.getLogger(__name__)

SUPPORTED_LANGS: tuple[Lang, ...] = ("ko", "en", "ja")


def lang_value(row: dict, base_name: str, lang: Lang, fallback: Lang = "en") -> str:
    value = row.get(f"{base_name}_{lang}")
    if value:
        return str(value)
    fallback_value = row.get(f"{base_name}_{fallback}")
    return str(fallback_value or "")


def clean_parts(parts: Iterable[str | None]) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def chunk_id(*parts: str | int | None) -> str:
    return ":".join(str(part).strip() for part in parts if part is not None and str(part).strip())


def with_common_metadata(
    *,
    domain: str,
    entity_id: str,
    entity_name: str,
    lang: Lang,
    section: str,
    url: str | None = None,
    image: str | None = None,
    source_table: str | None = None,
    source_updated_at: datetime | str | None = None,
    extra: dict | None = None,
) -> dict:
    metadata = {
        "domain": domain,
        "entity_id": entity_id,
        "entity_name": entity_name,
        "lang": lang,
        "section": section,
        "url": url,
        "image": image,
        "schema_version": "v3",
        "updated_at": source_updated_at.isoformat()
        if isinstance(source_updated_at, datetime)
        else source_updated_at,
    }
    if source_table:
        metadata["source_table"] = source_table
    if extra:
        metadata.update(extra)
    return {key: value for key, value in metadata.items() if value is not None}


async def embed_chunk(chunk: RagChunkV3) -> list[float]:
    return await get_embedding(chunk.content)


async def upsert_rag_chunk_v3(
    conn: asyncpg.Connection,
    chunk: RagChunkV3,
    embedding: list[float],
) -> None:
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    await conn.execute(
        """
        INSERT INTO rag_documents_v3 (
            domain, entity_id, chunk_id, chunk_type, lang,
            content, embedding, searchable, source_table,
            source_updated_at, metadata, is_active
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7::vector, $8, $9, $10, $11, true)
        ON CONFLICT (domain, entity_id, chunk_id, lang)
        DO UPDATE SET
            chunk_type        = EXCLUDED.chunk_type,
            content           = EXCLUDED.content,
            embedding         = EXCLUDED.embedding,
            searchable        = EXCLUDED.searchable,
            source_table      = EXCLUDED.source_table,
            source_updated_at = EXCLUDED.source_updated_at,
            metadata          = EXCLUDED.metadata,
            is_active         = true,
            updated_at        = now()
        """,
        chunk.domain,
        chunk.entity_id,
        chunk.chunk_id,
        chunk.chunk_type,
        chunk.lang,
        chunk.content,
        embedding_str,
        chunk.searchable,
        chunk.source_table,
        chunk.source_updated_at,
        json.dumps(chunk.metadata, ensure_ascii=False),
    )


async def upsert_rag_chunks_v3(
    conn: asyncpg.Connection,
    chunks: Iterable[RagChunkV3],
) -> int:
    count = 0
    for chunk in chunks:
        if not chunk.content.strip():
            continue
        embedding = await embed_chunk(chunk)
        await upsert_rag_chunk_v3(conn, chunk, embedding)
        count += 1
    log.info("[rag_builder_v3] upserted chunks=%s", count)
    return count


async def embed_chunks_with_client(
    chunks: Iterable[RagChunkV3],
    client: httpx.AsyncClient,
    ollama_base_url: str,
    embed_model: str,
) -> list[tuple[RagChunkV3, list[float]]]:
    embedded: list[tuple[RagChunkV3, list[float]]] = []
    for chunk in chunks:
        if not chunk.content.strip():
            continue
        response = await client.post(
            f"{ollama_base_url}/api/embed",
            json={"model": embed_model, "input": chunk.content},
            timeout=60.0,
        )
        response.raise_for_status()
        embedded.append((chunk, response.json()["embeddings"][0]))
    return embedded
