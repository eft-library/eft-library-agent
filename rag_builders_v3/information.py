import argparse
import asyncio
import logging

from bs4 import BeautifulSoup
from dotenv import load_dotenv

from db.connection import close_pool, get_pool
from rag_builders_v3.base import (
    SUPPORTED_LANGS,
    chunk_id,
    clean_parts,
    lang_value,
    upsert_rag_chunks_v3,
    with_common_metadata,
)
from schemas.models_v3 import Lang, RagChunkV3

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

LABELS = {
    "ko": {
        "information": "정보",
        "type": "유형",
        "title": "제목",
        "content": "본문",
        "news": "뉴스",
        "link": "링크",
        "new": "신규",
    },
    "en": {
        "information": "Information",
        "type": "Type",
        "title": "Title",
        "content": "Content",
        "news": "News",
        "link": "Link",
        "new": "New",
    },
    "ja": {
        "information": "情報",
        "type": "種類",
        "title": "タイトル",
        "content": "本文",
        "news": "ニュース",
        "link": "リンク",
        "new": "新規",
    },
}


def _clean_html(html_text: str | None) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for img in soup.find_all("img"):
        img.decompose()
    return soup.get_text(separator="\n", strip=True)


def _info_title(row: dict, lang: Lang) -> str:
    return lang_value(row, "title", lang)


def _header(row: dict, lang: Lang) -> str:
    lb = LABELS[lang]
    title = _info_title(row, lang)
    return clean_parts(
        [
            f"{lb['information']}: {title}",
            f"{lb['type']}: {row.get('information_type')}" if row.get("information_type") else None,
        ]
    )


async def fetch_information_rows(conn, info_id: str | None = None, limit: int | None = None):
    args = []
    where = ""
    if info_id:
        args.append(info_id)
        where = f"WHERE id = ${len(args)}"
    limit_clause = ""
    if limit is not None and not info_id:
        args.append(limit)
        limit_clause = f"LIMIT ${len(args)}"
    rows = await conn.fetch(
        f"""
        SELECT *
        FROM information
        {where}
        ORDER BY update_time DESC, id
        {limit_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


async def fetch_news_rows(conn, lang: Lang, limit: int | None = None):
    limit_clause = "LIMIT $1" if limit is not None else ""
    args = [limit] if limit is not None else []
    rows = await conn.fetch(
        f"""
        SELECT id::text, news_type AS information_type,
               title_en, title_ko, title_ja,
               link, is_new, is_renewal, is_active, update_time
        FROM news_items
        WHERE is_active IS TRUE
        ORDER BY sort_order NULLS LAST, id
        {limit_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


def build_information_chunks(row: dict, lang: Lang) -> list[RagChunkV3]:
    lb = LABELS[lang]
    info_id = row["id"]
    title = _info_title(row, lang)
    content = _clean_html(row.get(f"content_{lang}"))
    header = _header(row, lang)
    common = {
        "domain": "information",
        "entity_id": info_id,
        "lang": lang,
        "source_table": "information",
        "source_updated_at": row.get("update_time"),
    }
    metadata = {
        "domain": "information",
        "entity_id": info_id,
        "entity_name": title,
        "information_type": row.get("information_type"),
    }

    chunks = [
        RagChunkV3(
            **common,
            chunk_id=chunk_id(info_id, "identifier"),
            chunk_type="identifier",
            content=header,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=title,
                section="identifier",
                extra=metadata,
            ),
        ),
        RagChunkV3(
            **common,
            chunk_id=chunk_id(info_id, "retrieval"),
            chunk_type="retrieval",
            content=clean_parts([header, content[:1800]]),
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=title,
                section="retrieval",
                extra=metadata,
            ),
        ),
    ]
    if content:
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(info_id, "content", "main"),
                chunk_type="content",
                content=clean_parts([header, f"[{lb['content']}]\n{content}"]),
                searchable=False,
                metadata=with_common_metadata(
                    **common,
                    entity_name=title,
                    section="main",
                    extra=metadata,
                ),
            )
        )
    return chunks


def build_news_chunks(row: dict, lang: Lang) -> list[RagChunkV3]:
    lb = LABELS[lang]
    entity_id = f"news:{row['id']}"
    title = lang_value(row, "title", lang)
    content = clean_parts(
        [
            f"{lb['news']}: {title}",
            f"{lb['type']}: {row.get('information_type')}" if row.get("information_type") else None,
            f"{lb['link']}: {row.get('link')}" if row.get("link") else None,
            f"{lb['new']}: {row.get('is_new')}",
        ]
    )
    common = {
        "domain": "information",
        "entity_id": entity_id,
        "lang": lang,
        "source_table": "news_items",
        "source_updated_at": row.get("update_time"),
    }
    metadata = {
        "domain": "information",
        "entity_id": entity_id,
        "entity_name": title,
        "information_type": row.get("information_type"),
        "url": row.get("link"),
    }
    return [
        RagChunkV3(
            **common,
            chunk_id=chunk_id(entity_id, "retrieval"),
            chunk_type="retrieval",
            content=content,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=title,
                section="retrieval",
                url=row.get("link"),
                extra=metadata,
            ),
        )
    ]


async def build_and_upsert_information(
    info_id: str | None = None,
    limit: int | None = None,
    include_news: bool = True,
    langs: tuple[Lang, ...] = SUPPORTED_LANGS,
    dry_run: bool = False,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await fetch_information_rows(conn, info_id=info_id, limit=limit)
        total = 0
        for lang in langs:
            chunks = []
            for row in rows:
                chunks.extend(build_information_chunks(row, lang))
            if include_news and not info_id:
                for row in await fetch_news_rows(conn, lang):
                    chunks.extend(build_news_chunks(row, lang))

            if dry_run:
                for chunk in chunks[:12]:
                    log.info("[dry-run] %s %s %s", chunk.lang, chunk.chunk_id, chunk.chunk_type)
                    log.info("\n%s", chunk.content[:1200])
                total += len(chunks)
            else:
                total += await upsert_rag_chunks_v3(conn, chunks)
        log.info("[information_builder_v3] rows=%s chunks=%s dry_run=%s", len(rows), total, dry_run)
        return total


def _parse_langs(raw: str) -> tuple[Lang, ...]:
    langs = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = [lang for lang in langs if lang not in SUPPORTED_LANGS]
    if invalid:
        raise ValueError(f"unsupported langs: {invalid}")
    return langs or SUPPORTED_LANGS


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 information RAG chunks.")
    parser.add_argument("--info-id", help="Build a single information id.")
    parser.add_argument("--limit", type=int, help="Build a limited number of information rows.")
    parser.add_argument("--langs", default="ko,en,ja", help="Comma-separated langs: ko,en,ja")
    parser.add_argument("--no-news", action="store_true", help="Skip news_items rows.")
    parser.add_argument("--dry-run", action="store_true", help="Print chunks without upserting.")
    args = parser.parse_args()

    try:
        await build_and_upsert_information(
            info_id=args.info_id,
            limit=args.limit,
            include_news=not args.no_news,
            langs=_parse_langs(args.langs),
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
