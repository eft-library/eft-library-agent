import argparse
import asyncio
import logging
from collections import defaultdict
from typing import Any

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
        "story": "스토리",
        "roadmap": "스토리 로드맵",
        "objectives": "목표",
        "requirements": "요구사항",
        "guide": "가이드",
        "node_type": "노드 유형",
        "contents": "내용",
        "description": "설명",
        "value": "값",
        "next": "다음 단계",
    },
    "en": {
        "story": "Story",
        "roadmap": "Story Roadmap",
        "objectives": "Objectives",
        "requirements": "Requirements",
        "guide": "Guide",
        "node_type": "Node Type",
        "contents": "Contents",
        "description": "Description",
        "value": "Value",
        "next": "Next Steps",
    },
    "ja": {
        "story": "ストーリー",
        "roadmap": "ストーリーロードマップ",
        "objectives": "目標",
        "requirements": "条件",
        "guide": "ガイド",
        "node_type": "ノード種別",
        "contents": "内容",
        "description": "説明",
        "value": "値",
        "next": "次のステップ",
    },
}


def _clean_html(html_text: str | None) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for img in soup.find_all("img"):
        img.decompose()
    return soup.get_text(separator="\n", strip=True)


def _story_title(row: dict, lang: Lang) -> str:
    return lang_value(row, "title", lang)


def _line(label: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return f"{label}: {text}"


def _format_edge_targets(edge: Any) -> list[str]:
    if not edge:
        return []
    if isinstance(edge, str):
        import json

        try:
            edge = json.loads(edge)
        except json.JSONDecodeError:
            return []
    if not isinstance(edge, list):
        return []
    targets = []
    for item in edge:
        if isinstance(item, dict) and item.get("target"):
            targets.append(str(item["target"]))
    return targets


async def fetch_story_rows(conn, story_id: str | None = None, limit: int | None = None):
    args = []
    where = ""
    if story_id:
        args.append(story_id)
        where = f"WHERE id = ${len(args)}"
    limit_clause = ""
    if limit is not None and not story_id:
        args.append(limit)
        limit_clause = f"LIMIT ${len(args)}"
    rows = await conn.fetch(
        f"""
        SELECT *
        FROM story
        {where}
        ORDER BY sort_order NULLS LAST, id
        {limit_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


async def fetch_roadmap_rows(conn, limit: int | None = None):
    args = [limit] if limit is not None else []
    limit_clause = "LIMIT $1" if limit is not None else ""
    rows = await conn.fetch(
        f"""
        SELECT *
        FROM story_roadmap
        ORDER BY id
        {limit_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


async def fetch_roadmap_titles(conn, lang: Lang) -> dict[str, str]:
    rows = await conn.fetch(f"SELECT id, title_{lang}, title_en FROM story_roadmap")
    return {
        row["id"]: (row[f"title_{lang}"] or row["title_en"] or row["id"])
        for row in rows
    }


def build_story_chunks(row: dict, lang: Lang) -> list[RagChunkV3]:
    lb = LABELS[lang]
    story_id = row["id"]
    title = _story_title(row, lang)
    objectives = _clean_html(row.get(f"objectives_{lang}"))
    requirements = _clean_html(row.get(f"requirements_{lang}"))
    guide = _clean_html(row.get(f"guide_{lang}"))
    header = f"{lb['story']}: {title}"
    common = {
        "domain": "story",
        "entity_id": story_id,
        "lang": lang,
        "source_table": "story",
        "source_updated_at": row.get("update_time"),
    }
    metadata = {
        "domain": "story",
        "entity_id": story_id,
        "entity_name": title,
    }

    chunks = [
        RagChunkV3(
            **common,
            chunk_id=chunk_id(story_id, "identifier"),
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
            chunk_id=chunk_id(story_id, "retrieval"),
            chunk_type="retrieval",
            content=clean_parts(
                [
                    header,
                    f"\n[{lb['objectives']}]\n{objectives[:1200]}" if objectives else None,
                    f"\n[{lb['requirements']}]\n{requirements[:800]}" if requirements else None,
                    f"\n[{lb['guide']}]\n{guide[:1200]}" if guide else None,
                ]
            ),
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=title,
                section="retrieval",
                extra=metadata,
            ),
        ),
    ]
    if objectives or requirements:
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(story_id, "content", "main"),
                chunk_type="content",
                content=clean_parts(
                    [
                        header,
                        f"[{lb['objectives']}]\n{objectives}" if objectives else None,
                        f"[{lb['requirements']}]\n{requirements}" if requirements else None,
                    ]
                ),
                searchable=False,
                metadata=with_common_metadata(
                    **common,
                    entity_name=title,
                    section="main",
                    extra=metadata,
                ),
            )
        )
    if guide:
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(story_id, "guide"),
                chunk_type="guide",
                content=clean_parts([header, f"[{lb['guide']}]\n{guide}"]),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=title,
                    section="guide",
                    extra=metadata,
                ),
            )
        )
    return chunks


def build_roadmap_chunks(row: dict, target_titles: dict[str, str], lang: Lang) -> list[RagChunkV3]:
    lb = LABELS[lang]
    entity_id = f"roadmap:{row['id']}"
    title = _story_title(row, lang)
    contents = lang_value(row, "contents", lang)
    desc = lang_value(row, "desc", lang)
    targets = _format_edge_targets(row.get("edge"))
    target_lines = [f"- {target_titles.get(target, target)}" for target in targets]
    content = clean_parts(
        [
            f"{lb['roadmap']}: {title}",
            _line(lb["node_type"], row.get("node_type")),
            f"{lb['contents']}: {contents}" if contents else None,
            f"{lb['description']}: {desc}" if desc else None,
            f"{lb['value']}: {row.get('value_text')}" if row.get("value_text") else None,
            f"[{lb['next']}]\n" + "\n".join(target_lines) if target_lines else None,
        ]
    )
    common = {
        "domain": "story",
        "entity_id": entity_id,
        "lang": lang,
        "source_table": "story_roadmap",
        "source_updated_at": row.get("update_time"),
    }
    metadata = {
        "domain": "story",
        "entity_id": entity_id,
        "entity_name": title,
        "node_type": row.get("node_type"),
        "image": row.get("image"),
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
                image=row.get("image"),
                extra=metadata,
            ),
        )
    ]


async def build_and_upsert_stories(
    story_id: str | None = None,
    limit: int | None = None,
    include_roadmap: bool = True,
    langs: tuple[Lang, ...] = SUPPORTED_LANGS,
    dry_run: bool = False,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await fetch_story_rows(conn, story_id=story_id, limit=limit)
        roadmap_rows = [] if story_id or not include_roadmap else await fetch_roadmap_rows(conn)
        total = 0

        for lang in langs:
            target_titles = await fetch_roadmap_titles(conn, lang) if roadmap_rows else {}
            chunks = []
            for row in rows:
                chunks.extend(build_story_chunks(row, lang))
            for row in roadmap_rows:
                chunks.extend(build_roadmap_chunks(row, target_titles, lang))

            if dry_run:
                for chunk in chunks[:12]:
                    log.info("[dry-run] %s %s %s", chunk.lang, chunk.chunk_id, chunk.chunk_type)
                    log.info("\n%s", chunk.content[:1200])
                total += len(chunks)
            else:
                total += await upsert_rag_chunks_v3(conn, chunks)
        log.info("[story_builder_v3] stories=%s roadmap=%s chunks=%s dry_run=%s", len(rows), len(roadmap_rows), total, dry_run)
        return total


def _parse_langs(raw: str) -> tuple[Lang, ...]:
    langs = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = [lang for lang in langs if lang not in SUPPORTED_LANGS]
    if invalid:
        raise ValueError(f"unsupported langs: {invalid}")
    return langs or SUPPORTED_LANGS


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 story RAG chunks.")
    parser.add_argument("--story-id", help="Build a single story id.")
    parser.add_argument("--limit", type=int, help="Build a limited number of stories.")
    parser.add_argument("--langs", default="ko,en,ja", help="Comma-separated langs: ko,en,ja")
    parser.add_argument("--no-roadmap", action="store_true", help="Skip story_roadmap rows.")
    parser.add_argument("--dry-run", action="store_true", help="Print chunks without upserting.")
    args = parser.parse_args()

    try:
        await build_and_upsert_stories(
            story_id=args.story_id,
            limit=args.limit,
            include_roadmap=not args.no_roadmap,
            langs=_parse_langs(args.langs),
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
