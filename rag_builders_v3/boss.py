import argparse
import asyncio
import logging
from collections import defaultdict
from decimal import Decimal
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
        "boss": "보스",
        "aliases": "검색명",
        "faction": "진영",
        "total_health": "총 체력",
        "body_health": "부위별 체력",
        "spawns": "스폰 위치",
        "drops": "드랍 아이템",
        "followers": "부하/가드",
        "guide": "가이드",
        "related_info": "관련 정보",
        "head": "머리",
        "thorax": "흉부",
        "stomach": "복부",
        "left_arm": "왼팔",
        "right_arm": "오른팔",
        "left_leg": "왼다리",
        "right_leg": "오른다리",
    },
    "en": {
        "boss": "Boss",
        "aliases": "Search Names",
        "faction": "Faction",
        "total_health": "Total Health",
        "body_health": "Body Health",
        "spawns": "Spawn Locations",
        "drops": "Drop Items",
        "followers": "Followers/Guards",
        "guide": "Guide",
        "related_info": "Related Info",
        "head": "Head",
        "thorax": "Thorax",
        "stomach": "Stomach",
        "left_arm": "Left Arm",
        "right_arm": "Right Arm",
        "left_leg": "Left Leg",
        "right_leg": "Right Leg",
    },
    "ja": {
        "boss": "ボス",
        "aliases": "検索名",
        "faction": "陣営",
        "total_health": "総体力",
        "body_health": "部位別体力",
        "spawns": "スポーン位置",
        "drops": "ドロップアイテム",
        "followers": "護衛",
        "guide": "ガイド",
        "related_info": "関連情報",
        "head": "頭",
        "thorax": "胸部",
        "stomach": "腹部",
        "left_arm": "左腕",
        "right_arm": "右腕",
        "left_leg": "左脚",
        "right_leg": "右脚",
    },
}

BODY_HP_FIELDS = [
    ("head", "head_hp"),
    ("thorax", "thorax_hp"),
    ("stomach", "stomach_hp"),
    ("left_arm", "left_arm_hp"),
    ("right_arm", "right_arm_hp"),
    ("left_leg", "left_leg_hp"),
    ("right_leg", "right_leg_hp"),
]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        normalized = value.normalize()
        text = format(normalized, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    return str(value).strip()


def _line(label: str, value: Any) -> str | None:
    text = _as_text(value)
    if not text:
        return None
    return f"{label}: {text}"


def _clean_html(html_text: str | None) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for img in soup.find_all("img"):
        img.decompose()
    return soup.get_text(separator="\n", strip=True)


def _boss_name(row: dict, lang: Lang) -> tuple[str, list[str]]:
    primary = lang_value(row, "name", lang)
    aliases = [row.get("name_en"), row.get("normalized_name")]
    alias_list = []
    for alias in aliases:
        alias_text = str(alias or "").strip()
        if alias_text and alias_text != primary and alias_text not in alias_list:
            alias_list.append(alias_text)
    return primary, alias_list


def _format_percent(value: Any) -> str:
    if value is None:
        return ""
    numeric = Decimal(str(value)) * Decimal("100")
    text = format(numeric.normalize(), "f")
    text = text.rstrip("0").rstrip(".") if "." in text else text
    return f"{text}%"


def _format_name_rows(
    rows: list[dict],
    lang: Lang,
    *,
    include_quantity: bool = True,
    include_health: bool = False,
    include_spawn_chance: bool = False,
    limit: int = 40,
) -> str:
    lines = []
    seen = set()
    for row in rows:
        name = lang_value(row, "name", lang) or row.get("name_en") or row.get("id") or ""
        extras = []
        if include_quantity and row.get("quantity"):
            extras.append(f"x{_as_text(row['quantity'])}")
        if include_health and row.get("health_total"):
            extras.append(f"HP {_as_text(row['health_total'])}")
        if include_spawn_chance and row.get("spawn_chance") is not None:
            extras.append(_format_percent(row["spawn_chance"]))
        if row.get("faction"):
            extras.append(str(row["faction"]).strip())
        key = (name, tuple(extras))
        if key in seen:
            continue
        seen.add(key)
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {name}{suffix}")
        if len(lines) >= limit:
            break
    if len(rows) > limit:
        lines.append(f"- ... +{len(rows) - limit}")
    return "\n".join(lines)


def _format_body_health(row: dict, lang: Lang) -> str:
    lb = LABELS[lang]
    lines = []
    for label_key, field in BODY_HP_FIELDS:
        value = row.get(field)
        if value is not None:
            lines.append(f"- {lb[label_key]}: {_as_text(value)}")
    return "\n".join(lines)


async def fetch_boss_rows(conn, boss_id: str | None = None, limit: int | None = None):
    where_parts = ["b.is_boss IS TRUE"]
    args: list[Any] = []
    if boss_id:
        args.append(boss_id)
        where_parts.append(f"b.id = ${len(args)}")
    limit_clause = ""
    if limit is not None and not boss_id:
        args.append(limit)
        limit_clause = f"LIMIT ${len(args)}"

    rows = await conn.fetch(
        f"""
        SELECT b.*
        FROM bosses b
        WHERE {' AND '.join(where_parts)}
        ORDER BY b.sort_order NULLS LAST, b.id
        {limit_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


async def fetch_boss_details(conn, boss_ids: list[str], lang: Lang) -> dict[str, dict[str, list[dict]]]:
    details: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    if not boss_ids:
        return details

    name_col = f"name_{lang}"

    queries = {
        "spawns": f"""
            SELECT bs.boss_id, m.id, m.{name_col}, m.name_en, bs.spawn_chance
            FROM boss_spawn bs
            JOIN maps m ON m.id = bs.map_id
            WHERE bs.boss_id = ANY($1::text[])
            ORDER BY bs.boss_id, bs.spawn_chance DESC, m.sort_order NULLS LAST, m.{name_col}
        """,
        "drops": f"""
            SELECT bi.boss_id, i.id, i.{name_col}, i.name_en, bi.quantity
            FROM boss_item bi
            JOIN items i ON i.id = bi.item_id
            WHERE bi.boss_id = ANY($1::text[])
            ORDER BY bi.boss_id, bi.sort_order NULLS LAST, i.{name_col}
        """,
        "followers": f"""
            SELECT parent_boss_id AS boss_id, id, {name_col}, name_en, faction, health_total
            FROM bosses
            WHERE parent_boss_id = ANY($1::text[])
            ORDER BY parent_boss_id, sort_order NULLS LAST, {name_col}
        """,
    }

    for detail_key, sql in queries.items():
        rows = await conn.fetch(sql, boss_ids)
        for row in rows:
            details[row["boss_id"]][detail_key].append(dict(row))

    return details


def _header(row: dict, lang: Lang) -> str:
    lb = LABELS[lang]
    boss_name, aliases = _boss_name(row, lang)
    return clean_parts(
        [
            f"{lb['boss']}: {boss_name}",
            f"{lb['aliases']}: {', '.join(aliases)}" if aliases else None,
            _line(lb["faction"], row.get("faction")),
        ]
    )


def _base_metadata(row: dict, boss_name: str) -> dict:
    return {
        "domain": "boss",
        "entity_id": row["id"],
        "entity_name": boss_name,
        "faction": row.get("faction"),
        "normalized_name": row.get("normalized_name"),
        "health_total": row.get("health_total"),
        "is_boss": row.get("is_boss"),
    }


def build_boss_chunks(row: dict, details: dict[str, list[dict]], lang: Lang) -> list[RagChunkV3]:
    lb = LABELS[lang]
    boss_id = row["id"]
    boss_name, _aliases = _boss_name(row, lang)
    common = {
        "domain": "boss",
        "entity_id": boss_id,
        "lang": lang,
        "source_table": "bosses",
        "source_updated_at": row.get("update_time"),
    }
    metadata_base = _base_metadata(row, boss_name)
    image = row.get("image")
    header = _header(row, lang)

    spawns = details.get("spawns", [])
    drops = details.get("drops", [])
    followers = details.get("followers", [])
    body_health = _format_body_health(row, lang)
    guide = _clean_html(row.get(f"guide_{lang}"))

    chunks: list[RagChunkV3] = []
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(boss_id, "identifier"),
            chunk_type="identifier",
            content=header,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=boss_name,
                section="identifier",
                image=image,
                extra=metadata_base,
            ),
        )
    )

    relation_hints = []
    for key in ["spawns", "drops", "followers"]:
        if details.get(key):
            relation_hints.append(lb[key])

    retrieval_content = clean_parts(
        [
            header,
            _line(lb["total_health"], row.get("health_total")),
            f"\n[{lb['spawns']}]\n"
            + _format_name_rows(spawns, lang, include_spawn_chance=True, include_quantity=False, limit=20)
            if spawns
            else None,
            f"\n[{lb['drops']}]\n" + _format_name_rows(drops, lang, limit=25)
            if drops
            else None,
            f"\n[{lb['followers']}]\n"
            + _format_name_rows(followers, lang, include_quantity=False, include_health=True, limit=20)
            if followers
            else None,
            f"\n[{lb['related_info']}]\n" + "\n".join(f"- {hint}" for hint in relation_hints)
            if relation_hints
            else None,
        ]
    )
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(boss_id, "retrieval"),
            chunk_type="retrieval",
            content=retrieval_content,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=boss_name,
                section="retrieval",
                image=image,
                extra=metadata_base,
            ),
        )
    )

    content_sections = [
        header,
        _line(lb["total_health"], row.get("health_total")),
        f"\n[{lb['body_health']}]\n{body_health}" if body_health else None,
        f"\n[{lb['spawns']}]\n"
        + _format_name_rows(spawns, lang, include_spawn_chance=True, include_quantity=False)
        if spawns
        else None,
    ]
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(boss_id, "content", "main"),
            chunk_type="content",
            content=clean_parts(content_sections),
            searchable=False,
            metadata=with_common_metadata(
                **common,
                entity_name=boss_name,
                section="main",
                image=image,
                extra=metadata_base,
            ),
        )
    )

    relation_sections = [
        f"[{lb['spawns']}]\n"
        + _format_name_rows(spawns, lang, include_spawn_chance=True, include_quantity=False)
        if spawns
        else None,
        f"[{lb['drops']}]\n" + _format_name_rows(drops, lang, limit=80)
        if drops
        else None,
        f"[{lb['followers']}]\n"
        + _format_name_rows(followers, lang, include_quantity=False, include_health=True)
        if followers
        else None,
    ]
    if any(relation_sections):
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(boss_id, "relation"),
                chunk_type="relation",
                content=clean_parts([f"{lb['boss']}: {boss_name}", *relation_sections]),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=boss_name,
                    section="relation",
                    image=image,
                    extra={
                        **metadata_base,
                        "relation_counts": {
                            "spawns": len(spawns),
                            "drops": len(drops),
                            "followers": len(followers),
                        },
                    },
                ),
            )
        )

    if guide:
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(boss_id, "guide"),
                chunk_type="guide",
                content=clean_parts([f"{lb['boss']}: {boss_name}", f"[{lb['guide']}]\n{guide}"]),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=boss_name,
                    section="guide",
                    image=image,
                    extra=metadata_base,
                ),
            )
        )

    return chunks


async def build_and_upsert_bosses(
    boss_id: str | None = None,
    limit: int | None = None,
    langs: tuple[Lang, ...] = SUPPORTED_LANGS,
    dry_run: bool = False,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await fetch_boss_rows(conn, boss_id=boss_id, limit=limit)
        boss_ids = [row["id"] for row in rows]
        total = 0

        for lang in langs:
            details = await fetch_boss_details(conn, boss_ids, lang)
            chunks = []
            for row in rows:
                chunks.extend(build_boss_chunks(row, details.get(row["id"], {}), lang))

            if dry_run:
                for chunk in chunks[:12]:
                    log.info("[dry-run] %s %s %s", chunk.lang, chunk.chunk_id, chunk.chunk_type)
                    log.info("\n%s", chunk.content[:1200])
                total += len(chunks)
            else:
                total += await upsert_rag_chunks_v3(conn, chunks)

        log.info("[boss_builder_v3] bosses=%s chunks=%s dry_run=%s", len(rows), total, dry_run)
        return total


def _parse_langs(raw: str) -> tuple[Lang, ...]:
    langs = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = [lang for lang in langs if lang not in SUPPORTED_LANGS]
    if invalid:
        raise ValueError(f"unsupported langs: {invalid}")
    return langs or SUPPORTED_LANGS


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 boss RAG chunks.")
    parser.add_argument("--boss-id", help="Build a single boss id.")
    parser.add_argument("--limit", type=int, help="Build a limited number of bosses.")
    parser.add_argument("--langs", default="ko,en,ja", help="Comma-separated langs: ko,en,ja")
    parser.add_argument("--dry-run", action="store_true", help="Print chunks without upserting.")
    args = parser.parse_args()

    try:
        await build_and_upsert_bosses(
            boss_id=args.boss_id,
            limit=args.limit,
            langs=_parse_langs(args.langs),
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
