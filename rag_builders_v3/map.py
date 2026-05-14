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
        "map": "지도",
        "aliases": "검색명",
        "sub_areas": "하위 구역",
        "extractions": "탈출구",
        "transits": "트랜짓",
        "boss_spawns": "스폰 보스",
        "quest_objectives": "관련 퀘스트 목표",
        "requirements": "요구사항",
        "tip": "팁",
        "faction": "진영",
        "unlimited": "무제한 사용",
        "one_time": "1회용",
        "yes": "예",
        "no": "아니오",
        "related_info": "관련 정보",
    },
    "en": {
        "map": "Map",
        "aliases": "Search Names",
        "sub_areas": "Sub Areas",
        "extractions": "Extractions",
        "transits": "Transits",
        "boss_spawns": "Boss Spawns",
        "quest_objectives": "Related Quest Objectives",
        "requirements": "Requirements",
        "tip": "Tip",
        "faction": "Faction",
        "unlimited": "Unlimited Use",
        "one_time": "One Time Use",
        "yes": "Yes",
        "no": "No",
        "related_info": "Related Info",
    },
    "ja": {
        "map": "マップ",
        "aliases": "検索名",
        "sub_areas": "サブエリア",
        "extractions": "脱出地点",
        "transits": "トランジット",
        "boss_spawns": "ボススポーン",
        "quest_objectives": "関連クエスト目標",
        "requirements": "条件",
        "tip": "ヒント",
        "faction": "陣営",
        "unlimited": "無制限",
        "one_time": "一回限り",
        "yes": "はい",
        "no": "いいえ",
        "related_info": "関連情報",
    },
}


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


def _map_name(row: dict, lang: Lang) -> tuple[str, list[str]]:
    primary = lang_value(row, "name", lang)
    aliases = [row.get("name_en"), row.get("normalized_name")]
    alias_list = []
    for alias in aliases:
        alias_text = str(alias or "").strip()
        if alias_text and alias_text != primary and alias_text not in alias_list:
            alias_list.append(alias_text)
    return primary, alias_list


def _format_name_rows(rows: list[dict], lang: Lang, *, limit: int = 40) -> str:
    lines = []
    seen = set()
    for row in rows:
        name = lang_value(row, "name", lang) or row.get("name_en") or row.get("id")
        extras = []
        if row.get("spawn_chance") is not None:
            extras.append(f"{_as_text(row['spawn_chance'])}%")
        if row.get("description"):
            extras.append(str(row["description"]).strip())
        key = (name, tuple(extras))
        if key in seen:
            continue
        seen.add(key)
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {name}{suffix}")
        if len(lines) >= limit:
            break
    if len(seen) > limit:
        lines.append(f"- ... +{len(seen) - limit}")
    return "\n".join(lines)


def _format_map_points(rows: list[dict], lang: Lang, *, limit: int = 60) -> str:
    lb = LABELS[lang]
    sections = []
    for row in rows[:limit]:
        name = lang_value(row, "name", lang) or row.get("name_en") or row.get("id")
        lines = [f"- {name}"]
        if row.get("faction"):
            lines.append(f"  - {lb['faction']}: {row['faction']}")
        lines.append(
            f"  - {lb['unlimited']}: {lb['yes'] if row.get('is_unlimited_use') else lb['no']}"
        )
        lines.append(
            f"  - {lb['one_time']}: {lb['yes'] if row.get('is_one_time_use') else lb['no']}"
        )
        requirements = _clean_html(row.get(f"requirements_{lang}"))
        tip = _clean_html(row.get(f"tip_{lang}"))
        if requirements:
            lines.append(f"  - {lb['requirements']}: {requirements}")
        if tip:
            lines.append(f"  - {lb['tip']}: {tip}")
        sections.append("\n".join(lines))
    if len(rows) > limit:
        sections.append(f"- ... +{len(rows) - limit}")
    return "\n".join(sections)


def _format_map_point_names(rows: list[dict], lang: Lang, *, limit: int = 30) -> str:
    lines = []
    seen = set()
    for row in rows:
        name = lang_value(row, "name", lang) or row.get("name_en") or row.get("id")
        if not name or name in seen:
            continue
        seen.add(name)
        lines.append(f"- {name}")
        if len(lines) >= limit:
            break
    if len(seen) > limit:
        lines.append(f"- ... +{len(seen) - limit}")
    return "\n".join(lines)


async def fetch_map_rows(conn, map_id: str | None = None, limit: int | None = None):
    where_parts = ["m.map_depth = 1", "m.is_use IS TRUE"]
    args: list[Any] = []
    if map_id:
        args.append(map_id)
        where_parts.append(f"m.id = ${len(args)}")
    limit_clause = ""
    if limit is not None and not map_id:
        args.append(limit)
        limit_clause = f"LIMIT ${len(args)}"
    rows = await conn.fetch(
        f"""
        SELECT m.*
        FROM maps m
        WHERE {' AND '.join(where_parts)}
        ORDER BY m.sort_order NULLS LAST, m.id
        {limit_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


async def fetch_map_details(conn, map_ids: list[str], lang: Lang) -> dict[str, dict[str, list[dict]]]:
    details: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    if not map_ids:
        return details

    name_col = f"name_{lang}"
    desc_col = f"description_{lang}"

    queries = {
        "sub_areas": f"""
            SELECT parent_map_id AS map_id, id, {name_col}, name_en
            FROM maps
            WHERE parent_map_id = ANY($1::text[])
              AND is_use IS TRUE
            ORDER BY sort_order NULLS LAST, id
        """,
        "extractions": f"""
            SELECT map_id, id, point_type, {name_col}, name_en, faction,
                   is_unlimited_use, is_one_time_use,
                   requirements_en, requirements_ko, requirements_ja,
                   tip_en, tip_ko, tip_ja
            FROM map_points
            WHERE map_id = ANY($1::text[])
              AND point_type = 'extraction'
            ORDER BY map_id, faction NULLS LAST, sort_order NULLS LAST, {name_col}
        """,
        "transits": f"""
            SELECT map_id, id, point_type, {name_col}, name_en, faction,
                   is_unlimited_use, is_one_time_use,
                   requirements_en, requirements_ko, requirements_ja,
                   tip_en, tip_ko, tip_ja
            FROM map_points
            WHERE map_id = ANY($1::text[])
              AND point_type = 'transit'
            ORDER BY map_id, faction NULLS LAST, sort_order NULLS LAST, {name_col}
        """,
        "boss_spawns": f"""
            SELECT bs.map_id, b.id, b.{name_col}, b.name_en,
                   round((bs.spawn_chance * 100)::numeric, 2) AS spawn_chance
            FROM boss_spawn bs
            JOIN bosses b ON b.id = bs.boss_id
            WHERE bs.map_id = ANY($1::text[])
            ORDER BY bs.map_id, bs.spawn_chance DESC, b.sort_order NULLS LAST, b.{name_col}
        """,
        "quest_objectives": f"""
            SELECT qm.map_id, q.id, q.{name_col}, q.name_en,
                   qo.{desc_col} AS description
            FROM quest_objective_maps qm
            JOIN quest_objectives qo ON qo.objective_id = qm.objective_id
            JOIN quests q ON q.id = qo.quest_id
            WHERE qm.map_id = ANY($1::text[])
            ORDER BY qm.map_id, q.sort_order NULLS LAST, qo.sort_order NULLS LAST, q.{name_col}
        """,
    }

    for detail_key, sql in queries.items():
        rows = await conn.fetch(sql, map_ids)
        for row in rows:
            details[row["map_id"]][detail_key].append(dict(row))

    return details


def _header(row: dict, lang: Lang) -> str:
    lb = LABELS[lang]
    map_name, aliases = _map_name(row, lang)
    return clean_parts(
        [
            f"{lb['map']}: {map_name}",
            f"{lb['aliases']}: {', '.join(aliases)}" if aliases else None,
        ]
    )


def _base_metadata(row: dict, map_name: str, lang: Lang) -> dict:
    return {
        "domain": "map",
        "entity_id": row["id"],
        "entity_name": map_name,
        "map_depth": row.get("map_depth"),
        "normalized_name": row.get("normalized_name"),
    }


def build_map_chunks(row: dict, details: dict[str, list[dict]], lang: Lang) -> list[RagChunkV3]:
    lb = LABELS[lang]
    map_id = row["id"]
    map_name, _aliases = _map_name(row, lang)
    common = {
        "domain": "map",
        "entity_id": map_id,
        "lang": lang,
        "source_table": "maps",
        "source_updated_at": row.get("update_time"),
    }
    metadata_base = _base_metadata(row, map_name, lang)
    image = row.get(f"mot_image_{lang}") or row.get("three_image")
    header = _header(row, lang)

    sub_areas = details.get("sub_areas", [])
    extractions = details.get("extractions", [])
    transits = details.get("transits", [])
    boss_spawns = details.get("boss_spawns", [])
    quest_objectives = details.get("quest_objectives", [])

    chunks: list[RagChunkV3] = []
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(map_id, "identifier"),
            chunk_type="identifier",
            content=header,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=map_name,
                section="identifier",
                image=image,
                extra=metadata_base,
            ),
        )
    )

    relation_hints = []
    for key in ["sub_areas", "extractions", "transits", "boss_spawns", "quest_objectives"]:
        if details.get(key):
            relation_hints.append(lb[key])

    retrieval_content = clean_parts(
        [
            header,
            f"\n[{lb['sub_areas']}]\n" + _format_name_rows(sub_areas, lang, limit=20)
            if sub_areas
            else None,
            f"\n[{lb['extractions']}]\n"
            + _format_map_point_names(extractions, lang, limit=20)
            if extractions
            else None,
            f"\n[{lb['transits']}]\n" + _format_map_point_names(transits, lang, limit=20)
            if transits
            else None,
            f"\n[{lb['boss_spawns']}]\n" + _format_name_rows(boss_spawns, lang, limit=12)
            if boss_spawns
            else None,
            f"\n[{lb['related_info']}]\n" + "\n".join(f"- {hint}" for hint in relation_hints)
            if relation_hints
            else None,
        ]
    )
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(map_id, "retrieval"),
            chunk_type="retrieval",
            content=retrieval_content,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=map_name,
                section="retrieval",
                image=image,
                extra=metadata_base,
            ),
        )
    )

    content_sections = [
        header,
        f"\n[{lb['sub_areas']}]\n" + _format_name_rows(sub_areas, lang)
        if sub_areas
        else None,
        f"\n[{lb['boss_spawns']}]\n" + _format_name_rows(boss_spawns, lang)
        if boss_spawns
        else None,
    ]
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(map_id, "content", "main"),
            chunk_type="content",
            content=clean_parts(content_sections),
            searchable=False,
            metadata=with_common_metadata(
                **common,
                entity_name=map_name,
                section="main",
                image=image,
                extra=metadata_base,
            ),
        )
    )

    relation_sections = [
        f"[{lb['extractions']}]\n" + _format_map_points(extractions, lang)
        if extractions
        else None,
        f"[{lb['transits']}]\n" + _format_map_points(transits, lang)
        if transits
        else None,
        f"[{lb['quest_objectives']}]\n"
        + _format_name_rows(quest_objectives, lang, limit=60)
        if quest_objectives
        else None,
        f"[{lb['boss_spawns']}]\n" + _format_name_rows(boss_spawns, lang)
        if boss_spawns
        else None,
    ]
    if any(relation_sections):
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(map_id, "relation"),
                chunk_type="relation",
                content=clean_parts([f"{lb['map']}: {map_name}", *relation_sections]),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=map_name,
                    section="relation",
                    image=image,
                    extra={
                        **metadata_base,
                        "relation_counts": {
                            "extractions": len(extractions),
                            "transits": len(transits),
                            "quest_objectives": len(quest_objectives),
                            "boss_spawns": len(boss_spawns),
                        },
                    },
                ),
            )
        )

    return chunks


async def build_and_upsert_maps(
    map_id: str | None = None,
    limit: int | None = None,
    langs: tuple[Lang, ...] = SUPPORTED_LANGS,
    dry_run: bool = False,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await fetch_map_rows(conn, map_id=map_id, limit=limit)
        map_ids = [row["id"] for row in rows]
        total = 0

        for lang in langs:
            details = await fetch_map_details(conn, map_ids, lang)
            chunks = []
            for row in rows:
                chunks.extend(build_map_chunks(row, details.get(row["id"], {}), lang))

            if dry_run:
                for chunk in chunks[:12]:
                    log.info("[dry-run] %s %s %s", chunk.lang, chunk.chunk_id, chunk.chunk_type)
                    log.info("\n%s", chunk.content[:1200])
                total += len(chunks)
            else:
                total += await upsert_rag_chunks_v3(conn, chunks)

        log.info("[map_builder_v3] maps=%s chunks=%s dry_run=%s", len(rows), total, dry_run)
        return total


def _parse_langs(raw: str) -> tuple[Lang, ...]:
    langs = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = [lang for lang in langs if lang not in SUPPORTED_LANGS]
    if invalid:
        raise ValueError(f"unsupported langs: {invalid}")
    return langs or SUPPORTED_LANGS


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 map RAG chunks.")
    parser.add_argument("--map-id", help="Build a single map id.")
    parser.add_argument("--limit", type=int, help="Build a limited number of maps.")
    parser.add_argument("--langs", default="ko,en,ja", help="Comma-separated langs: ko,en,ja")
    parser.add_argument("--dry-run", action="store_true", help="Print chunks without upserting.")
    args = parser.parse_args()

    try:
        await build_and_upsert_maps(
            map_id=args.map_id,
            limit=args.limit,
            langs=_parse_langs(args.langs),
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
