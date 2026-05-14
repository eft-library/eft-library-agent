import argparse
import asyncio
import logging
from collections import defaultdict
from decimal import Decimal
from typing import Any

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
        "hideout": "은신처",
        "station": "은신처 시설",
        "aliases": "검색명",
        "levels": "레벨",
        "level": "레벨",
        "construction_time": "건설 시간",
        "requirements": "업그레이드 요구사항",
        "required_items": "필요 아이템",
        "required_traders": "필요 상인",
        "required_stations": "필요 시설",
        "required_skills": "필요 스킬",
        "crafts": "제작",
        "craft_inputs": "제작 재료",
        "bonuses": "보너스",
        "related_info": "관련 정보",
        "in_raid": "인레이드",
    },
    "en": {
        "hideout": "Hideout",
        "station": "Hideout Station",
        "aliases": "Search Names",
        "levels": "Levels",
        "level": "Level",
        "construction_time": "Construction Time",
        "requirements": "Upgrade Requirements",
        "required_items": "Required Items",
        "required_traders": "Required Traders",
        "required_stations": "Required Stations",
        "required_skills": "Required Skills",
        "crafts": "Crafts",
        "craft_inputs": "Craft Inputs",
        "bonuses": "Bonuses",
        "related_info": "Related Info",
        "in_raid": "Found in Raid",
    },
    "ja": {
        "hideout": "隠れ家",
        "station": "隠れ家設備",
        "aliases": "検索名",
        "levels": "レベル",
        "level": "レベル",
        "construction_time": "建設時間",
        "requirements": "アップグレード条件",
        "required_items": "必要アイテム",
        "required_traders": "必要トレーダー",
        "required_stations": "必要設備",
        "required_skills": "必要スキル",
        "crafts": "クラフト",
        "craft_inputs": "クラフト材料",
        "bonuses": "ボーナス",
        "related_info": "関連情報",
        "in_raid": "レイド内入手",
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


def _format_duration(seconds: Any) -> str:
    if seconds is None:
        return ""
    try:
        total = int(Decimal(str(seconds)))
    except Exception:
        return _as_text(seconds)
    if total <= 0:
        return "0"
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _seconds = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    return " ".join(parts) or f"{total}s"


def _station_name(row: dict, lang: Lang) -> tuple[str, list[str]]:
    primary = lang_value(row, "name", lang)
    aliases = [row.get("name_en"), row.get("normalized_name")]
    alias_list = []
    for alias in aliases:
        alias_text = str(alias or "").strip()
        if alias_text and alias_text != primary and alias_text not in alias_list:
            alias_list.append(alias_text)
    return primary, alias_list


def _header(row: dict, lang: Lang) -> str:
    lb = LABELS[lang]
    name, aliases = _station_name(row, lang)
    return clean_parts(
        [
            f"{lb['station']}: {name}",
            f"{lb['aliases']}: {', '.join(aliases)}" if aliases else None,
        ]
    )


def _format_name_rows(
    rows: list[dict],
    lang: Lang,
    *,
    name_prefix: str = "name",
    include_quantity: bool = True,
    include_level: bool = True,
    limit: int = 50,
) -> str:
    lines = []
    seen = set()
    for row in rows:
        name = lang_value(row, name_prefix, lang) or row.get(f"{name_prefix}_en") or row.get("id") or ""
        aliases = str(row.get("aliases") or "").strip()
        if aliases:
            name = f"{name} ({aliases})"
        extras = []
        if include_quantity and row.get("quantity"):
            extras.append(f"x{_as_text(row['quantity'])}")
        if include_level and row.get("hideout_level"):
            extras.append(f"Lv.{row['hideout_level']}")
        if row.get("trader_level"):
            extras.append(f"LL{row['trader_level']}")
        if row.get("station_level"):
            extras.append(f"Lv.{row['station_level']}")
        if row.get("require_level"):
            extras.append(f"Lv.{row['require_level']}")
        if row.get("in_raid"):
            extras.append(LABELS[lang]["in_raid"])
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


def _name_with_aliases(row: dict, lang: Lang) -> str:
    name = lang_value(row, "name", lang) or row.get("name_en") or row.get("id") or ""
    aliases = str(row.get("aliases") or "").strip()
    if aliases:
        return f"{name} ({aliases})"
    return name


def _group_by_level(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        level = row.get("hideout_level")
        if level is not None:
            grouped[int(level)].append(row)
    return grouped


def _format_level_overview(levels: list[dict], lang: Lang) -> str:
    lb = LABELS[lang]
    lines = []
    for level in levels:
        parts = [f"{lb['level']} {level['hideout_level']}"]
        duration = _format_duration(level.get("construction_time"))
        if duration:
            parts.append(f"{lb['construction_time']}: {duration}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def _format_requirements(details: dict[str, list[dict]], lang: Lang, *, limit: int = 80) -> str:
    lb = LABELS[lang]
    sections = []
    for key, label in [
        ("item_requirements", lb["required_items"]),
        ("trader_requirements", lb["required_traders"]),
        ("station_requirements", lb["required_stations"]),
        ("skill_requirements", lb["required_skills"]),
    ]:
        rows = details.get(key, [])
        if rows:
            sections.append(f"[{label}]\n" + _format_name_rows(rows, lang, limit=limit))
    return "\n".join(sections)


def _format_bonuses(rows: list[dict], lang: Lang, *, limit: int = 50) -> str:
    lines = []
    seen = set()
    for row in rows:
        name = lang_value(row, "name", lang) or row.get("name_en") or row.get("bonus_type") or ""
        extras = []
        if row.get("bonus_type"):
            extras.append(str(row["bonus_type"]))
        skill_name = lang_value(row, "skill_name", lang) or row.get("skill_name_en")
        if skill_name:
            extras.append(skill_name)
        if row.get("bonus_value") is not None:
            extras.append(_as_text(row["bonus_value"]))
        if row.get("hideout_level"):
            extras.append(f"Lv.{row['hideout_level']}")
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


def _format_crafts(crafts: list[dict], craft_inputs: dict[str, list[dict]], lang: Lang, *, limit: int = 80) -> str:
    lines = []
    seen = set()
    for craft in crafts:
        reward = lang_value(craft, "reward_name", lang) or craft.get("reward_name_en") or craft.get("reward_item_id")
        reward_aliases = str(craft.get("reward_aliases") or "").strip()
        if reward_aliases:
            reward = f"{reward} ({reward_aliases})"
        extras = []
        if craft.get("reward_quantity"):
            extras.append(f"x{_as_text(craft['reward_quantity'])}")
        if craft.get("hideout_level"):
            extras.append(f"Lv.{craft['hideout_level']}")
        duration = _format_duration(craft.get("duration"))
        if duration:
            extras.append(duration)
        suffix = f" ({', '.join(extras)})" if extras else ""
        line = f"- {reward}{suffix}"
        inputs = craft_inputs.get(craft["id"], [])
        if inputs:
            input_text = ", ".join(
                f"{_name_with_aliases(row, lang)} x{_as_text(row['quantity'])}"
                for row in inputs[:8]
            )
            if len(inputs) > 8:
                input_text += f", ... +{len(inputs) - 8}"
            line += f"\n  - {LABELS[lang]['craft_inputs']}: {input_text}"
        else:
            input_text = ""
        key = (reward, tuple(extras), input_text)
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
        if len(lines) >= limit:
            break
    if len(seen) < len(crafts):
        remaining = max(len(crafts) - len(seen), len(crafts) - limit)
        if remaining > 0:
            lines.append(f"- ... +{remaining}")
    return "\n".join(lines)


def _format_craft_outputs(crafts: list[dict], lang: Lang, *, limit: int = 120) -> str:
    lines = []
    seen = set()
    for craft in crafts:
        reward = lang_value(craft, "reward_name", lang) or craft.get("reward_name_en") or craft.get("reward_item_id")
        reward_aliases = str(craft.get("reward_aliases") or "").strip()
        if reward_aliases:
            reward = f"{reward} ({reward_aliases})"
        extras = []
        if craft.get("reward_quantity"):
            extras.append(f"x{_as_text(craft['reward_quantity'])}")
        if craft.get("hideout_level"):
            extras.append(f"Lv.{craft['hideout_level']}")
        key = (reward, tuple(extras))
        if key in seen:
            continue
        seen.add(key)
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {reward}{suffix}")
        if len(lines) >= limit:
            break
    if len(crafts) > limit:
        lines.append(f"- ... +{len(crafts) - limit}")
    return "\n".join(lines)


async def fetch_hideout_rows(conn, station_id: str | None = None, limit: int | None = None):
    args: list[Any] = []
    where = ""
    if station_id:
        args.append(station_id)
        where = f"WHERE hm.id = ${len(args)}"
    limit_clause = ""
    if limit is not None and not station_id:
        args.append(limit)
        limit_clause = f"LIMIT ${len(args)}"
    rows = await conn.fetch(
        f"""
        SELECT hm.*
        FROM hideout_master hm
        {where}
        ORDER BY hm.name_en, hm.id
        {limit_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


async def fetch_hideout_details(conn, station_ids: list[str], lang: Lang) -> dict[str, dict[str, list[dict]]]:
    details: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    if not station_ids:
        return details

    name_col = f"name_{lang}"

    queries = {
        "levels": """
            SELECT master_id, id, hideout_level, construction_time
            FROM hideout_levels
            WHERE master_id = ANY($1::text[])
            ORDER BY master_id, hideout_level
        """,
        "item_requirements": f"""
            SELECT hl.master_id, hl.hideout_level, i.id, i.{name_col}, i.name_en,
                   hir.quantity, hir.in_raid, ia.aliases
            FROM hideout_item_require hir
            JOIN hideout_levels hl ON hl.id = hir.hideout_level_id
            JOIN items i ON i.id = hir.item_id
            LEFT JOIN LATERAL (
                SELECT string_agg(a.alias, ', ' ORDER BY a.alias) AS aliases
                FROM rag_entity_aliases_v3 a
                WHERE a.domain = 'item'
                  AND a.entity_id = i.id
                  AND a.lang = '{lang}'
                  AND a.status = 'approved'
            ) ia ON true
            WHERE hl.master_id = ANY($1::text[])
            ORDER BY hl.master_id, hl.hideout_level, i.{name_col}
        """,
        "trader_requirements": f"""
            SELECT hl.master_id, hl.hideout_level, t.id, t.{name_col}, t.name_en,
                   htr.trader_level
            FROM hideout_trader_require htr
            JOIN hideout_levels hl ON hl.id = htr.hideout_level_id
            JOIN traders t ON t.id = htr.trader_id
            WHERE hl.master_id = ANY($1::text[])
            ORDER BY hl.master_id, hl.hideout_level, t.{name_col}
        """,
        "station_requirements": f"""
            SELECT hl.master_id, hl.hideout_level, hm.id, hm.{name_col}, hm.name_en,
                   hsr.station_level
            FROM hideout_station_require hsr
            JOIN hideout_levels hl ON hl.id = hsr.hideout_level_id
            JOIN hideout_master hm ON hm.id = hsr.require_master_id
            WHERE hl.master_id = ANY($1::text[])
            ORDER BY hl.master_id, hl.hideout_level, hm.{name_col}
        """,
        "skill_requirements": f"""
            SELECT hl.master_id, hl.hideout_level, hsr.id, hsr.{name_col}, hsr.name_en,
                   hsr.require_level
            FROM hideout_skill_require hsr
            JOIN hideout_levels hl ON hl.id = hsr.hideout_level_id
            WHERE hl.master_id = ANY($1::text[])
            ORDER BY hl.master_id, hl.hideout_level, hsr.{name_col}
        """,
        "bonuses": f"""
            SELECT hl.master_id, hl.hideout_level, hb.id, hb.bonus_type,
                   hb.{name_col}, hb.name_en,
                   hb.skill_name_{lang}, hb.skill_name_en,
                   hb.bonus_value
            FROM hideout_bonus hb
            JOIN hideout_levels hl ON hl.id = hb.hideout_level_id
            WHERE hl.master_id = ANY($1::text[])
            ORDER BY hl.master_id, hl.hideout_level, hb.bonus_type, hb.{name_col}
        """,
        "crafts": f"""
            SELECT hl.master_id, hl.hideout_level, hc.id, hc.reward_item_id,
                   i.{name_col} AS reward_name_{lang}, i.name_en AS reward_name_en,
                   ria.aliases AS reward_aliases, hc.duration, hc.reward_quantity
            FROM hideout_crafts hc
            JOIN hideout_levels hl ON hl.id = hc.hideout_level_id
            JOIN items i ON i.id = hc.reward_item_id
            LEFT JOIN LATERAL (
                SELECT string_agg(a.alias, ', ' ORDER BY a.alias) AS aliases
                FROM rag_entity_aliases_v3 a
                WHERE a.domain = 'item'
                  AND a.entity_id = i.id
                  AND a.lang = '{lang}'
                  AND a.status = 'approved'
            ) ria ON true
            WHERE hl.master_id = ANY($1::text[])
            ORDER BY hl.master_id, hl.hideout_level, i.{name_col}
        """,
        "craft_inputs": f"""
            SELECT hl.master_id, hcri.craft_id, i.id, i.{name_col}, i.name_en,
                   hcri.quantity, ia.aliases
            FROM hideout_craft_require_items hcri
            JOIN hideout_crafts hc ON hc.id = hcri.craft_id
            JOIN hideout_levels hl ON hl.id = hc.hideout_level_id
            JOIN items i ON i.id = hcri.item_id
            LEFT JOIN LATERAL (
                SELECT string_agg(a.alias, ', ' ORDER BY a.alias) AS aliases
                FROM rag_entity_aliases_v3 a
                WHERE a.domain = 'item'
                  AND a.entity_id = i.id
                  AND a.lang = '{lang}'
                  AND a.status = 'approved'
            ) ia ON true
            WHERE hl.master_id = ANY($1::text[])
            ORDER BY hl.master_id, hcri.craft_id, i.{name_col}
        """,
    }

    for detail_key, sql in queries.items():
        rows = await conn.fetch(sql, station_ids)
        for row in rows:
            details[row["master_id"]][detail_key].append(dict(row))

    return details


def _base_metadata(row: dict, station_name: str) -> dict:
    return {
        "domain": "hideout",
        "entity_id": row["id"],
        "entity_name": station_name,
        "normalized_name": row.get("normalized_name"),
    }


def build_hideout_chunks(row: dict, details: dict[str, list[dict]], lang: Lang) -> list[RagChunkV3]:
    lb = LABELS[lang]
    station_id = row["id"]
    station_name, _aliases = _station_name(row, lang)
    common = {
        "domain": "hideout",
        "entity_id": station_id,
        "lang": lang,
        "source_table": "hideout_master",
        "source_updated_at": row.get("update_time"),
    }
    metadata_base = _base_metadata(row, station_name)
    header = _header(row, lang)

    levels = details.get("levels", [])
    item_requirements = details.get("item_requirements", [])
    trader_requirements = details.get("trader_requirements", [])
    station_requirements = details.get("station_requirements", [])
    skill_requirements = details.get("skill_requirements", [])
    bonuses = details.get("bonuses", [])
    crafts = details.get("crafts", [])
    craft_inputs = defaultdict(list)
    for row_input in details.get("craft_inputs", []):
        craft_inputs[row_input["craft_id"]].append(row_input)

    chunks: list[RagChunkV3] = []
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(station_id, "identifier"),
            chunk_type="identifier",
            content=header,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=station_name,
                section="identifier",
                extra=metadata_base,
            ),
        )
    )

    relation_hints = []
    for key, label in [
        ("item_requirements", lb["required_items"]),
        ("trader_requirements", lb["required_traders"]),
        ("station_requirements", lb["required_stations"]),
        ("skill_requirements", lb["required_skills"]),
        ("crafts", lb["crafts"]),
        ("bonuses", lb["bonuses"]),
    ]:
        if details.get(key):
            relation_hints.append(label)

    retrieval_content = clean_parts(
        [
            header,
            f"\n[{lb['levels']}]\n" + _format_level_overview(levels, lang) if levels else None,
            f"\n[{lb['required_items']}]\n" + _format_name_rows(item_requirements, lang, limit=25)
            if item_requirements
            else None,
            f"\n[{lb['required_stations']}]\n" + _format_name_rows(station_requirements, lang, limit=20)
            if station_requirements
            else None,
            f"\n[{lb['crafts']}]\n" + _format_crafts(crafts, craft_inputs, lang, limit=25)
            if crafts
            else None,
            f"\n[{lb['related_info']}]\n" + "\n".join(f"- {hint}" for hint in relation_hints)
            if relation_hints
            else None,
        ]
    )
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(station_id, "retrieval"),
            chunk_type="retrieval",
            content=retrieval_content,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=station_name,
                section="retrieval",
                extra=metadata_base,
            ),
        )
    )

    if crafts:
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(station_id, "summary", "crafts"),
                chunk_type="summary",
                content=clean_parts(
                    [
                        header,
                        f"{lb['crafts']}: {station_name}",
                        f"\n[{lb['crafts']}]\n" + _format_craft_outputs(crafts, lang),
                    ]
                ),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=station_name,
                    section="crafts",
                    extra={
                        **metadata_base,
                        "relation_counts": {"crafts": len(crafts)},
                    },
                ),
            )
        )

    content_sections = [
        header,
        f"\n[{lb['levels']}]\n" + _format_level_overview(levels, lang) if levels else None,
        f"\n[{lb['requirements']}]\n" + _format_requirements(details, lang, limit=80)
        if item_requirements or trader_requirements or station_requirements or skill_requirements
        else None,
        f"\n[{lb['bonuses']}]\n" + _format_bonuses(bonuses, lang) if bonuses else None,
    ]
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(station_id, "content", "main"),
            chunk_type="content",
            content=clean_parts(content_sections),
            searchable=False,
            metadata=with_common_metadata(
                **common,
                entity_name=station_name,
                section="main",
                extra=metadata_base,
            ),
        )
    )

    relation_sections = [
        f"[{lb['requirements']}]\n" + _format_requirements(details, lang, limit=120)
        if item_requirements or trader_requirements or station_requirements or skill_requirements
        else None,
        f"[{lb['crafts']}]\n" + _format_crafts(crafts, craft_inputs, lang, limit=120)
        if crafts
        else None,
        f"[{lb['bonuses']}]\n" + _format_bonuses(bonuses, lang, limit=80) if bonuses else None,
    ]
    if any(relation_sections):
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(station_id, "relation"),
                chunk_type="relation",
                content=clean_parts([f"{lb['station']}: {station_name}", *relation_sections]),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=station_name,
                    section="relation",
                    extra={
                        **metadata_base,
                        "relation_counts": {
                            "levels": len(levels),
                            "item_requirements": len(item_requirements),
                            "trader_requirements": len(trader_requirements),
                            "station_requirements": len(station_requirements),
                            "skill_requirements": len(skill_requirements),
                            "crafts": len(crafts),
                            "bonuses": len(bonuses),
                        },
                    },
                ),
            )
        )

    return chunks


async def build_and_upsert_hideouts(
    station_id: str | None = None,
    limit: int | None = None,
    langs: tuple[Lang, ...] = SUPPORTED_LANGS,
    dry_run: bool = False,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await fetch_hideout_rows(conn, station_id=station_id, limit=limit)
        station_ids = [row["id"] for row in rows]
        total = 0

        for lang in langs:
            details = await fetch_hideout_details(conn, station_ids, lang)
            chunks = []
            for row in rows:
                chunks.extend(build_hideout_chunks(row, details.get(row["id"], {}), lang))

            if dry_run:
                for chunk in chunks[:12]:
                    log.info("[dry-run] %s %s %s", chunk.lang, chunk.chunk_id, chunk.chunk_type)
                    log.info("\n%s", chunk.content[:1600])
                total += len(chunks)
            else:
                total += await upsert_rag_chunks_v3(conn, chunks)

        log.info("[hideout_builder_v3] stations=%s chunks=%s dry_run=%s", len(rows), total, dry_run)
        return total


def _parse_langs(raw: str) -> tuple[Lang, ...]:
    langs = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = [lang for lang in langs if lang not in SUPPORTED_LANGS]
    if invalid:
        raise ValueError(f"unsupported langs: {invalid}")
    return langs or SUPPORTED_LANGS


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 hideout RAG chunks.")
    parser.add_argument("--station-id", help="Build a single hideout station id.")
    parser.add_argument("--limit", type=int, help="Build a limited number of stations.")
    parser.add_argument("--langs", default="ko,en,ja", help="Comma-separated langs: ko,en,ja")
    parser.add_argument("--dry-run", action="store_true", help="Print chunks without upserting.")
    args = parser.parse_args()

    try:
        await build_and_upsert_hideouts(
            station_id=args.station_id,
            limit=args.limit,
            langs=_parse_langs(args.langs),
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
