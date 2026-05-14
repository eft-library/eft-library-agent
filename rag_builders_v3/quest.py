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
        "quest": "퀘스트",
        "trader": "상인",
        "aliases": "검색명",
        "min_level": "최소 레벨",
        "experience": "경험치",
        "kappa": "카파 필요",
        "objectives": "목표",
        "maps": "관련 맵",
        "required_items": "필요 아이템",
        "objective_items": "필요 아이템",
        "required_keys": "필요 키",
        "objective_keys": "필요 키",
        "previous_quests": "선행 퀘스트",
        "next_quests": "후행 퀘스트",
        "rewards": "보상",
        "reward_items": "보상 아이템",
        "skill_rewards": "스킬 보상",
        "standing_rewards": "상인 평판 보상",
        "offer_unlocks": "구매 해금",
        "craft_unlocks": "제작 해금",
        "guide": "가이드",
        "related_info": "관련 정보",
        "yes": "예",
        "no": "아니오",
    },
    "en": {
        "quest": "Quest",
        "trader": "Trader",
        "aliases": "Search Names",
        "min_level": "Min Level",
        "experience": "Experience",
        "kappa": "Kappa Required",
        "objectives": "Objectives",
        "maps": "Related Maps",
        "required_items": "Required Items",
        "objective_items": "Required Items",
        "required_keys": "Required Keys",
        "objective_keys": "Required Keys",
        "previous_quests": "Previous Quests",
        "next_quests": "Next Quests",
        "rewards": "Rewards",
        "reward_items": "Reward Items",
        "skill_rewards": "Skill Rewards",
        "standing_rewards": "Trader Standing Rewards",
        "offer_unlocks": "Offer Unlocks",
        "craft_unlocks": "Craft Unlocks",
        "guide": "Guide",
        "related_info": "Related Info",
        "yes": "Yes",
        "no": "No",
    },
    "ja": {
        "quest": "クエスト",
        "trader": "トレーダー",
        "aliases": "検索名",
        "min_level": "最低レベル",
        "experience": "経験値",
        "kappa": "カッパ必要",
        "objectives": "目標",
        "maps": "関連マップ",
        "required_items": "必要アイテム",
        "objective_items": "必要アイテム",
        "required_keys": "必要キー",
        "objective_keys": "必要キー",
        "previous_quests": "前提クエスト",
        "next_quests": "後続クエスト",
        "rewards": "報酬",
        "reward_items": "報酬アイテム",
        "skill_rewards": "スキル報酬",
        "standing_rewards": "トレーダー評価報酬",
        "offer_unlocks": "購入アンロック",
        "craft_unlocks": "クラフトアンロック",
        "guide": "ガイド",
        "related_info": "関連情報",
        "yes": "はい",
        "no": "いいえ",
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


def _line(label: str, value: Any, suffix: str = "") -> str | None:
    text = _as_text(value)
    if not text:
        return None
    return f"{label}: {text}{suffix}"


def _clean_html(html_text: str | None) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for img in soup.find_all("img"):
        img.decompose()
    return soup.get_text(separator="\n", strip=True)


def _quest_name(row: dict, lang: Lang) -> tuple[str, list[str]]:
    primary = lang_value(row, "name", lang)
    aliases = [row.get("name_en"), row.get("normalized_name")]
    alias_list = []
    for alias in aliases:
        alias_text = str(alias or "").strip()
        if alias_text and alias_text != primary and alias_text not in alias_list:
            alias_list.append(alias_text)
    return primary, alias_list


def _format_rows(
    rows: list[dict],
    lang: Lang,
    *,
    name_prefix: str = "name",
    include_quantity: bool = True,
    include_description: bool = False,
    limit: int = 30,
) -> str:
    lines = []
    seen = set()
    for row in rows:
        name = (
            str(row.get("description") or "").strip()
            if name_prefix == "description"
            else ""
        ) or (
            lang_value(row, name_prefix, lang)
            or row.get(f"{name_prefix}_en")
            or row.get("id")
            or ""
        )
        extras = []
        if include_quantity and row.get("quantity"):
            extras.append(f"x{_as_text(row['quantity'])}")
        if row.get("item_type"):
            extras.append(str(row["item_type"]))
        if row.get("map_name"):
            extras.append(str(row["map_name"]))
        if row.get("trader_name"):
            trader_name = str(row["trader_name"]).strip()
            if trader_name and trader_name != name:
                extras.append(trader_name)
        if row.get("standing"):
            extras.append(f"+{_as_text(row['standing'])}")
        if row.get("level"):
            extras.append(f"LL{row['level']}")
        if row.get("station_level"):
            extras.append(f"Lv.{row['station_level']}")
        if include_description and row.get("description"):
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


async def fetch_quest_rows(conn, quest_id: str | None = None, limit: int | None = None):
    where = "WHERE q.id = $1" if quest_id else ""
    limit_clause = "LIMIT $1" if limit and not quest_id else ""
    args = [quest_id] if quest_id else ([limit] if limit else [])
    rows = await conn.fetch(
        f"""
        SELECT
            q.*,
            t.name_en AS trader_name_en,
            t.name_ko AS trader_name_ko,
            t.name_ja AS trader_name_ja,
            t.normalized_name AS trader_normalized_name,
            t.image AS trader_image
        FROM quests q
        LEFT JOIN traders t ON t.id = q.trader_id
        {where}
        ORDER BY q.update_time DESC, q.sort_order, q.id
        {limit_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


async def fetch_quest_details(conn, quest_ids: list[str], lang: Lang) -> dict[str, dict[str, list[dict]]]:
    details: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    if not quest_ids:
        return details

    name_col = f"name_{lang}"
    desc_col = f"description_{lang}"

    queries = {
        "objectives": f"""
            SELECT qo.quest_id, qo.objective_id AS id, qo.type, qo.{desc_col} AS description,
                   qo.count AS quantity, qo.found_in_raid
            FROM quest_objectives qo
            WHERE qo.quest_id = ANY($1::text[])
            ORDER BY qo.quest_id, qo.sort_order, qo.objective_id
        """,
        "objective_items": f"""
            SELECT qo.quest_id, i.id, i.{name_col}, i.name_en,
                   qoi.item_type, qo.count AS quantity, qo.{desc_col} AS description
            FROM quest_objective_items qoi
            JOIN quest_objectives qo ON qo.objective_id = qoi.objective_id
            JOIN items i ON i.id = qoi.item_id
            WHERE qo.quest_id = ANY($1::text[])
            ORDER BY qo.quest_id, qo.sort_order, qoi.sort_order, i.{name_col}
        """,
        "objective_keys": f"""
            SELECT qo.quest_id, i.id, i.{name_col}, i.name_en,
                   qo.{desc_col} AS description
            FROM quest_objective_required_keys qk
            JOIN quest_objectives qo ON qo.objective_id = qk.objective_id
            JOIN items i ON i.id = qk.key_id
            WHERE qo.quest_id = ANY($1::text[])
            ORDER BY qo.quest_id, qo.sort_order, i.{name_col}
        """,
        "objective_maps": f"""
            SELECT qo.quest_id, m.id, m.{name_col}, m.name_en,
                   m.{name_col} AS map_name, qo.{desc_col} AS description
            FROM quest_objective_maps qm
            JOIN quest_objectives qo ON qo.objective_id = qm.objective_id
            JOIN maps m ON m.id = qm.map_id
            WHERE qo.quest_id = ANY($1::text[])
            ORDER BY qo.quest_id, qo.sort_order, qm.sort_order, m.{name_col}
        """,
        "previous_quests": f"""
            SELECT qr.quest_id, q.id, q.{name_col}, q.name_en
            FROM quest_relations qr
            JOIN quests q ON q.id = qr.related_quest_id
            WHERE qr.quest_id = ANY($1::text[])
              AND qr.relation_type = 'require'
            ORDER BY qr.quest_id, qr.sort_order, q.{name_col}
        """,
        "next_quests": f"""
            SELECT qr.quest_id, q.id, q.{name_col}, q.name_en
            FROM quest_relations qr
            JOIN quests q ON q.id = qr.related_quest_id
            WHERE qr.quest_id = ANY($1::text[])
              AND qr.relation_type = 'next'
            ORDER BY qr.quest_id, qr.sort_order, q.{name_col}
        """,
        "reward_items": f"""
            SELECT r.quest_id, i.id, i.{name_col}, i.name_en, r.quantity
            FROM quest_finish_reward_items r
            JOIN items i ON i.id = r.item_id
            WHERE r.quest_id = ANY($1::text[])
            ORDER BY r.quest_id, r.sort_order, i.{name_col}
        """,
        "skill_rewards": f"""
            SELECT r.quest_id, r.name_en, r.name_ko, r.name_ja,
                   r.skill_level AS quantity
            FROM quest_finish_reward_skills r
            WHERE r.quest_id = ANY($1::text[])
            ORDER BY r.quest_id, r.sort_order, r.{name_col}
        """,
        "standing_rewards": f"""
            SELECT r.quest_id, t.id, t.{name_col}, t.name_en,
                   t.{name_col} AS trader_name, r.standing
            FROM quest_finish_reward_trader_standing r
            JOIN traders t ON t.id = r.trader_id
            WHERE r.quest_id = ANY($1::text[])
            ORDER BY r.quest_id, r.sort_order, t.{name_col}
        """,
        "offer_unlocks": f"""
            SELECT r.quest_id, i.id, i.{name_col}, i.name_en,
                   r.level, t.{name_col} AS trader_name, t.name_en AS trader_name_en
            FROM quest_finish_reward_offer_unlock r
            LEFT JOIN items i ON i.id = r.item_id
            LEFT JOIN traders t ON t.id = r.trader_id
            WHERE r.quest_id = ANY($1::text[])
            ORDER BY r.quest_id, r.sort_order, i.{name_col}
        """,
        "craft_unlocks": f"""
            SELECT r.quest_id, hc.id, i.{name_col}, i.name_en,
                   r.station_level
            FROM quest_finish_reward_craft_unlocks r
            JOIN hideout_crafts hc ON hc.id = r.craft_id
            LEFT JOIN items i ON i.id = hc.reward_item_id
            WHERE r.quest_id = ANY($1::text[])
            ORDER BY r.quest_id, r.sort_order, i.{name_col}
        """,
    }

    for detail_key, sql in queries.items():
        rows = await conn.fetch(sql, quest_ids)
        for row in rows:
            details[row["quest_id"]][detail_key].append(dict(row))

    return details


def _base_metadata(row: dict, quest_name: str, lang: Lang) -> dict:
    return {
        "domain": "quest",
        "entity_id": row["id"],
        "entity_name": quest_name,
        "trader_id": row.get("trader_id"),
        "trader_name": lang_value(row, "trader_name", lang),
        "min_player_level": row.get("min_player_level"),
        "kappa_required": bool(row.get("kappa_required")),
        "wiki_url": row.get("wiki_url"),
    }


def _build_header(row: dict, lang: Lang) -> str:
    lb = LABELS[lang]
    quest_name, aliases = _quest_name(row, lang)
    trader_name = lang_value(row, "trader_name", lang)
    return clean_parts(
        [
            f"{lb['quest']}: {quest_name}",
            f"{lb['aliases']}: {', '.join(aliases)}" if aliases else None,
            _line(lb["trader"], trader_name),
            _line(lb["min_level"], row.get("min_player_level")),
            _line(lb["experience"], row.get("experience")),
            _line(lb["kappa"], lb["yes"] if row.get("kappa_required") else lb["no"]),
        ]
    )


def build_quest_chunks(row: dict, details: dict[str, list[dict]], lang: Lang) -> list[RagChunkV3]:
    lb = LABELS[lang]
    quest_id = row["id"]
    quest_name, _aliases = _quest_name(row, lang)
    common = {
        "domain": "quest",
        "entity_id": quest_id,
        "lang": lang,
        "source_table": "quests",
        "source_updated_at": row.get("update_time"),
    }
    metadata_base = _base_metadata(row, quest_name, lang)

    chunks: list[RagChunkV3] = []
    header = _build_header(row, lang)

    objectives = details.get("objectives", [])
    objective_items = details.get("objective_items", [])
    objective_keys = details.get("objective_keys", [])
    objective_maps = details.get("objective_maps", [])
    previous_quests = details.get("previous_quests", [])
    next_quests = details.get("next_quests", [])
    reward_items = details.get("reward_items", [])
    skill_rewards = details.get("skill_rewards", [])
    standing_rewards = details.get("standing_rewards", [])
    offer_unlocks = details.get("offer_unlocks", [])
    craft_unlocks = details.get("craft_unlocks", [])

    identifier_content = clean_parts(
        [
            header,
            f"\n[{lb['objectives']}]\n"
            + _format_rows(
                objectives,
                lang,
                name_prefix="description",
                include_quantity=True,
                include_description=False,
                limit=8,
            )
            if objectives
            else None,
        ]
    )
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(quest_id, "identifier"),
            chunk_type="identifier",
            content=identifier_content,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=quest_name,
                section="identifier",
                url=row.get("wiki_url"),
                image=row.get("trader_image"),
                extra=metadata_base,
            ),
        )
    )

    relation_hints = []
    for key in [
        "objective_items",
        "objective_keys",
        "objective_maps",
        "previous_quests",
        "next_quests",
        "reward_items",
        "skill_rewards",
        "standing_rewards",
        "offer_unlocks",
        "craft_unlocks",
    ]:
        if details.get(key):
            relation_hints.append(lb.get(key, key))

    retrieval_content = clean_parts(
        [
            header,
            f"\n[{lb['objectives']}]\n"
            + _format_rows(
                objectives,
                lang,
                name_prefix="description",
                include_quantity=True,
                limit=12,
            )
            if objectives
            else None,
            f"\n[{lb['maps']}]\n" + _format_rows(objective_maps, lang, limit=12)
            if objective_maps
            else None,
            f"\n[{lb['required_items']}]\n"
            + _format_rows(objective_items, lang, include_description=True, limit=12)
            if objective_items
            else None,
            f"\n[{lb['rewards']}]\n"
            + _format_rows(reward_items, lang, include_quantity=True, limit=12)
            if reward_items
            else None,
            f"\n[{lb['related_info']}]\n"
            + "\n".join(f"- {hint}" for hint in relation_hints)
            if relation_hints
            else None,
        ]
    )
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(quest_id, "retrieval"),
            chunk_type="retrieval",
            content=retrieval_content,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=quest_name,
                section="retrieval",
                url=row.get("wiki_url"),
                image=row.get("trader_image"),
                extra=metadata_base,
            ),
        )
    )

    content_sections = [
        header,
        f"\n[{lb['objectives']}]\n"
        + _format_rows(
            objectives,
            lang,
            name_prefix="description",
            include_quantity=True,
            limit=50,
        )
        if objectives
        else None,
        f"\n[{lb['previous_quests']}]\n" + _format_rows(previous_quests, lang)
        if previous_quests
        else None,
        f"\n[{lb['next_quests']}]\n" + _format_rows(next_quests, lang)
        if next_quests
        else None,
        f"\n[{lb['reward_items']}]\n" + _format_rows(reward_items, lang)
        if reward_items
        else None,
        f"\n[{lb['skill_rewards']}]\n" + _format_rows(skill_rewards, lang)
        if skill_rewards
        else None,
        f"\n[{lb['standing_rewards']}]\n" + _format_rows(standing_rewards, lang)
        if standing_rewards
        else None,
    ]
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(quest_id, "content", "main"),
            chunk_type="content",
            content=clean_parts(content_sections),
            searchable=False,
            metadata=with_common_metadata(
                **common,
                entity_name=quest_name,
                section="main",
                url=row.get("wiki_url"),
                image=row.get("trader_image"),
                extra=metadata_base,
            ),
        )
    )

    relation_sections = [
        f"[{lb['maps']}]\n" + _format_rows(objective_maps, lang, include_description=True)
        if objective_maps
        else None,
        f"[{lb['required_items']}]\n"
        + _format_rows(objective_items, lang, include_description=True)
        if objective_items
        else None,
        f"[{lb['required_keys']}]\n"
        + _format_rows(objective_keys, lang, include_description=True)
        if objective_keys
        else None,
        f"[{lb['previous_quests']}]\n" + _format_rows(previous_quests, lang)
        if previous_quests
        else None,
        f"[{lb['next_quests']}]\n" + _format_rows(next_quests, lang)
        if next_quests
        else None,
        f"[{lb['reward_items']}]\n" + _format_rows(reward_items, lang)
        if reward_items
        else None,
        f"[{lb['offer_unlocks']}]\n" + _format_rows(offer_unlocks, lang)
        if offer_unlocks
        else None,
        f"[{lb['craft_unlocks']}]\n" + _format_rows(craft_unlocks, lang)
        if craft_unlocks
        else None,
    ]
    if any(relation_sections):
        relation_counts = {
            "objective_items": len(objective_items),
            "objective_keys": len(objective_keys),
            "objective_maps": len(objective_maps),
            "previous_quests": len(previous_quests),
            "next_quests": len(next_quests),
            "reward_items": len(reward_items),
            "offer_unlocks": len(offer_unlocks),
            "craft_unlocks": len(craft_unlocks),
        }
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(quest_id, "relation"),
                chunk_type="relation",
                content=clean_parts([f"{lb['quest']}: {quest_name}", *relation_sections]),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=quest_name,
                    section="relation",
                    url=row.get("wiki_url"),
                    image=row.get("trader_image"),
                    extra={**metadata_base, "relation_counts": relation_counts},
                ),
            )
        )

    guide = _clean_html(row.get(f"guide_{lang}"))
    if guide and guide.strip("- \n\t"):
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(quest_id, "guide"),
                chunk_type="guide",
                content=clean_parts([f"{lb['quest']}: {quest_name}", f"[{lb['guide']}]\n{guide}"]),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=quest_name,
                    section="guide",
                    url=row.get("wiki_url"),
                    image=row.get("trader_image"),
                    extra=metadata_base,
                ),
            )
        )

    return chunks


async def build_and_upsert_quests(
    quest_id: str | None = None,
    limit: int | None = None,
    langs: tuple[Lang, ...] = SUPPORTED_LANGS,
    dry_run: bool = False,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await fetch_quest_rows(conn, quest_id=quest_id, limit=limit)
        quest_ids = [row["id"] for row in rows]
        total = 0

        for lang in langs:
            details = await fetch_quest_details(conn, quest_ids, lang)
            chunks = []
            for row in rows:
                chunks.extend(build_quest_chunks(row, details.get(row["id"], {}), lang))

            if dry_run:
                for chunk in chunks[:12]:
                    log.info("[dry-run] %s %s %s", chunk.lang, chunk.chunk_id, chunk.chunk_type)
                    log.info("\n%s", chunk.content[:1000])
                total += len(chunks)
            else:
                total += await upsert_rag_chunks_v3(conn, chunks)

        log.info("[quest_builder_v3] quests=%s chunks=%s dry_run=%s", len(rows), total, dry_run)
        return total


def _parse_langs(raw: str) -> tuple[Lang, ...]:
    langs = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = [lang for lang in langs if lang not in SUPPORTED_LANGS]
    if invalid:
        raise ValueError(f"unsupported langs: {invalid}")
    return langs or SUPPORTED_LANGS


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 quest RAG chunks.")
    parser.add_argument("--quest-id", help="Build a single quest id.")
    parser.add_argument("--limit", type=int, help="Build a limited number of latest quests.")
    parser.add_argument("--langs", default="ko,en,ja", help="Comma-separated langs: ko,en,ja")
    parser.add_argument("--dry-run", action="store_true", help="Print chunks without upserting.")
    args = parser.parse_args()

    try:
        await build_and_upsert_quests(
            quest_id=args.quest_id,
            limit=args.limit,
            langs=_parse_langs(args.langs),
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
