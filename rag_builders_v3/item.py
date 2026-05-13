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
        "item": "아이템",
        "category": "카테고리",
        "parent_category": "상위 카테고리",
        "aliases": "검색명",
        "size": "크기",
        "weight": "무게",
        "weapon": "무기 정보",
        "ammo": "탄약 정보",
        "armor": "방어구 정보",
        "consumable": "소모품 정보",
        "storage": "보관함 정보",
        "throwable": "투척물 정보",
        "melee": "근접무기 정보",
        "penalty": "패널티",
        "relations": "관련 정보",
        "craft_with": "이 아이템을 재료로 제작 가능한 아이템",
        "crafted_from": "이 아이템 제작에 필요한 재료",
        "barter_with": "이 아이템을 재료로 교환 가능한 아이템",
        "bartered_from": "이 아이템 교환에 필요한 재료",
        "quest_requires": "이 아이템을 요구하는 퀘스트",
        "quest_objectives": "이 아이템과 관련된 퀘스트 목표",
        "quest_rewards": "이 아이템을 보상으로 주는 퀘스트",
        "quest_unlocks": "이 아이템 구매를 해금하는 퀘스트",
        "hideout_requires": "이 아이템이 필요한 은신처 업그레이드",
        "boss_drops": "이 아이템을 드랍하는 보스",
        "weapon_ammo": "사용 가능한 탄약",
        "ammo_weapons": "호환 무기",
    },
    "en": {
        "item": "Item",
        "category": "Category",
        "parent_category": "Parent Category",
        "aliases": "Search Names",
        "size": "Size",
        "weight": "Weight",
        "weapon": "Weapon Info",
        "ammo": "Ammo Info",
        "armor": "Protection Info",
        "consumable": "Consumable Info",
        "storage": "Storage Info",
        "throwable": "Throwable Info",
        "melee": "Melee Info",
        "penalty": "Penalties",
        "relations": "Relations",
        "craft_with": "Items craftable with this item",
        "crafted_from": "Items required to craft this item",
        "barter_with": "Items obtainable by bartering this item",
        "bartered_from": "Items required to barter for this item",
        "quest_requires": "Quests requiring this item",
        "quest_objectives": "Quest objectives involving this item",
        "quest_rewards": "Quests rewarding this item",
        "quest_unlocks": "Quests unlocking this item",
        "hideout_requires": "Hideout upgrades requiring this item",
        "boss_drops": "Bosses that can drop this item",
        "weapon_ammo": "Compatible ammo",
        "ammo_weapons": "Compatible weapons",
    },
    "ja": {
        "item": "アイテム",
        "category": "カテゴリ",
        "parent_category": "親カテゴリ",
        "aliases": "検索名",
        "size": "サイズ",
        "weight": "重量",
        "weapon": "武器情報",
        "ammo": "弾薬情報",
        "armor": "防具情報",
        "consumable": "消耗品情報",
        "storage": "収納情報",
        "throwable": "投擲物情報",
        "melee": "近接武器情報",
        "penalty": "ペナルティ",
        "relations": "関連情報",
        "craft_with": "このアイテムを材料に制作できるアイテム",
        "crafted_from": "このアイテムの制作に必要な材料",
        "barter_with": "このアイテムを材料に交換できるアイテム",
        "bartered_from": "このアイテム交換に必要な材料",
        "quest_requires": "このアイテムを要求するクエスト",
        "quest_objectives": "このアイテムに関連するクエスト目標",
        "quest_rewards": "このアイテムを報酬にするクエスト",
        "quest_unlocks": "このアイテムを解放するクエスト",
        "hideout_requires": "このアイテムが必要な隠れ家アップグレード",
        "boss_drops": "このアイテムをドロップするボス",
        "weapon_ammo": "使用可能な弾薬",
        "ammo_weapons": "対応武器",
    },
}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        normalized = value.normalize()
        text = format(normalized, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    return str(value)


def _line(label: str, value: Any, suffix: str = "") -> str | None:
    text = _as_text(value)
    if not text:
        return None
    return f"{label}: {text}{suffix}"


def _names(row: dict, lang: Lang, aliases_by_item: dict[str, list[str]]) -> tuple[str, list[str]]:
    primary = lang_value(row, "name", lang)
    # Keep aliases high-signal for retrieval without mixing every answer language.
    aliases = [row.get("name_en"), row.get("normalized_name")]
    aliases.extend(aliases_by_item.get(row["id"], []))
    alias_list = []
    for alias in aliases:
        if alias and alias not in alias_list and alias != primary:
            alias_list.append(str(alias))
    return primary, alias_list


def _dedupe_relation_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped_rows = []
    for row in rows:
        key = (
            row.get("id"),
            row.get("quantity"),
            row.get("trader_name"),
            row.get("hideout_name"),
            row.get("hideout_level"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_rows.append(row)
    return deduped_rows


def _format_relation_list(rows: list[dict], lang: Lang, name_key: str = "name") -> str:
    lines = []
    deduped_rows = _dedupe_relation_rows(rows)

    for row in deduped_rows[:20]:
        name = lang_value(row, name_key, lang) or row.get(f"{name_key}_en") or row.get("id")
        quantity = row.get("quantity") or row.get("reward_quantity")
        extra = []
        if quantity:
            extra.append(f"x{_as_text(quantity)}")
        if row.get("trader_name"):
            extra.append(str(row["trader_name"]))
        if row.get("hideout_name"):
            extra.append(str(row["hideout_name"]))
        if row.get("hideout_level"):
            extra.append(f"Lv.{row['hideout_level']}")
        if row.get("item_type"):
            extra.append(str(row["item_type"]))
        if row.get("description"):
            extra.append(str(row["description"]).strip())
        suffix = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"- {name}{suffix}")
    if len(deduped_rows) > 20:
        lines.append(f"- ... +{len(deduped_rows) - 20}")
    return "\n".join(lines)


def _item_spec_lines(row: dict, lang: Lang) -> list[str]:
    lb = LABELS[lang]
    lines = [
        _line(lb["category"], row.get("category")),
        _line(lb["parent_category"], row.get("parent_category")),
        _line(lb["weight"], row.get("weight"), "kg"),
        _line(lb["size"], f"{row.get('width')}x{row.get('height')}" if row.get("width") else None),
    ]

    if row.get("caliber") or row.get("fire_rate"):
        modes = [
            mode
            for mode, enabled in [
                ("single", row.get("is_single_fire")),
                ("full-auto", row.get("is_full_auto")),
                ("burst", row.get("is_burst_fire")),
                ("semi-auto", row.get("is_semi_auto")),
            ]
            if enabled
        ]
        lines.append(f"\n[{lb['weapon']}]")
        lines.extend(
            [
                _line("Caliber", row.get("caliber")),
                _line("Fire Rate", row.get("fire_rate"), "rpm"),
                _line("Ergonomics", row.get("ergonomics")),
                _line("Recoil", f"{row.get('recoil_vertical')}/{row.get('recoil_horizontal')}" if row.get("recoil_vertical") else None),
                _line("Modes", ", ".join(modes)),
            ]
        )

    if row.get("damage") or row.get("penetration_power"):
        lines.append(f"\n[{lb['ammo']}]")
        lines.extend(
            [
                _line("Damage", row.get("damage")),
                _line("Penetration", row.get("penetration_power")),
                _line("Armor Damage", row.get("armor_damage")),
                _line("Recoil Modifier", row.get("recoil_modifier")),
                _line("Accuracy Modifier", row.get("accuracy_modifier")),
            ]
        )

    if row.get("armor_class") or row.get("protection_type"):
        lines.append(f"\n[{lb['armor']}]")
        lines.extend(
            [
                _line("Type", row.get("protection_type")),
                _line("Armor Class", row.get("armor_class")),
                _line("Durability", row.get("durability")),
                _line("Material", row.get("material")),
                _line("Ricochet", row.get("ricochet_y")),
            ]
        )

    if row.get("consumable_type") or row.get("hitpoints"):
        lines.append(f"\n[{lb['consumable']}]")
        lines.extend(
            [
                _line("Type", row.get("consumable_type")),
                _line("Hitpoints", row.get("hitpoints")),
                _line("Units", row.get("units")),
                _line("Use Time", row.get("use_time"), "s"),
                _line("Energy", row.get("energy")),
                _line("Hydration", row.get("hydration")),
            ]
        )

    if row.get("storage_type") or row.get("capacity"):
        lines.append(f"\n[{lb['storage']}]")
        lines.extend(
            [
                _line("Type", row.get("storage_type")),
                _line("Capacity", row.get("capacity")),
            ]
        )

    if row.get("throwable_type") or row.get("fuse"):
        lines.append(f"\n[{lb['throwable']}]")
        lines.extend(
            [
                _line("Type", row.get("throwable_type")),
                _line("Fuse", row.get("fuse"), "s"),
                _line("Fragments", row.get("fragments")),
                _line("Explosion Distance", f"{row.get('min_explosion_distance')}~{row.get('max_explosion_distance')}" if row.get("min_explosion_distance") else None),
            ]
        )

    if row.get("slash_damage") or row.get("stab_damage"):
        lines.append(f"\n[{lb['melee']}]")
        lines.extend(
            [
                _line("Slash Damage", row.get("slash_damage")),
                _line("Stab Damage", row.get("stab_damage")),
                _line("Hit Radius", row.get("hit_radius")),
            ]
        )

    if row.get("ergonomics_penalty") or row.get("movement_speed_penalty"):
        lines.append(f"\n[{lb['penalty']}]")
        lines.extend(
            [
                _line("Ergonomics", row.get("ergonomics_penalty")),
                _line("Turn Speed", row.get("turn_speed_penalty")),
                _line("Movement Speed", row.get("movement_speed_penalty")),
                _line("Distance Modifier", row.get("distance_modifier")),
            ]
        )

    return [line for line in lines if line]


async def fetch_item_rows(conn, item_id: str | None = None, limit: int | None = None):
    where = "WHERE i.id = $1" if item_id else ""
    limit_clause = "LIMIT $1" if limit and not item_id else ""
    args = [item_id] if item_id else ([limit] if limit else [])
    return await conn.fetch(
        f"""
        SELECT
            i.*,
            p.ergonomics_penalty, p.turn_speed_penalty, p.movement_speed_penalty, p.distance_modifier,
            w.caliber, w.fire_rate, w.ergonomics, w.recoil_horizontal, w.recoil_vertical,
            w.default_ammo_item_id, w.is_single_fire, w.is_full_auto, w.is_burst_fire,
            w.is_semi_auto,
            a.damage, a.armor_damage, a.penetration_power, a.recoil_modifier, a.accuracy_modifier,
            m.hit_radius, m.slash_damage, m.stab_damage,
            th.throwable_type, th.fuse, th.fragments, th.min_explosion_distance, th.max_explosion_distance,
            s.storage_type, s.capacity,
            pr.protection_type, pr.armor_class, pr.durability, pr.material, pr.ricochet_y,
            c.consumable_type, c.energy, c.hydration, c.units, c.use_time, c.hitpoints,
            u.max_uses
        FROM items i
        LEFT JOIN item_penalties p ON p.item_id = i.id
        LEFT JOIN weapon_items w ON w.item_id = i.id
        LEFT JOIN ammo_items a ON a.item_id = i.id
        LEFT JOIN melee_items m ON m.item_id = i.id
        LEFT JOIN throwable_items th ON th.item_id = i.id
        LEFT JOIN storage_items s ON s.item_id = i.id
        LEFT JOIN protection_items pr ON pr.item_id = i.id
        LEFT JOIN consumable_items c ON c.item_id = i.id
        LEFT JOIN usage_items u ON u.item_id = i.id
        {where}
        ORDER BY i.update_time DESC, i.id
        {limit_clause}
        """,
        *args,
    )


async def fetch_item_relations(conn, item_ids: list[str], lang: Lang) -> dict[str, dict[str, list[dict]]]:
    relations: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    if not item_ids:
        return relations

    name_col = f"name_{lang}"

    queries = {
        "craft_with": f"""
            SELECT r.item_id AS source_item_id, hc.reward_item_id AS id, i.{name_col}, i.name_en,
                   hc.reward_quantity, hm.{name_col} AS hideout_name, hm.name_en AS hideout_name_en,
                   hl.hideout_level
            FROM hideout_craft_require_items r
            JOIN hideout_crafts hc ON hc.id = r.craft_id
            JOIN items i ON i.id = hc.reward_item_id
            LEFT JOIN hideout_levels hl ON hl.id = hc.hideout_level_id
            LEFT JOIN hideout_master hm ON hm.id = hl.master_id
            WHERE r.item_id = ANY($1::text[])
            ORDER BY i.{name_col}
        """,
        "crafted_from": f"""
            SELECT hc.reward_item_id AS source_item_id, r.item_id AS id, i.{name_col}, i.name_en,
                   r.quantity, hm.{name_col} AS hideout_name, hm.name_en AS hideout_name_en,
                   hl.hideout_level
            FROM hideout_crafts hc
            JOIN hideout_craft_require_items r ON r.craft_id = hc.id
            JOIN items i ON i.id = r.item_id
            LEFT JOIN hideout_levels hl ON hl.id = hc.hideout_level_id
            LEFT JOIN hideout_master hm ON hm.id = hl.master_id
            WHERE hc.reward_item_id = ANY($1::text[])
            ORDER BY i.{name_col}
        """,
        "barter_with": f"""
            SELECT bri.item_id AS source_item_id, bwi.item_id AS id, i.{name_col}, i.name_en,
                   bwi.quantity, t.{name_col} AS trader_name, t.name_en AS trader_name_en,
                   tb.trader_level
            FROM barter_required_items bri
            JOIN trader_barters tb ON tb.id = bri.barter_id
            JOIN barter_reward_items bwi ON bwi.barter_id = tb.id
            JOIN items i ON i.id = bwi.item_id
            LEFT JOIN traders t ON t.id = tb.trader_id
            WHERE bri.item_id = ANY($1::text[])
            ORDER BY i.{name_col}
        """,
        "bartered_from": f"""
            SELECT bwi.item_id AS source_item_id, bri.item_id AS id, i.{name_col}, i.name_en,
                   bri.quantity, t.{name_col} AS trader_name, t.name_en AS trader_name_en,
                   tb.trader_level
            FROM barter_reward_items bwi
            JOIN trader_barters tb ON tb.id = bwi.barter_id
            JOIN barter_required_items bri ON bri.barter_id = tb.id
            JOIN items i ON i.id = bri.item_id
            LEFT JOIN traders t ON t.id = tb.trader_id
            WHERE bwi.item_id = ANY($1::text[])
            ORDER BY i.{name_col}
        """,
        "quest_objectives": f"""
            SELECT qoi.item_id AS source_item_id, q.id, q.{name_col}, q.name_en,
                   qo.count AS quantity, qoi.item_type,
                   qo.{f"description_{lang}"} AS description
            FROM quest_objective_items qoi
            JOIN quest_objectives qo ON qo.objective_id = qoi.objective_id
            JOIN quests q ON q.id = qo.quest_id
            WHERE qoi.item_id = ANY($1::text[])
            ORDER BY q.sort_order, q.{name_col}
        """,
        "quest_rewards": f"""
            SELECT qri.item_id AS source_item_id, q.id, q.{name_col}, q.name_en,
                   qri.quantity
            FROM quest_finish_reward_items qri
            JOIN quests q ON q.id = qri.quest_id
            WHERE qri.item_id = ANY($1::text[])
            ORDER BY q.sort_order, q.{name_col}
        """,
        "quest_unlocks": f"""
            SELECT qou.item_id AS source_item_id, q.id, q.{name_col}, q.name_en,
                   qou.level AS quantity
            FROM quest_finish_reward_offer_unlock qou
            JOIN quests q ON q.id = qou.quest_id
            WHERE qou.item_id = ANY($1::text[])
            ORDER BY q.sort_order, q.{name_col}
        """,
        "hideout_requires": f"""
            SELECT hir.item_id AS source_item_id, hl.id, hm.{name_col}, hm.name_en,
                   hir.quantity, hm.{name_col} AS hideout_name, hl.hideout_level
            FROM hideout_item_require hir
            JOIN hideout_levels hl ON hl.id = hir.hideout_level_id
            JOIN hideout_master hm ON hm.id = hl.master_id
            WHERE hir.item_id = ANY($1::text[])
            ORDER BY hm.{name_col}, hl.hideout_level
        """,
        "boss_drops": f"""
            SELECT bi.item_id AS source_item_id, b.id, b.{name_col}, b.name_en,
                   bi.quantity
            FROM boss_item bi
            JOIN bosses b ON b.id = bi.boss_id
            WHERE bi.item_id = ANY($1::text[])
            ORDER BY b.sort_order, b.{name_col}
        """,
        "weapon_ammo": f"""
            SELECT waa.item_id AS source_item_id, i.id, i.{name_col}, i.name_en
            FROM weapon_allowed_ammo waa
            JOIN items i ON i.id = waa.ammo_item_id
            WHERE waa.item_id = ANY($1::text[])
            ORDER BY i.{name_col}
        """,
        "ammo_weapons": f"""
            SELECT waa.ammo_item_id AS source_item_id, i.id, i.{name_col}, i.name_en
            FROM weapon_allowed_ammo waa
            JOIN items i ON i.id = waa.item_id
            WHERE waa.ammo_item_id = ANY($1::text[])
            ORDER BY i.{name_col}
        """,
    }

    for relation_type, sql in queries.items():
        rows = await conn.fetch(sql, item_ids)
        for row in rows:
            relations[row["source_item_id"]][relation_type].append(dict(row))

    return relations


async def fetch_item_aliases(
    conn,
    item_ids: list[str],
    lang: Lang,
) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = defaultdict(list)
    if not item_ids:
        return aliases

    rows = await conn.fetch(
        """
        SELECT entity_id, alias
        FROM rag_entity_aliases_v3
        WHERE domain = 'item'
          AND entity_id = ANY($1::text[])
          AND lang = $2
          AND status = 'approved'
        ORDER BY alias
        """,
        item_ids,
        lang,
    )
    for row in rows:
        alias = str(row["alias"]).strip()
        if alias:
            aliases[row["entity_id"]].append(alias)
    return aliases


def build_item_chunks(
    row: dict,
    relations: dict[str, list[dict]],
    lang: Lang,
    aliases_by_item: dict[str, list[str]] | None = None,
) -> list[RagChunkV3]:
    lb = LABELS[lang]
    entity_id = row["id"]
    entity_name, aliases = _names(row, lang, aliases_by_item or {})
    metadata_base = {
        "domain": "item",
        "entity_id": entity_id,
        "entity_name": entity_name,
        "category": row.get("category"),
        "parent_category": row.get("parent_category"),
    }

    chunks: list[RagChunkV3] = []
    common = {
        "domain": "item",
        "entity_id": entity_id,
        "lang": lang,
        "source_table": "items",
        "source_updated_at": row.get("update_time"),
    }

    identifier_content = clean_parts(
        [
            f"{lb['item']}: {entity_name}",
            f"{lb['aliases']}: {', '.join(aliases)}" if aliases else None,
            _line(lb["category"], row.get("category")),
            _line(lb["parent_category"], row.get("parent_category")),
        ]
    )
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(entity_id, "identifier"),
            chunk_type="identifier",
            content=identifier_content,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=entity_name,
                section="identifier",
                image=row.get("image"),
                extra=metadata_base,
            ),
        )
    )

    relation_hints = []
    for relation_type in relations:
        if relations[relation_type]:
            relation_hints.append(LABELS[lang][relation_type])

    retrieval_content = clean_parts(
        [
            identifier_content,
            "\n".join(
                line
                for line in _item_spec_lines(row, lang)[:12]
                if not line.startswith(f"{lb['category']}:")
                and not line.startswith(f"{lb['parent_category']}:")
            ),
            f"\n[{lb['relations']}]\n" + "\n".join(f"- {hint}" for hint in relation_hints)
            if relation_hints
            else None,
        ]
    )
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(entity_id, "retrieval"),
            chunk_type="retrieval",
            content=retrieval_content,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=entity_name,
                section="retrieval",
                image=row.get("image"),
                extra=metadata_base,
            ),
        )
    )

    spec_content = clean_parts(
        [
            f"{lb['item']}: {entity_name}",
            "\n".join(_item_spec_lines(row, lang)),
        ]
    )
    if spec_content.strip() != identifier_content.strip():
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(entity_id, "content", "spec"),
                chunk_type="content",
                content=spec_content,
                searchable=False,
                metadata=with_common_metadata(
                    **common,
                    entity_name=entity_name,
                    section="spec",
                    image=row.get("image"),
                    extra=metadata_base,
                ),
            )
        )

    relation_sections = []
    relation_meta_counts = {}
    for relation_type, rows in relations.items():
        if not rows:
            continue
        deduped_rows = _dedupe_relation_rows(rows)
        relation_sections.append(
            f"[{LABELS[lang][relation_type]}]\n{_format_relation_list(deduped_rows, lang)}"
        )
        relation_meta_counts[relation_type] = len(deduped_rows)

    if relation_sections:
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(entity_id, "relation"),
                chunk_type="relation",
                content=clean_parts([f"{lb['item']}: {entity_name}", *relation_sections]),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=entity_name,
                    section="relation",
                    image=row.get("image"),
                    extra={**metadata_base, "relation_counts": relation_meta_counts},
                ),
            )
        )

    return chunks


async def build_and_upsert_items(
    item_id: str | None = None,
    limit: int | None = None,
    langs: tuple[Lang, ...] = SUPPORTED_LANGS,
    dry_run: bool = False,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = [dict(row) for row in await fetch_item_rows(conn, item_id=item_id, limit=limit)]
        item_ids = [row["id"] for row in rows]
        total = 0

        for lang in langs:
            relations = await fetch_item_relations(conn, item_ids, lang)
            aliases = await fetch_item_aliases(conn, item_ids, lang)
            chunks = []
            for row in rows:
                chunks.extend(
                    build_item_chunks(
                        row,
                        relations.get(row["id"], {}),
                        lang,
                        aliases_by_item=aliases,
                    )
                )

            if dry_run:
                for chunk in chunks[:10]:
                    log.info("[dry-run] %s %s %s", chunk.lang, chunk.chunk_id, chunk.chunk_type)
                    log.info("\n%s", chunk.content[:800])
                total += len(chunks)
            else:
                total += await upsert_rag_chunks_v3(conn, chunks)

        log.info("[item_builder_v3] items=%s chunks=%s dry_run=%s", len(rows), total, dry_run)
        return total


def _parse_langs(raw: str) -> tuple[Lang, ...]:
    langs = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = [lang for lang in langs if lang not in SUPPORTED_LANGS]
    if invalid:
        raise ValueError(f"unsupported langs: {invalid}")
    return langs or SUPPORTED_LANGS


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 item RAG chunks.")
    parser.add_argument("--item-id", help="Build a single item id.")
    parser.add_argument("--limit", type=int, help="Build a limited number of latest items.")
    parser.add_argument("--langs", default="ko,en,ja", help="Comma-separated langs: ko,en,ja")
    parser.add_argument("--dry-run", action="store_true", help="Print chunks without upserting.")
    args = parser.parse_args()

    try:
        await build_and_upsert_items(
            item_id=args.item_id,
            limit=args.limit,
            langs=_parse_langs(args.langs),
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
