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
        "trader": "상인",
        "aliases": "검색명",
        "barters": "바터",
        "barter_rewards": "바터 보상",
        "barter_inputs": "바터 재료",
        "levels": "상인 레벨",
        "prices": "가격 데이터",
        "price_items": "가격 등록 아이템",
        "related_info": "관련 정보",
    },
    "en": {
        "trader": "Trader",
        "aliases": "Search Names",
        "barters": "Barters",
        "barter_rewards": "Barter Rewards",
        "barter_inputs": "Barter Inputs",
        "levels": "Trader Levels",
        "prices": "Price Data",
        "price_items": "Priced Items",
        "related_info": "Related Info",
    },
    "ja": {
        "trader": "トレーダー",
        "aliases": "検索名",
        "barters": "物々交換",
        "barter_rewards": "交換報酬",
        "barter_inputs": "交換材料",
        "levels": "トレーダーレベル",
        "prices": "価格データ",
        "price_items": "価格登録アイテム",
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


def _trader_name(row: dict, lang: Lang) -> tuple[str, list[str]]:
    primary = lang_value(row, "name", lang)
    aliases = [row.get("name_en"), row.get("normalized_name")]
    alias_list = []
    for alias in aliases:
        alias_text = str(alias or "").strip()
        if alias_text and alias_text != primary and alias_text not in alias_list:
            alias_list.append(alias_text)
    return primary, alias_list


def _item_name(row: dict, lang: Lang, *, prefix: str = "name") -> str:
    name = lang_value(row, prefix, lang) or row.get(f"{prefix}_en") or row.get("id") or ""
    aliases = str(row.get("aliases") or "").strip()
    if aliases:
        return f"{name} ({aliases})"
    return name


def _header(row: dict, lang: Lang) -> str:
    lb = LABELS[lang]
    trader_name, aliases = _trader_name(row, lang)
    return clean_parts(
        [
            f"{lb['trader']}: {trader_name}",
            f"{lb['aliases']}: {', '.join(aliases)}" if aliases else None,
        ]
    )


def _format_level_counts(barters: list[dict], lang: Lang) -> str:
    lb = LABELS[lang]
    counts: dict[int, int] = defaultdict(int)
    for barter in barters:
        if barter.get("trader_level") is not None:
            counts[int(barter["trader_level"])] += 1
    return "\n".join(f"- LL{level}: {count} {lb['barters']}" for level, count in sorted(counts.items()))


def _format_barter_rewards(barters: list[dict], lang: Lang, *, limit: int = 120) -> str:
    lines = []
    seen = set()
    for barter in barters:
        reward = _item_name(barter, lang, prefix="reward_name")
        extras = []
        if barter.get("reward_quantity"):
            extras.append(f"x{_as_text(barter['reward_quantity'])}")
        if barter.get("trader_level"):
            extras.append(f"LL{barter['trader_level']}")
        key = (reward, tuple(extras))
        if key in seen:
            continue
        seen.add(key)
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {reward}{suffix}")
        if len(lines) >= limit:
            break
    if len(barters) > limit:
        lines.append(f"- ... +{len(barters) - limit}")
    return "\n".join(lines)


def _format_barters(
    barters: list[dict],
    inputs_by_barter: dict[str, list[dict]],
    lang: Lang,
    *,
    limit: int = 120,
) -> str:
    lines = []
    seen = set()
    for barter in barters:
        reward = _item_name(barter, lang, prefix="reward_name")
        extras = []
        if barter.get("reward_quantity"):
            extras.append(f"x{_as_text(barter['reward_quantity'])}")
        if barter.get("trader_level"):
            extras.append(f"LL{barter['trader_level']}")
        inputs = inputs_by_barter.get(barter["id"], [])
        input_text = ", ".join(
            f"{_item_name(row, lang)} x{_as_text(row['quantity'])}" for row in inputs[:10]
        )
        if len(inputs) > 10:
            input_text += f", ... +{len(inputs) - 10}"
        key = (reward, tuple(extras), input_text)
        if key in seen:
            continue
        seen.add(key)
        suffix = f" ({', '.join(extras)})" if extras else ""
        line = f"- {reward}{suffix}"
        if input_text:
            line += f"\n  - {LABELS[lang]['barter_inputs']}: {input_text}"
        lines.append(line)
        if len(lines) >= limit:
            break
    if len(barters) > limit:
        lines.append(f"- ... +{len(barters) - limit}")
    return "\n".join(lines)


def _format_price_summary(rows: list[dict], lang: Lang, *, limit: int = 40) -> str:
    lines = []
    for row in rows[:limit]:
        name = _item_name(row, lang)
        extras = []
        if row.get("game_mode"):
            extras.append(str(row["game_mode"]))
        if row.get("price") is not None:
            extras.append(_as_text(row["price"]))
        suffix = f" ({', '.join(extras)})" if extras else ""
        lines.append(f"- {name}{suffix}")
    if len(rows) > limit:
        lines.append(f"- ... +{len(rows) - limit}")
    return "\n".join(lines)


async def fetch_trader_rows(conn, trader_id: str | None = None, limit: int | None = None):
    args: list[Any] = []
    where_parts = ["is_use IS TRUE"]
    if trader_id:
        args.append(trader_id)
        where_parts.append(f"id = ${len(args)}")
    limit_clause = ""
    if limit is not None and not trader_id:
        args.append(limit)
        limit_clause = f"LIMIT ${len(args)}"
    rows = await conn.fetch(
        f"""
        SELECT *
        FROM traders
        WHERE {' AND '.join(where_parts)}
        ORDER BY sort_order NULLS LAST, id
        {limit_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


async def fetch_trader_details(conn, trader_ids: list[str], lang: Lang) -> dict[str, dict[str, list[dict]]]:
    details: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    if not trader_ids:
        return details

    name_col = f"name_{lang}"
    queries = {
        "barters": f"""
            SELECT tb.trader_id, tb.id, tb.trader_level,
                   ri.id AS reward_item_id,
                   ri.{name_col} AS reward_name_{lang}, ri.name_en AS reward_name_en,
                   ria.aliases, bwi.quantity AS reward_quantity
            FROM trader_barters tb
            JOIN barter_reward_items bwi ON bwi.barter_id = tb.id
            JOIN items ri ON ri.id = bwi.item_id
            LEFT JOIN LATERAL (
                SELECT string_agg(a.alias, ', ' ORDER BY a.alias) AS aliases
                FROM rag_entity_aliases_v3 a
                WHERE a.domain = 'item'
                  AND a.entity_id = ri.id
                  AND a.lang = '{lang}'
                  AND a.status = 'approved'
            ) ria ON true
            WHERE tb.trader_id = ANY($1::text[])
            ORDER BY tb.trader_id, tb.trader_level, ri.{name_col}
        """,
        "barter_inputs": f"""
            SELECT tb.trader_id, bri.barter_id, ii.id,
                   ii.{name_col}, ii.name_en, ia.aliases, bri.quantity
            FROM trader_barters tb
            JOIN barter_required_items bri ON bri.barter_id = tb.id
            JOIN items ii ON ii.id = bri.item_id
            LEFT JOIN LATERAL (
                SELECT string_agg(a.alias, ', ' ORDER BY a.alias) AS aliases
                FROM rag_entity_aliases_v3 a
                WHERE a.domain = 'item'
                  AND a.entity_id = ii.id
                  AND a.lang = '{lang}'
                  AND a.status = 'approved'
            ) ia ON true
            WHERE tb.trader_id = ANY($1::text[])
            ORDER BY tb.trader_id, bri.barter_id, ii.{name_col}
        """,
        "prices": f"""
            WITH ranked AS (
                SELECT itp.trader_id, itp.game_mode, itp.price,
                       i.id, i.{name_col}, i.name_en, ia.aliases,
                       row_number() OVER (
                           PARTITION BY itp.trader_id, itp.game_mode
                           ORDER BY itp.price DESC NULLS LAST, i.{name_col}
                       ) AS rn
                FROM item_trader_prices itp
                JOIN items i ON i.id = itp.item_id
                LEFT JOIN LATERAL (
                    SELECT string_agg(a.alias, ', ' ORDER BY a.alias) AS aliases
                    FROM rag_entity_aliases_v3 a
                    WHERE a.domain = 'item'
                      AND a.entity_id = i.id
                      AND a.lang = '{lang}'
                      AND a.status = 'approved'
                ) ia ON true
                WHERE itp.trader_id = ANY($1::text[])
            )
            SELECT *
            FROM ranked
            WHERE rn <= 50
            ORDER BY trader_id, game_mode, price DESC NULLS LAST
        """,
    }

    for detail_key, sql in queries.items():
        rows = await conn.fetch(sql, trader_ids)
        for row in rows:
            details[row["trader_id"]][detail_key].append(dict(row))

    return details


def _base_metadata(row: dict, trader_name: str) -> dict:
    return {
        "domain": "trader",
        "entity_id": row["id"],
        "entity_name": trader_name,
        "normalized_name": row.get("normalized_name"),
        "is_use": row.get("is_use"),
    }


def build_trader_chunks(row: dict, details: dict[str, list[dict]], lang: Lang) -> list[RagChunkV3]:
    lb = LABELS[lang]
    trader_id = row["id"]
    trader_name, _aliases = _trader_name(row, lang)
    common = {
        "domain": "trader",
        "entity_id": trader_id,
        "lang": lang,
        "source_table": "traders",
        "source_updated_at": row.get("update_time"),
    }
    metadata_base = _base_metadata(row, trader_name)
    header = _header(row, lang)
    image = row.get("image")
    barters = details.get("barters", [])
    prices = details.get("prices", [])
    inputs_by_barter: dict[str, list[dict]] = defaultdict(list)
    for input_row in details.get("barter_inputs", []):
        inputs_by_barter[input_row["barter_id"]].append(input_row)

    chunks: list[RagChunkV3] = [
        RagChunkV3(
            **common,
            chunk_id=chunk_id(trader_id, "identifier"),
            chunk_type="identifier",
            content=header,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=trader_name,
                section="identifier",
                image=image,
                extra=metadata_base,
            ),
        )
    ]

    retrieval_content = clean_parts(
        [
            header,
            f"\n[{lb['levels']}]\n" + _format_level_counts(barters, lang) if barters else None,
            f"\n[{lb['barter_rewards']}]\n" + _format_barter_rewards(barters, lang, limit=35)
            if barters
            else None,
            f"\n[{lb['prices']}]\n{lb['price_items']}: {len(prices)}" if prices else None,
            f"\n[{lb['related_info']}]\n- {lb['barters']}" if barters else None,
        ]
    )
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(trader_id, "retrieval"),
            chunk_type="retrieval",
            content=retrieval_content,
            searchable=True,
            metadata=with_common_metadata(
                **common,
                entity_name=trader_name,
                section="retrieval",
                image=image,
                extra=metadata_base,
            ),
        )
    )

    if barters:
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(trader_id, "summary", "barters"),
                chunk_type="summary",
                content=clean_parts(
                    [header, f"\n[{lb['barter_rewards']}]\n" + _format_barter_rewards(barters, lang)]
                ),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=trader_name,
                    section="barters",
                    image=image,
                    extra={**metadata_base, "relation_counts": {"barters": len(barters)}},
                ),
            )
        )

    content_sections = [
        header,
        f"\n[{lb['levels']}]\n" + _format_level_counts(barters, lang) if barters else None,
        f"\n[{lb['prices']}]\n" + _format_price_summary(prices, lang, limit=30) if prices else None,
    ]
    chunks.append(
        RagChunkV3(
            **common,
            chunk_id=chunk_id(trader_id, "content", "main"),
            chunk_type="content",
            content=clean_parts(content_sections),
            searchable=False,
            metadata=with_common_metadata(
                **common,
                entity_name=trader_name,
                section="main",
                image=image,
                extra=metadata_base,
            ),
        )
    )

    if barters:
        chunks.append(
            RagChunkV3(
                **common,
                chunk_id=chunk_id(trader_id, "relation"),
                chunk_type="relation",
                content=clean_parts(
                    [f"{lb['trader']}: {trader_name}", f"[{lb['barters']}]\n" + _format_barters(barters, inputs_by_barter, lang)]
                ),
                searchable=True,
                metadata=with_common_metadata(
                    **common,
                    entity_name=trader_name,
                    section="relation",
                    image=image,
                    extra={
                        **metadata_base,
                        "relation_counts": {
                            "barters": len(barters),
                            "prices": len(prices),
                        },
                    },
                ),
            )
        )

    return chunks


async def build_and_upsert_traders(
    trader_id: str | None = None,
    limit: int | None = None,
    langs: tuple[Lang, ...] = SUPPORTED_LANGS,
    dry_run: bool = False,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await fetch_trader_rows(conn, trader_id=trader_id, limit=limit)
        trader_ids = [row["id"] for row in rows]
        total = 0

        for lang in langs:
            details = await fetch_trader_details(conn, trader_ids, lang)
            chunks = []
            for row in rows:
                chunks.extend(build_trader_chunks(row, details.get(row["id"], {}), lang))

            if dry_run:
                for chunk in chunks[:12]:
                    log.info("[dry-run] %s %s %s", chunk.lang, chunk.chunk_id, chunk.chunk_type)
                    log.info("\n%s", chunk.content[:1600])
                total += len(chunks)
            else:
                total += await upsert_rag_chunks_v3(conn, chunks)

        log.info("[trader_builder_v3] traders=%s chunks=%s dry_run=%s", len(rows), total, dry_run)
        return total


def _parse_langs(raw: str) -> tuple[Lang, ...]:
    langs = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = [lang for lang in langs if lang not in SUPPORTED_LANGS]
    if invalid:
        raise ValueError(f"unsupported langs: {invalid}")
    return langs or SUPPORTED_LANGS


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Build V3 trader RAG chunks.")
    parser.add_argument("--trader-id", help="Build a single trader id.")
    parser.add_argument("--limit", type=int, help="Build a limited number of traders.")
    parser.add_argument("--langs", default="ko,en,ja", help="Comma-separated langs: ko,en,ja")
    parser.add_argument("--dry-run", action="store_true", help="Print chunks without upserting.")
    args = parser.parse_args()

    try:
        await build_and_upsert_traders(
            trader_id=args.trader_id,
            limit=args.limit,
            langs=_parse_langs(args.langs),
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
