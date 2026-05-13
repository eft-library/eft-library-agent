import argparse
import asyncio
import ast
import json
import logging
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv

from db.connection import close_pool, get_pool
from rag_builders_v3.base import SUPPORTED_LANGS, lang_value
from schemas.models_v3 import Lang

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_LLM_MODEL")

GENERIC_ALIASES = {
    "ko": {
        "총",
        "탄",
        "탄약",
        "키",
        "열쇠",
        "템",
        "아이템",
        "퀘템",
        "퀘스트템",
        "치료템",
        "약",
        "방어구",
        "가방",
    },
    "en": {"gun", "ammo", "key", "item", "quest item", "medicine", "armor", "bag"},
    "ja": {"銃", "弾", "鍵", "アイテム", "クエストアイテム"},
}

DEFAULT_EXCLUDED_PARENT_CATEGORIES = {
    "Ammo",
    "Compound item",
    "Cylinder Magazine",
    "Equipment",
    "Essential mod",
    "Functional mod",
    "Gear mod",
    "Info",
    "Magazine",
    "Muzzle device",
    "Sights",
    "Special scope",
    "Stackable item",
    "Weapon",
}


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _normalize_alias(alias: str) -> str:
    return re.sub(r"\s+", " ", alias).strip()


def _flatten_alias_candidates(values: Any) -> list[str]:
    flattened: list[str] = []
    if isinstance(values, str):
        stripped = values.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                return _flatten_alias_candidates(parsed)
            except (SyntaxError, ValueError):
                pass
        return [values]

    if isinstance(values, list):
        for value in values:
            flattened.extend(_flatten_alias_candidates(value))
        return flattened

    return []


def _filter_aliases(
    aliases: list[Any],
    *,
    lang: Lang,
    official_values: set[str],
    max_aliases: int,
) -> list[str]:
    filtered = []
    seen = set()
    generic = GENERIC_ALIASES.get(lang, set())

    for raw_alias in _flatten_alias_candidates(aliases):
        alias = _normalize_alias(str(raw_alias))
        alias_key = alias.lower()
        if not alias:
            continue
        if len(alias) < 2:
            continue
        if len(alias) > 30:
            continue
        if alias_key in generic:
            continue
        if alias_key in official_values:
            continue
        if lang == "ko":
            has_hangul = bool(re.search(r"[가-힣]", alias))
            if not has_hangul:
                continue
        if alias_key in seen:
            continue
        seen.add(alias_key)
        filtered.append(alias)
        if len(filtered) >= max_aliases:
            break

    return filtered


def _prompt_for_item(row: dict, lang: Lang, max_aliases: int) -> list[dict]:
    lang_name = {"ko": "Korean", "en": "English", "ja": "Japanese"}[lang]
    user_content = {
        "task": "Generate Escape from Tarkov item search aliases.",
        "language": lang_name,
        "rules": [
            "Return JSON only.",
            "Generate aliases real players might type in search.",
            "Prefer abbreviations, transliterations, romanized/Korean community terms, and short nicknames.",
            "For Korean, include Hangul pronunciation/transliteration for Latin brand names when natural.",
            "Do not include generic words like gun, ammo, key, item, medicine, armor, or bag.",
            "Do not duplicate official names, English names, Japanese names, or normalized slug.",
            "Return an empty aliases array when unsure.",
            f"Return at most {max_aliases} aliases.",
        ],
        "examples": [
            {
                "name_en": "Salewa first aid kit",
                "name_ko": "Salewa 응급 치료 키트",
                "lang": "ko",
                "aliases": ["살레와", "살레와킷"],
            },
            {
                "name_en": "Graphics card",
                "name_ko": "그래픽 카드",
                "lang": "ko",
                "aliases": ["글카"],
            },
            {
                "name_en": "Tetriz portable game console",
                "name_ko": "Tetriz 휴대용 게임기",
                "lang": "ko",
                "aliases": ["테트리즈"],
            },
        ],
        "item": {
            "id": row["id"],
            "name_en": row.get("name_en"),
            "name_ko": row.get("name_ko"),
            "name_ja": row.get("name_ja"),
            "normalized_name": row.get("normalized_name"),
            "parent_category": row.get("parent_category"),
            "category": row.get("category"),
        },
        "output_schema": {"aliases": ["string"]},
    }
    return [
        {
            "role": "system",
            "content": "You generate concise search alias candidates for game database retrieval. Output valid JSON only.",
        },
        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
    ]


async def generate_aliases_for_item(
    client: httpx.AsyncClient,
    row: dict,
    lang: Lang,
    max_aliases: int,
) -> list[str]:
    if not OLLAMA_BASE_URL:
        raise RuntimeError("OLLAMA_BASE_URL is not set")
    if not CHAT_MODEL:
        raise RuntimeError("OLLAMA_CHAT_MODEL or OLLAMA_LLM_MODEL is not set")

    payload = {
        "model": CHAT_MODEL,
        "messages": _prompt_for_item(row, lang, max_aliases),
        "stream": False,
        "think": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": int(os.getenv("NUM_CTX", "8192"))},
    }
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    data = _extract_json_object(content)
    if isinstance(data, list):
        aliases = data
    elif isinstance(data, dict):
        aliases = data.get("aliases", [])
    else:
        aliases = []
    if not isinstance(aliases, list):
        return []

    official_values = {
        str(value).strip().lower()
        for value in [
            row.get("name_en"),
            row.get("name_ko"),
            row.get("name_ja"),
            row.get("normalized_name"),
            lang_value(row, "name", lang),
        ]
        if value
    }
    return _filter_aliases(
        aliases,
        lang=lang,
        official_values=official_values,
        max_aliases=max_aliases,
    )


async def fetch_items(
    conn,
    *,
    item_id: str | None,
    limit: int | None,
    offset: int,
    only_missing: bool,
    lang: Lang,
    excluded_parent_categories: set[str],
) -> list[dict]:
    where = []
    args: list[Any] = []

    if item_id:
        args.append(item_id)
        where.append(f"i.id = ${len(args)}")

    if only_missing:
        args.append(lang)
        where.append(
            f"""
            NOT EXISTS (
                SELECT 1
                FROM rag_entity_aliases_v3 a
                WHERE a.domain = 'item'
                  AND a.entity_id = i.id
                  AND a.lang = ${len(args)}
                  AND a.source = 'llm'
            )
            """
        )

    if excluded_parent_categories:
        args.append(sorted(excluded_parent_categories))
        where.append(
            f"(i.parent_category IS NULL OR i.parent_category <> ALL(${len(args)}::text[]))"
        )

    where_clause = "WHERE " + " AND ".join(where) if where else ""
    limit_clause = ""
    if limit is not None:
        args.append(limit)
        limit_clause = f"LIMIT ${len(args)}"
    args.append(offset)
    offset_clause = f"OFFSET ${len(args)}"

    rows = await conn.fetch(
        f"""
        SELECT id, parent_category, category, name_en, name_ko, name_ja, normalized_name
        FROM items i
        {where_clause}
        ORDER BY i.update_time DESC, i.id
        {limit_clause}
        {offset_clause}
        """,
        *args,
    )
    return [dict(row) for row in rows]


async def upsert_alias_candidates(
    conn,
    *,
    entity_id: str,
    lang: Lang,
    aliases: list[str],
    confidence: float,
) -> int:
    count = 0
    for alias in aliases:
        await conn.execute(
            """
            INSERT INTO rag_entity_aliases_v3 (
                domain, entity_id, lang, alias, source, status, confidence, note
            )
            VALUES ('item', $1, $2, $3, 'llm', 'pending', $4, 'Generated by local chat model')
            ON CONFLICT (domain, entity_id, lang, alias)
            DO UPDATE SET
                source = EXCLUDED.source,
                confidence = EXCLUDED.confidence,
                note = EXCLUDED.note,
                updated_at = now()
            WHERE rag_entity_aliases_v3.status = 'pending'
            """,
            entity_id,
            lang,
            alias,
            confidence,
        )
        count += 1
    return count


async def generate_item_alias_candidates(
    *,
    item_id: str | None = None,
    lang: Lang = "ko",
    limit: int | None = None,
    offset: int = 0,
    max_aliases: int = 3,
    only_missing: bool = True,
    excluded_parent_categories: set[str] | None = None,
    dry_run: bool = False,
) -> int:
    pool = await get_pool()
    total = 0
    async with pool.acquire() as conn:
        rows = await fetch_items(
            conn,
            item_id=item_id,
            limit=limit,
            offset=offset,
            only_missing=only_missing,
            lang=lang,
            excluded_parent_categories=excluded_parent_categories
            if excluded_parent_categories is not None
            else DEFAULT_EXCLUDED_PARENT_CATEGORIES,
        )
        log.info(
            "[alias_v3] target items=%s lang=%s excluded_parent_categories=%s dry_run=%s",
            len(rows),
            lang,
            sorted(
                excluded_parent_categories
                if excluded_parent_categories is not None
                else DEFAULT_EXCLUDED_PARENT_CATEGORIES
            ),
            dry_run,
        )

        async with httpx.AsyncClient() as client:
            for index, row in enumerate(rows, 1):
                try:
                    aliases = await generate_aliases_for_item(
                        client, row, lang, max_aliases
                    )
                except Exception as exc:
                    log.warning(
                        "[alias_v3] failed item=%s name=%s error=%s",
                        row["id"],
                        lang_value(row, "name", lang),
                        exc,
                    )
                    continue
                name = lang_value(row, "name", lang)
                log.info(
                    "[alias_v3] %s/%s item=%s name=%s aliases=%s",
                    index,
                    len(rows),
                    row["id"],
                    name,
                    aliases,
                )

                if dry_run:
                    total += len(aliases)
                    continue

                total += await upsert_alias_candidates(
                    conn,
                    entity_id=row["id"],
                    lang=lang,
                    aliases=aliases,
                    confidence=0.7,
                )

    log.info("[alias_v3] stored candidates=%s dry_run=%s", total, dry_run)
    return total


def _parse_lang(raw: str) -> Lang:
    if raw not in SUPPORTED_LANGS:
        raise ValueError(f"unsupported lang: {raw}")
    return raw  # type: ignore[return-value]


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate pending item alias candidates.")
    parser.add_argument("--item-id", help="Generate aliases for a single item id.")
    parser.add_argument("--lang", default="ko", choices=["ko", "en", "ja"])
    parser.add_argument("--limit", type=int, help="Limit number of items.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-aliases", type=int, default=3)
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Also generate for items that already have llm alias candidates.",
    )
    parser.add_argument(
        "--include-excluded-parent-categories",
        action="store_true",
        help="Do not apply the default parent_category exclusion list.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        await generate_item_alias_candidates(
            item_id=args.item_id,
            lang=_parse_lang(args.lang),
            limit=args.limit,
            offset=args.offset,
            max_aliases=args.max_aliases,
            only_missing=not args.include_existing,
            excluded_parent_categories=set()
            if args.include_excluded_parent_categories
            else DEFAULT_EXCLUDED_PARENT_CATEGORIES,
            dry_run=args.dry_run,
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
