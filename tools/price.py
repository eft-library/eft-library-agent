import json
import logging
from db.connection import get_pool

log = logging.getLogger(__name__)


def parse_jsonb(value) -> list | dict | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def format_price(price: int) -> str:
    return f"{price:,}₽"


PRICE_LABELS = {
    "ko": {
        "title": "시세 정보",
        "updated": "업데이트",
        "pve": "PvE",
        "pvp": "PvP",
        "flea": "플리마켓",
    },
    "en": {
        "title": "Price Info",
        "updated": "Updated",
        "pve": "PvE",
        "pvp": "PvP",
        "flea": "Flea Market",
    },
    "ja": {
        "title": "価格情報",
        "updated": "更新",
        "pve": "PvE",
        "pvp": "PvP",
        "flea": "フリーマーケット",
    },
}

NPC_NAME_KEY = {
    "ko": "npc_name_ko",
    "en": "npc_name_en",
    "ja": "npc_name_ja",
}

FLEA_IDS = {"FLEA_MARKET"}


def build_price_text(row: dict, lang: str) -> str:
    lb = PRICE_LABELS[lang]
    nk = NPC_NAME_KEY[lang]
    trader = parse_jsonb(row["trader"]) or {}
    update_time = row["update_time"]
    updated_str = update_time.strftime("%Y-%m-%d %H:%M") if update_time else ""

    lines = [f"[{lb['title']}] ({lb['updated']}: {updated_str})"]

    for mode in ("pve", "pvp"):
        traders = trader.get(f"{mode}_trader") or []
        if not traders:
            continue

        # 플리마켓 분리
        flea = next((t for t in traders if t["trader"]["npc_id"] in FLEA_IDS), None)
        npc_list = [t for t in traders if t["trader"]["npc_id"] not in FLEA_IDS]

        mode_lines = []
        for t in npc_list:
            name = t["trader"].get(nk) or t["trader"].get("npc_name_en", "")
            price = format_price(t["price"])
            mode_lines.append(f"  · {name}: {price}")
        if flea:
            price = format_price(flea["price"])
            mode_lines.append(f"  · {lb['flea']}: {price}")

        lines.append(f"{lb[mode]}:")
        lines.extend(mode_lines)

    return "\n".join(lines)


async def get_item_prices(item_ids: list[str], lang: str = "ko") -> dict[str, str]:
    """
    item_ids 목록으로 ITEM_PRICE_I18N 조회
    반환: { item_id: price_text }
    """
    if not item_ids:
        return {}

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, trader, update_time
            FROM item_price_i18n
            WHERE id = ANY($1)
        """,
            item_ids,
        )

    result = {}
    for row in rows:
        result[row["id"]] = build_price_text(dict(row), lang)
        log.info(f"[price] item={row['id']} lang={lang}")

    return result
