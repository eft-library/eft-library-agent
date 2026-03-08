import logging

log = logging.getLogger(__name__)

ROUTING_RULES: dict[str, list[str]] = {
    "quest_i18n": ["퀘스트:", "quest:", "임무:", "미션:", "クエスト:", "任務:"],
    "boss_i18n": ["보스:", "boss:", "ボス:"],
    "item_i18n": ["아이템:", "item:", "장비:", "アイテム:", "装備:"],
    "map_i18n": ["지도:", "맵:", "map:", "지역:", "マップ:", "地図:", "エリア:"],
}


def rule_based_routing(query: str) -> str | None:
    query_lower = query.lower().strip()
    for table, keywords in ROUTING_RULES.items():
        if any(query_lower.startswith(kw.lower()) for kw in keywords):
            log.info(f"[router] 규칙 라우팅: '{query[:30]}' → {table}")
            return table
    log.info(f"[router] 라우팅 실패, 전체 검색: '{query[:30]}'")
    return None
