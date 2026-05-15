import re

from schemas.models_v3 import Domain


DOMAIN_ROUTE_BOOSTS: dict[Domain, float] = {
    "item": 1.35,
    "quest": 1.35,
    "map": 1.35,
    "boss": 1.35,
    "hideout": 1.25,
    "trader": 1.25,
    "information": 1.2,
    "story": 1.25,
}


MAP_ALIASES = {
    "팩토리",
    "공장",
    "세관",
    "쇼어라인",
    "해안선",
    "리저브",
    "등대",
    "라이트하우스",
    "인터체인지",
    "인타체인지",
    "우드",
    "삼림",
    "랩",
    "연구소",
    "타르코프시내",
    "타르코프 시내",
    "스트리트",
    "그라운드제로",
    "그라운드 제로",
    "미궁",
    "랩린스",
    "labyrinth",
    "factory",
    "customs",
    "shoreline",
    "reserve",
    "lighthouse",
    "interchange",
    "woods",
    "the lab",
    "streets",
    "ground zero",
}


BOSS_ALIASES = {
    "킬라",
    "타길라",
    "글루하",
    "글루카",
    "슈터본",
    "슈트르만",
    "세니타",
    "새니타",
    "카반",
    "콜론타이",
    "광신도 사제",
    "컬티스트",
    "로그",
    "나이트",
    "빅파이프",
    "버드아이",
    "레샬라",
    "킬러",
    "killa",
    "tagilla",
    "glukhar",
    "shturman",
    "sanitar",
    "kaban",
    "kollontay",
    "cultist",
    "knight",
    "big pipe",
    "birdeye",
    "reshala",
}


QUEST_ALIASES = {
    "데뷔",
    "수집가",
    "사격 연습",
    "재고 부족",
    "건스미스",
    "gunsmith",
}


def _normalized(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())


def _compact(query: str) -> str:
    return re.sub(r"\s+", "", query.strip().lower())


def infer_domain_boosts_v3(query: str) -> dict[Domain, float]:
    text = _normalized(query)
    compact = _compact(query)
    boosts: dict[Domain, float] = {}

    def boost(domain: Domain, multiplier: float | None = None) -> None:
        boosts[domain] = max(boosts.get(domain, 1.0), multiplier or DOMAIN_ROUTE_BOOSTS[domain])

    if re.search(r"(^|\s)(퀘스트|quest|미션|임무)[:\s]", text):
        boost("quest", 1.55)
    if re.search(r"(^|\s)(아이템|item)[:\s]", text):
        boost("item", 1.55)
    if re.search(r"(^|\s)(보스|boss)[:\s]", text):
        boost("boss", 1.55)
    if re.search(r"(^|\s)(지도|맵|map)[:\s]", text):
        boost("map", 1.55)
    if re.search(r"(^|\s)(은신처|hideout)[:\s]", text):
        boost("hideout", 1.55)
    if re.search(r"(^|\s)(상인|trader)[:\s]", text):
        boost("trader", 1.55)
    if re.search(r"(^|\s)(스토리|story)[:\s]", text):
        boost("story", 1.55)

    if any(term in compact for term in ("선행퀘스트", "후행퀘스트", "이전에는어떤퀘스트", "다음에는어떤퀘스트")):
        boost("quest", 1.45)
    if "퀘스트" in compact and any(term in compact for term in ("필요", "해금", "보상", "완료", "깨면", "열려")):
        boost("quest", 1.3)

    if any(term in compact for term in ("어디서얻", "어디서구", "파밍처", "구하는법", "필요한퀘스트")):
        boost("item", 1.35)
    if any(term in compact for term in ("어디서만들", "제작", "재료로제작", "어디에써", "어디에사용")):
        boost("item", 1.3)

    if any(term in compact for term in ("드랍하는아이템", "드랍템", "스폰위치", "어디나와")):
        if any(alias.replace(" ", "") in compact for alias in BOSS_ALIASES):
            boost("boss", 1.45)

    if any(term in compact for term in ("탈출구", "지도", "맵", "열쇠")):
        if any(alias.replace(" ", "") in compact for alias in MAP_ALIASES):
            boost("map", 1.35)
    if compact in {alias.replace(" ", "") for alias in MAP_ALIASES}:
        boost("map", 1.45)

    if any(term in compact for term in ("상점", "상인", "바터", "우호도")):
        boost("trader", 1.25)

    if any(term in compact for term in ("은신처", "시설", "채굴시설")):
        boost("hideout", 1.25)

    if any(term in compact for term in ("스토리", "로드맵", "구원자루트", "엔딩")):
        boost("story", 1.3)

    if any(term in compact for term in ("패치노트", "이벤트", "공지", "코드")):
        boost("information", 1.3)

    return boosts


def infer_domain_filter_v3(query: str) -> Domain | None:
    text = _normalized(query)
    compact = _compact(query)

    prefix_rules: tuple[tuple[Domain, tuple[str, ...]], ...] = (
        ("quest", ("퀘스트:", "quest:", "미션:", "임무:")),
        ("item", ("아이템:", "item:")),
        ("boss", ("보스:", "boss:")),
        ("map", ("지도:", "맵:", "map:")),
        ("hideout", ("은신처:", "hideout:")),
        ("trader", ("상인:", "trader:")),
        ("story", ("스토리:", "story:")),
    )
    for domain, prefixes in prefix_rules:
        if any(text.startswith(prefix) for prefix in prefixes):
            return domain

    if any(term in compact for term in ("선행퀘스트", "후행퀘스트", "이전에는어떤퀘스트", "다음에는어떤퀘스트")):
        return "quest"
    if "퀘스트" in compact and any(term in compact for term in ("완료해야", "다음에는", "이전에는", "깨면뭐가해금")):
        return "quest"

    if any(alias.replace(" ", "") in compact for alias in BOSS_ALIASES):
        if any(term in compact for term in ("드랍", "스폰", "어디나와", "정보알려")):
            return "boss"

    if compact in {alias.replace(" ", "") for alias in MAP_ALIASES}:
        return "map"
    if any(alias.replace(" ", "") in compact for alias in MAP_ALIASES):
        if any(term in compact for term in ("탈출구", "지도", "맵", "열쇠")):
            return "map"

    if any(term in compact for term in ("어디서만들", "어디서얻", "어디서구", "파밍처", "구하는법", "어디에써", "어디에사용")):
        return "item"
    if any(term in compact for term in ("보상으로얻을수있는퀘스트", "무슨퀘스트보상", "구매해금퀘스트", "필요한퀘스트")):
        return "item"
    if re.search(r"(^|[^a-z0-9])m80([^a-z0-9]|$)", text):
        return "item"
    if re.search(r"\b(ak|m4|mcx|rpk|vpo|rd)-?\d*", text) and "퀘스트" in compact:
        return "item"

    if any(term in compact for term in ("상점", "상인", "바터", "우호도")):
        return "trader"
    if any(term in compact for term in ("은신처", "채굴시설")):
        return "hideout"
    if "시설" in compact and "필요재료" in compact:
        return "hideout"
    if any(term in compact for term in ("스토리", "로드맵", "구원자루트", "엔딩")):
        return "story"
    if any(term in compact for term in ("패치노트", "최신패치", "진행중인이벤트", "현재이벤트", "사용가능한코드")):
        return "information"

    if "정보알려" in compact:
        for alias in QUEST_ALIASES:
            if alias.replace(" ", "").lower() in compact:
                return "quest"

    return None
