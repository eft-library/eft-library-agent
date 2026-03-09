"""
map_group_i18n + extraction_i18n + transit_i18n 배치 임베딩 스크립트
- 맵 기본 정보 / 탈출구 / 트랜짓 3개 문서로 분리
- bge-m3로 임베딩 생성
- rag_documents 테이블에 upsert

청크 분리:
  - {map_id}           : 이름만 (chunk_type: identifier) → RDB 조회용
  - {map_id}_main      : 기본 정보 + 구역 목록 + 스폰 보스 (chunk_type: content)
  - {map_id}_extract   : 탈출구 목록 (chunk_type: content)
  - {map_id}_transit   : 트랜짓 목록 (chunk_type: content)
"""

import asyncio
import asyncpg
import httpx
import json
import logging
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")
LANGS = ["ko", "en", "ja"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── 유틸 ──────────────────────────────────────────────────────────────────────

SKIP_HEADERS = {"아이콘", "icon", "アイコン"}


def clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for img in soup.find_all("img"):
        img.decompose()
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        headers, skip_indices, lines = [], set(), []
        for i, row in enumerate(rows):
            cols = row.find_all(["th", "td"])
            values = [c.get_text(strip=True) for c in cols]
            if i == 0:
                skip_indices = {j for j, h in enumerate(values) if h in SKIP_HEADERS}
                headers = [h for j, h in enumerate(values) if j not in skip_indices]
            else:
                values = [v for j, v in enumerate(values) if j not in skip_indices]
                line = (
                    " | ".join(f"{h}: {v}" for h, v in zip(headers, values))
                    if headers
                    else " | ".join(values)
                )
                lines.append(line)
        table.replace_with(soup.new_string("\n" + "\n".join(lines) + "\n"))
    return soup.get_text(separator="\n", strip=True)


def get_lang_value(jsonb_field, lang: str) -> str:
    if not jsonb_field:
        return ""
    if isinstance(jsonb_field, str):
        try:
            jsonb_field = json.loads(jsonb_field)
        except json.JSONDecodeError:
            return ""
    return jsonb_field.get(lang, "") or ""


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


# ── 라벨 ──────────────────────────────────────────────────────────────────────

LANG_LABELS = {
    "ko": {
        "map": "지도",
        "sub_areas": "구역 목록",
        "extractions": "탈출구 목록",
        "extraction": "탈출구",
        "transits": "트랜짓 목록",
        "transit": "트랜짓",
        "faction": "소속",
        "always_available": "항상 열림",
        "single_use": "1회용",
        "requirements": "요구사항",
        "tip": "팁",
        "yes": "✅",
        "no": "❌",
        "boss_spawn": "스폰 보스",
    },
    "en": {
        "map": "Map",
        "sub_areas": "Sub Areas",
        "extractions": "Extraction Points",
        "extraction": "Extraction",
        "transits": "Transit Points",
        "transit": "Transit",
        "faction": "Faction",
        "always_available": "Always Available",
        "single_use": "Single Use",
        "requirements": "Requirements",
        "tip": "Tip",
        "yes": "✅",
        "no": "❌",
        "boss_spawn": "Spawning Bosses",
    },
    "ja": {
        "map": "マップ",
        "sub_areas": "エリア一覧",
        "extractions": "脱出ポイント一覧",
        "extraction": "脱出ポイント",
        "transits": "トランジット一覧",
        "transit": "トランジット",
        "faction": "所属",
        "always_available": "常時利用可能",
        "single_use": "一回限り",
        "requirements": "必要条件",
        "tip": "ヒント",
        "yes": "✅",
        "no": "❌",
        "boss_spawn": "スポーンボス",
    },
}


# ── content 빌더 ───────────────────────────────────────────────────────────────


def build_identifier_content(map_row: dict, lang: str) -> str:
    label = LANG_LABELS[lang]
    map_name_ko = get_lang_value(map_row["name"], "ko")
    map_name_en = get_lang_value(map_row["name"], "en")
    map_name_ja = get_lang_value(map_row["name"], "ja")
    return f"{label['map']}: {map_name_ko} | {map_name_en} | {map_name_ja}"


def build_map_content(map_row: dict, sub_areas: list, bosses: list, lang: str) -> str:
    """기본 정보 + 구역 목록 + 스폰 보스 (chunk_type: content)"""
    label = LANG_LABELS[lang]
    map_name = get_lang_value(map_row["name"], lang)
    map_name_ko = get_lang_value(map_row["name"], "ko")
    map_name_en = get_lang_value(map_row["name"], "en")
    map_id = map_row["id"]

    keywords = {
        "ko": f"{map_name} 맵 지도 기본 정보 구역 {map_name_en}",
        "en": f"{map_name} map basic info areas {map_name_ko}",
        "ja": f"{map_name} マップ 基本情報 エリア {map_name_en}",
    }
    parts = [keywords[lang], f"{label['map']}: {map_name}"]

    if sub_areas:
        lines = [
            f"- {get_lang_value(a['name'], lang)}"
            for a in sub_areas
            if get_lang_value(a["name"], lang)
        ]
        if lines:
            parts.append(f"\n[{label['sub_areas']}]\n" + "\n".join(lines))

    if bosses:
        lines = []
        for boss in bosses:
            boss_name = get_lang_value(boss["name"], lang)
            spawn_chance = parse_jsonb(boss.get("spawn_chance")) or []

            # 해당 맵의 스폰 확률 찾기
            chance_str = ""
            for sc in spawn_chance:
                sc_map_id = sc.get("name_en", "").upper().replace(" ", "_")
                if sc_map_id == map_id.upper():
                    chance = sc.get("spawnChance", "")
                    if chance != "":
                        chance_str = f" ({int(float(chance) * 100)}%)"
                    break

            lines.append(f"- {boss_name}{chance_str}")

        parts.append(f"\n[{label['boss_spawn']}]\n" + "\n".join(lines))

    return "\n".join(parts).strip()


def build_extraction_content(map_row: dict, extractions: list, lang: str) -> str:
    """탈출구 목록 (chunk_type: content)"""
    label = LANG_LABELS[lang]
    map_name = get_lang_value(map_row["name"], lang)
    map_name_ko = get_lang_value(map_row["name"], "ko")
    map_name_en = get_lang_value(map_row["name"], "en")
    keywords = {
        "ko": f"{map_name} 탈출구 탈출 포인트 출구 {map_name_en}",
        "en": f"{map_name} extraction points exits {map_name_ko}",
        "ja": f"{map_name} 脱出ポイント 出口 {map_name_en}",
    }
    parts = [keywords[lang], f"{label['map']}: {map_name}"]
    ext_parts = []
    for ext in extractions:
        ext_name = get_lang_value(ext["name"], lang)
        faction = ext.get("faction") or ""
        always = label["yes"] if ext.get("always_available") else label["no"]
        single = label["yes"] if ext.get("single_use") else label["no"]
        requirements = clean_html(get_lang_value(ext.get("requirements"), lang))
        tip = clean_html(get_lang_value(ext.get("tip"), lang))
        lines = [f"{label['extraction']}: {ext_name}"]
        if faction:
            lines.append(f"{label['faction']}: {faction}")
        lines.append(f"{label['always_available']}: {always}")
        lines.append(f"{label['single_use']}: {single}")
        if requirements:
            lines.append(f"{label['requirements']}: {requirements}")
        if tip:
            lines.append(f"{label['tip']}: {tip}")
        ext_parts.append("\n".join(lines))
    parts.append(f"\n[{label['extractions']}]\n" + "\n\n".join(ext_parts))
    return "\n".join(parts).strip()


def build_transit_content(map_row: dict, transits: list, lang: str) -> str:
    """트랜짓 목록 (chunk_type: content)"""
    label = LANG_LABELS[lang]
    map_name = get_lang_value(map_row["name"], lang)
    map_name_ko = get_lang_value(map_row["name"], "ko")
    map_name_en = get_lang_value(map_row["name"], "en")
    keywords = {
        "ko": f"{map_name} 트랜짓 이동 환승 다른 맵 이동 {map_name_en}",
        "en": f"{map_name} transit travel between maps {map_name_ko}",
        "ja": f"{map_name} トランジット マップ間移動 {map_name_en}",
    }
    parts = [keywords[lang], f"{label['map']}: {map_name}"]
    transit_parts = []
    for tr in transits:
        tr_name = get_lang_value(tr["name"], lang)
        faction = tr.get("faction") or ""
        always = label["yes"] if tr.get("always_available") else label["no"]
        single = label["yes"] if tr.get("single_use") else label["no"]
        requirements = clean_html(get_lang_value(tr.get("requirements"), lang))
        tip = clean_html(get_lang_value(tr.get("tip"), lang))
        lines = [f"{label['transit']}: {tr_name}"]
        if faction:
            lines.append(f"{label['faction']}: {faction}")
        lines.append(f"{label['always_available']}: {always}")
        lines.append(f"{label['single_use']}: {single}")
        if requirements:
            lines.append(f"{label['requirements']}: {requirements}")
        if tip:
            lines.append(f"{label['tip']}: {tip}")
        transit_parts.append("\n".join(lines))
    parts.append(f"\n[{label['transits']}]\n" + "\n\n".join(transit_parts))
    return "\n".join(parts).strip()


# ── 임베딩 + upsert ────────────────────────────────────────────────────────────


async def get_embedding(client: httpx.AsyncClient, text: str) -> list[float]:
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


async def upsert_rag_document(
    conn: asyncpg.Connection,
    source_id: str,
    lang: str,
    content: str,
    embedding: list[float],
    chunk_type: str,
    ref_type: str,
    ref_id: str,
    metadata: dict,
):
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    await conn.execute(
        """
        INSERT INTO rag_documents (
            source_table, source_id, lang,
            content, embedding,
            chunk_type, ref_type, ref_id,
            metadata
        )
        VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $9)
        ON CONFLICT (source_table, source_id, lang, chunk_type)
        DO UPDATE SET
            content    = EXCLUDED.content,
            embedding  = EXCLUDED.embedding,
            ref_type   = EXCLUDED.ref_type,
            ref_id     = EXCLUDED.ref_id,
            metadata   = EXCLUDED.metadata,
            updated_at = NOW()
        """,
        "map_group_i18n",
        source_id,
        lang,
        content,
        embedding_str,
        chunk_type,
        ref_type,
        ref_id,
        json.dumps(metadata, ensure_ascii=False),
    )


# ── 맵별 처리 ──────────────────────────────────────────────────────────────────


async def process_map(conn, client, map_row, sub_areas, extractions, transits):
    map_id = map_row["id"]

    # 해당 맵에 스폰하는 보스 조회
    boss_rows = await conn.fetch(
        """
        SELECT name, spawn_chance
        FROM boss_i18n
        WHERE is_boss = true
          AND $1 = ANY(spawn_map)
        ORDER BY "order" ASC NULLS LAST
        """,
        map_id.upper(),
    )
    bosses = [dict(r) for r in boss_rows]
    log.info(f"  보스 {len(bosses)}개 조회됨: {map_id}")

    base_metadata = {
        "map_id": map_id,
        "map_name": {
            "ko": get_lang_value(map_row["name"], "ko"),
            "en": get_lang_value(map_row["name"], "en"),
            "ja": get_lang_value(map_row["name"], "ja"),
        },
        "url": f"https://eftlibrary.com/map-of-tarkov/{map_id}",
    }

    docs = [
        {
            "source_id": map_id,
            "chunk_type": "identifier",
            "ref_type": "map",
            "ref_id": map_id,
            "build_fn": lambda lang: build_identifier_content(map_row, lang),
            "extra": {},
            "skip": False,
        },
        {
            "source_id": f"{map_id}_main",
            "chunk_type": "content",
            "ref_type": "map",
            "ref_id": map_id,
            "build_fn": lambda lang: build_map_content(
                map_row, sub_areas, bosses, lang
            ),
            "extra": {"sub_area_count": len(sub_areas), "boss_count": len(bosses)},
            "skip": False,
        },
        {
            "source_id": f"{map_id}_extract",
            "chunk_type": "content",
            "ref_type": "map",
            "ref_id": map_id,
            "build_fn": lambda lang: build_extraction_content(
                map_row, extractions, lang
            ),
            "extra": {"extraction_count": len(extractions)},
            "skip": not extractions,
        },
        {
            "source_id": f"{map_id}_transit",
            "chunk_type": "content",
            "ref_type": "map",
            "ref_id": map_id,
            "build_fn": lambda lang: build_transit_content(map_row, transits, lang),
            "extra": {"transit_count": len(transits)},
            "skip": not transits,
        },
    ]

    for doc in docs:
        if doc["skip"]:
            log.info(f"  - {doc['source_id']} 스킵")
            continue
        for lang in LANGS:
            content = doc["build_fn"](lang)
            if not content.strip():
                log.warning(f"  ⚠ 빈 content 스킵: {doc['source_id']} [{lang}]")
                continue
            try:
                embedding = await get_embedding(client, content)
                metadata = {**base_metadata, **doc["extra"]}
                await upsert_rag_document(
                    conn,
                    doc["source_id"],
                    lang,
                    content,
                    embedding,
                    doc["chunk_type"],
                    doc["ref_type"],
                    doc["ref_id"],
                    metadata,
                )
                log.info(f"  ✓ {doc['source_id']} [{lang}] 완료")
            except httpx.HTTPError as e:
                log.error(f"  ✗ 임베딩 실패: {doc['source_id']} [{lang}] - {e}")
            except asyncpg.PostgresError as e:
                log.error(f"  ✗ DB 저장 실패: {doc['source_id']} [{lang}] - {e}")


# ── 메인 ──────────────────────────────────────────────────────────────────────


async def main():
    log.info("=== map 배치 임베딩 시작 ===")
    conn = await asyncpg.connect(DATABASE_URL)
    client = httpx.AsyncClient()

    try:
        map_rows = await conn.fetch(
            "SELECT id, name FROM map_group_i18n WHERE depth = 1 ORDER BY id ASC"
        )
        total = len(map_rows)
        log.info(f"총 {total}개 지도 → 최대 {total * 4}개 문서 × 3개 언어")

        for map_row in map_rows:
            map_id = map_row["id"]
            sub_areas = await conn.fetch(
                "SELECT id, name FROM map_group_i18n WHERE parent_value = $1 AND depth = 2 ORDER BY id ASC",
                map_id,
            )
            extractions = await conn.fetch(
                "SELECT id, name, faction, always_available, single_use, requirements, tip FROM extraction_i18n WHERE map = $1 ORDER BY faction ASC, id ASC",
                map_id,
            )
            transits = await conn.fetch(
                "SELECT id, name, faction, always_available, single_use, requirements, tip FROM transit_i18n WHERE map = $1 ORDER BY faction ASC, id ASC",
                map_id,
            )

            log.info(
                f"처리중: {map_id} | 구역 {len(sub_areas)}개 | 탈출구 {len(extractions)}개 | 트랜짓 {len(transits)}개"
            )
            await process_map(
                conn,
                client,
                dict(map_row),
                [dict(r) for r in sub_areas],
                [dict(r) for r in extractions],
                [dict(r) for r in transits],
            )

        log.info("=== 완료 ===")
    finally:
        await client.aclose()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
