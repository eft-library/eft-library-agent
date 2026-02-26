"""
hideout 배치 임베딩 스크립트 (level 단위)
- hideout_master_i18n + hideout_level_i18n 기준
- hideout_item_require_i18n, hideout_skill_require_i18n
- hideout_station_require_i18n, hideout_trader_require_i18n
- hideout_bonus_i18n, hideout_crafts_i18n 조인
- bge-m3로 임베딩 생성
- rag_documents 테이블에 upsert
"""

import asyncio
import asyncpg
import httpx
import json
import logging
from dotenv import load_dotenv
import os

load_dotenv()

# 설정
DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")
BATCH_SIZE = 10
LANGS = ["ko", "en", "ja"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# 유틸
def get_lang_value(jsonb_field: dict | str | None, lang: str) -> str:
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


def fmt_duration(seconds: int | None) -> str:
    """초 → 시간/분 변환"""
    if not seconds:
        return "0분"
    if seconds < 3600:
        return f"{seconds // 60}분"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}시간 {m}분" if m else f"{h}시간"


def fmt_duration_en(seconds: int | None) -> str:
    if not seconds:
        return "0min"
    if seconds < 3600:
        return f"{seconds // 60}min"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}min" if m else f"{h}h"


def fmt_duration_ja(seconds: int | None) -> str:
    if not seconds:
        return "0分"
    if seconds < 3600:
        return f"{seconds // 60}分"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}時間 {m}分" if m else f"{h}時間"


DURATION_FMT = {
    "ko": fmt_duration,
    "en": fmt_duration_en,
    "ja": fmt_duration_ja,
}

NAME_KEY = {"ko": "name_ko", "en": "name_en", "ja": "name_ja"}

# 라벨
LABELS = {
    "ko": {
        "hideout": "은신처",
        "level": "레벨",
        "build_time": "건설 시간",
        "items": "건설 필요 아이템",
        "skills": "필요 스킬",
        "stations": "필요 시설",
        "traders": "필요 상인",
        "bonuses": "레벨업 보너스",
        "crafts": "제작 레시피",
        "inraid": "인레이드",
        "duration": "제작 시간",
        "reward": "결과물",
        "requires": "필요 재료",
        "none": "없음",
        "lv": "레벨",
    },
    "en": {
        "hideout": "Hideout",
        "level": "Level",
        "build_time": "Build Time",
        "items": "Required Items",
        "skills": "Required Skills",
        "stations": "Required Stations",
        "traders": "Required Traders",
        "bonuses": "Level Bonuses",
        "crafts": "Craft Recipes",
        "inraid": "Found in Raid",
        "duration": "Craft Time",
        "reward": "Output",
        "requires": "Materials",
        "none": "None",
        "lv": "Level",
    },
    "ja": {
        "hideout": "隠れ家",
        "level": "レベル",
        "build_time": "建設時間",
        "items": "建設必要アイテム",
        "skills": "必要スキル",
        "stations": "必要施設",
        "traders": "必要商人",
        "bonuses": "レベルアップボーナス",
        "crafts": "クラフトレシピ",
        "inraid": "レイド内発見",
        "duration": "クラフト時間",
        "reward": "生成物",
        "requires": "必要素材",
        "none": "なし",
        "lv": "レベル",
    },
}


# content 조합
def build_content(
    master_name: dict,
    level_row: dict,
    items: list,
    skills: list,
    stations: list,
    traders: list,
    bonuses: list,
    crafts: list,
    lang: str,
) -> str:
    lb = LABELS[lang]
    nk = NAME_KEY[lang]
    dur_fmt = DURATION_FMT[lang]

    hideout_name = get_lang_value(master_name, lang)
    level_num = level_row.get("level", "")
    construction_sec = level_row.get("construction_time") or 0

    parts = [
        f"{lb['hideout']}: {hideout_name}",
        f"{lb['level']}: {level_num}",
        f"{lb['build_time']}: {dur_fmt(construction_sec)}",
    ]

    # 건설 필요 아이템
    if items:
        lines = []
        for item in items:
            name = get_lang_value(item.get("name"), lang)
            quantity = item.get("quantity") or item.get("count", "")
            inraid = f" ({lb['inraid']})" if item.get("found_in_raid") else ""
            lines.append(f"- {name.strip()} x{quantity}{inraid}")
        parts.append(f"\n[{lb['items']}]\n" + "\n".join(lines))

    # 필요 스킬
    if skills:
        lines = []
        for s in skills:
            name = get_lang_value(s.get("name"), lang)
            level = s.get("level", "")
            lines.append(f"- {name} {lb['lv']}{level}")
        parts.append(f"\n[{lb['skills']}]\n" + "\n".join(lines))

    # 필요 시설
    if stations:
        lines = []
        for st in stations:
            name = get_lang_value(st.get("name"), lang)
            level = st.get("level", "")
            lines.append(f"- {name} {lb['lv']}{level}")
        parts.append(f"\n[{lb['stations']}]\n" + "\n".join(lines))

    # 필요 상인
    if traders:
        lines = []
        for t in traders:
            name = get_lang_value(t.get("name"), lang)
            value = t.get("value", "")
            lines.append(f"- {name} {lb['lv']}{value}")
        parts.append(f"\n[{lb['traders']}]\n" + "\n".join(lines))

    # 레벨업 보너스
    if bonuses:
        lines = []
        for b in bonuses:
            name = get_lang_value(b.get("name"), lang)
            skill_name = get_lang_value(b.get("skill_name"), lang)
            value = b.get("value")
            val_str = f"{float(value):+.4g}" if value is not None else ""
            line = f"- {name}: {val_str}"
            if skill_name:
                line += f" ({skill_name})"
            lines.append(line)
        parts.append(f"\n[{lb['bonuses']}]\n" + "\n".join(lines))

    # 제작 레시피
    if crafts:
        craft_parts = []
        for c in crafts:
            craft_name = get_lang_value(c.get("name"), lang)
            quantity = c.get("quantity") or 1
            duration = c.get("duration") or 0
            req_items = parse_jsonb(c.get("req_item")) or []

            lines = [
                f"{lb['reward']}: {craft_name.strip()} x{quantity}",
                f"{lb['duration']}: {dur_fmt(int(duration))}",
            ]
            if req_items:
                req_lines = []
                for r in req_items:
                    item = r.get("item") or {}
                    item_name = item.get(nk) or item.get("name_en", "")
                    qty = r.get("quantity", "")
                    req_lines.append(f"  · {item_name.strip()} x{qty}")
                lines.append(f"{lb['requires']}:\n" + "\n".join(req_lines))

            craft_parts.append("\n".join(lines))

        parts.append(f"\n[{lb['crafts']}]\n" + "\n\n".join(craft_parts))
    else:
        parts.append(f"\n[{lb['crafts']}]\n{lb['none']}")

    return "\n".join(parts).strip()


# 임베딩 생성
async def get_embedding(client: httpx.AsyncClient, text: str) -> list[float]:
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


# DB upsert
async def upsert_rag_document(
    conn: asyncpg.Connection,
    source_id: str,
    lang: str,
    content: str,
    embedding: list[float],
    metadata: dict,
):
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    await conn.execute(
        """
        INSERT INTO rag_documents (source_table, source_id, lang, content, embedding, metadata)
        VALUES ($1, $2, $3, $4, $5::vector, $6)
        ON CONFLICT (source_table, source_id, lang)
        DO UPDATE SET
            content    = EXCLUDED.content,
            embedding  = EXCLUDED.embedding,
            metadata   = EXCLUDED.metadata,
            updated_at = NOW()
    """,
        "hideout_level_i18n",
        source_id,
        lang,
        content,
        embedding_str,
        json.dumps(metadata, ensure_ascii=False),
    )


# 레벨 처리
async def process_level(
    pool: asyncpg.Pool,
    client: httpx.AsyncClient,
    master_row: dict,
    level_row: dict,
    items: list,
    skills: list,
    stations: list,
    traders: list,
    bonuses: list,
    crafts: list,
):
    level_id = level_row["id"]
    master_id = master_row["id"]
    level_num = level_row.get("level", "")
    master_name = master_row["name"]

    for lang in LANGS:
        content = build_content(
            master_name,
            level_row,
            items,
            skills,
            stations,
            traders,
            bonuses,
            crafts,
            lang,
        )

        if not content.strip():
            log.warning(f"  ⚠ 빈 content 스킵: {level_id} [{lang}]")
            continue

        try:
            embedding = await get_embedding(client, content)

            metadata = {
                "content_type": "joined",
                "source_tables": [
                    "hideout_master_i18n",
                    "hideout_level_i18n",
                    "hideout_item_require_i18n",
                    "hideout_skill_require_i18n",
                    "hideout_station_require_i18n",
                    "hideout_trader_require_i18n",
                    "hideout_bonus_i18n",
                    "hideout_crafts_i18n",
                ],
                "master_id": master_id,
                "level_id": level_id,
                "level": level_num,
                "hideout_name": {
                    "ko": get_lang_value(master_name, "ko"),
                    "en": get_lang_value(master_name, "en"),
                    "ja": get_lang_value(master_name, "ja"),
                },
                "craft_count": len(crafts),
                "url": f"https://eftlibrary.com/hideout",
            }

            # ── 확인용 출력 ──────────────────────────────────────
            # print(f"\n{'='*60}")
            # print(f"[{level_id}] [{lang}]")
            # print(f"{'─'*60}")
            # print(f"[content]\n{content}")
            # print(f"{'─'*60}")
            # print(f"[metadata]\n{json.dumps(metadata, ensure_ascii=False, indent=2)}")
            # print(f"{'─'*60}")
            # print(f"[embedding] 차원: {len(embedding)} | 앞 5개: {embedding[:5]}")
            # print(f"{'='*60}")
            # ────────────────────────────────────────────────────

            async with pool.acquire() as conn:
                await upsert_rag_document(
                    conn, level_id, lang, content, embedding, metadata
                )
            log.info(f"  ✓ {level_id} [{lang}] 완료")

        except httpx.HTTPError as e:
            log.error(f"  ✗ 임베딩 실패: {level_id} [{lang}] - {e}")
        except asyncpg.PostgresError as e:
            log.error(f"  ✗ DB 저장 실패: {level_id} [{lang}] - {e}")


# 메인
async def main():
    log.info("=== hideout 배치 임베딩 시작 ===")
    log.info(f"모델: {EMBED_MODEL} | 배치 크기: {BATCH_SIZE} | 언어: {LANGS}")

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=5)
    client = httpx.AsyncClient()

    try:
        # master 전체 조회
        async with pool.acquire() as conn:
            master_rows = await conn.fetch("""
                SELECT id, name, level_ids
                FROM hideout_master_i18n
                ORDER BY id ASC
            """)
        log.info(f"총 {len(master_rows)}개 hideout master 로드")

        processed = 0

        for master in master_rows:
            master_dict = dict(master)
            level_ids = list(master["level_ids"] or [])
            master_id = master["id"]
            master_name = get_lang_value(master["name"], "ko")

            if not level_ids:
                continue

            # 해당 master의 level 전체 조회
            async with pool.acquire() as conn:
                level_rows = await conn.fetch(
                    """
                    SELECT id, level, construction_time
                    FROM hideout_level_i18n
                    WHERE id = ANY($1)
                    ORDER BY level ASC
                """,
                    level_ids,
                )

            log.info(f"처리중: {master_name} ({master_id}) | {len(level_rows)}개 레벨")

            for level_row in level_rows:
                level_dict = dict(level_row)
                level_id = level_row["id"]

                # 하위 데이터 순차 조회 (단일 커넥션으로 안전하게)
                async with pool.acquire() as conn:
                    items = await conn.fetch(
                        "SELECT name, quantity, count, found_in_raid FROM hideout_item_require_i18n WHERE level_id = $1 ORDER BY id ASC",
                        level_id,
                    )
                    skills = await conn.fetch(
                        "SELECT name, level FROM hideout_skill_require_i18n WHERE level_id = $1 ORDER BY id ASC",
                        level_id,
                    )
                    stations = await conn.fetch(
                        "SELECT name, level FROM hideout_station_require_i18n WHERE level_id = $1 ORDER BY id ASC",
                        level_id,
                    )
                    traders = await conn.fetch(
                        "SELECT name, value FROM hideout_trader_require_i18n WHERE level_id = $1 ORDER BY id ASC",
                        level_id,
                    )
                    bonuses = await conn.fetch(
                        "SELECT name, skill_name, value FROM hideout_bonus_i18n WHERE level_id = $1 ORDER BY type ASC",
                        level_id,
                    )
                    crafts = await conn.fetch(
                        "SELECT name, quantity, duration, req_item FROM hideout_crafts_i18n WHERE level_id = $1 ORDER BY id ASC",
                        level_id,
                    )

                await process_level(
                    pool,
                    client,
                    master_dict,
                    level_dict,
                    [dict(r) for r in items],
                    [dict(r) for r in skills],
                    [dict(r) for r in stations],
                    [dict(r) for r in traders],
                    [dict(r) for r in bonuses],
                    [dict(r) for r in crafts],
                )
                processed += 1

        log.info(
            f"=== 완료: {processed}개 레벨, {processed * 3}개 row 생성/업데이트 ==="
        )

    finally:
        await client.aclose()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
