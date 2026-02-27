"""
map_group_i18n + extraction_i18n 배치 임베딩 스크립트
- map_group_i18n depth 1 (최상위 지도) 단위로 묶어서 임베딩
- 하위 구역(depth 2) 목록 포함
- 해당 지도의 탈출구 전체 포함
- bge-m3로 임베딩 생성
- rag_documents 테이블에 upsert
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


# 설정
DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
LANGS = ["ko", "en", "ja"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# HTML 파싱
SKIP_HEADERS = {"아이콘", "icon", "アイコン"}


def clean_html(html_text: str) -> str:
    """HTML 태그 제거, img 제거, table 구조화, 아이콘 컬럼 제거"""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")

    for img in soup.find_all("img"):
        img.decompose()

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        headers = []
        skip_indices = set()
        lines = []

        for i, row in enumerate(rows):
            cols = row.find_all(["th", "td"])
            values = [c.get_text(strip=True) for c in cols]
            if i == 0:
                skip_indices = {j for j, h in enumerate(values) if h in SKIP_HEADERS}
                headers = [h for j, h in enumerate(values) if j not in skip_indices]
            else:
                values = [v for j, v in enumerate(values) if j not in skip_indices]
                if headers:
                    line = " | ".join(f"{h}: {v}" for h, v in zip(headers, values))
                else:
                    line = " | ".join(values)
                lines.append(line)

        table.replace_with(soup.new_string("\n" + "\n".join(lines) + "\n"))

    return soup.get_text(separator="\n", strip=True)


def get_lang_value(jsonb_field: dict | str | None, lang: str) -> str:
    """JSONB {ko: '', en: '', ja: ''} 에서 언어별 값 추출"""
    if not jsonb_field:
        return ""
    if isinstance(jsonb_field, str):
        try:
            jsonb_field = json.loads(jsonb_field)
        except json.JSONDecodeError:
            return ""
    return jsonb_field.get(lang, "") or ""


def parse_jsonb(value) -> list | dict | None:
    """asyncpg JSONB 필드 str/dict/list 모두 처리"""
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


# content 조합
LANG_LABELS = {
    "ko": {
        "map": "지도",
        "sub_areas": "구역 목록",
        "extractions": "탈출구 목록",
        "extraction": "탈출구",
        "faction": "소속",
        "always_available": "항상 열림",
        "single_use": "1회용",
        "requirements": "요구사항",
        "tip": "팁",
        "yes": "예",
        "no": "아니오",
    },
    "en": {
        "map": "Map",
        "sub_areas": "Sub Areas",
        "extractions": "Extraction Points",
        "extraction": "Extraction",
        "faction": "Faction",
        "always_available": "Always Available",
        "single_use": "Single Use",
        "requirements": "Requirements",
        "tip": "Tip",
        "yes": "Yes",
        "no": "No",
    },
    "ja": {
        "map": "マップ",
        "sub_areas": "エリア一覧",
        "extractions": "脱出ポイント一覧",
        "extraction": "脱出ポイント",
        "faction": "所属",
        "always_available": "常時利用可能",
        "single_use": "一回限り",
        "requirements": "必要条件",
        "tip": "ヒント",
        "yes": "はい",
        "no": "いいえ",
    },
}


def build_content(map_row: dict, sub_areas: list, extractions: list, lang: str) -> str:
    """지도 + 구역 + 탈출구 조합 텍스트 생성"""
    label = LANG_LABELS[lang]
    map_name = get_lang_value(map_row["name"], lang)

    parts = [f"{label['map']}: {map_name}"]

    # 하위 구역 목록 (depth 2)
    if sub_areas:
        lines = []
        for area in sub_areas:
            area_name = get_lang_value(area["name"], lang)
            if area_name:
                lines.append(f"- {area_name}")
        if lines:
            parts.append(f"\n[{label['sub_areas']}]\n" + "\n".join(lines))

    # 탈출구 목록
    if extractions:
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
            lines.append(
                f"{label['always_available']}: {always} | {label['single_use']}: {single}"
            )
            if requirements:
                lines.append(f"{label['requirements']}: {requirements}")
            if tip:
                lines.append(f"{label['tip']}: {tip}")

            ext_parts.append("\n".join(lines))

        parts.append(f"\n[{label['extractions']}]\n" + "\n\n".join(ext_parts))

    return "\n".join(parts).strip()


# 임베딩 생성 (Ollama bge-m3)
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
        "map_group_i18n",
        source_id,
        lang,
        content,
        embedding_str,
        json.dumps(metadata, ensure_ascii=False),
    )


# 메인 배치 처리
async def process_map(
    conn: asyncpg.Connection,
    client: httpx.AsyncClient,
    map_row: dict,
    sub_areas: list,
    extractions: list,
):
    map_id = map_row["id"]

    for lang in LANGS:
        content = build_content(map_row, sub_areas, extractions, lang)

        if not content.strip():
            log.warning(f"  ⚠ 빈 content 스킵: {map_id} [{lang}]")
            continue

        try:
            embedding = await get_embedding(client, content)

            metadata = {
                "content_type": "joined",
                "source_tables": ["map_group_i18n", "extraction_i18n"],
                "map_id": map_id,
                "map_name": {
                    "ko": get_lang_value(map_row["name"], "ko"),
                    "en": get_lang_value(map_row["name"], "en"),
                    "ja": get_lang_value(map_row["name"], "ja"),
                },
                "sub_area_count": len(sub_areas),
                "extraction_count": len(extractions),
                "url": f"https://eftlibrary.com/map-of-tarkov/{map_id}",
            }

            # ── 확인용 출력 ──────────────────────────────────────
            # print(f"\n{'='*60}")
            # print(f"[{map_id}] [{lang}]")
            # print(f"{'─'*60}")
            # print(f"[content]\n{content}")
            # print(f"{'─'*60}")
            # print(f"[metadata]\n{json.dumps(metadata, ensure_ascii=False, indent=2)}")
            # print(f"{'─'*60}")
            # print(f"[embedding] 차원: {len(embedding)} | 앞 5개: {embedding[:5]}")
            # print(f"{'='*60}")
            # ─ ───────────────────────────────────────────────────

            await upsert_rag_document(conn, map_id, lang, content, embedding, metadata)
            log.info(f"  ✓ {map_id} [{lang}] 완료")

        except httpx.HTTPError as e:
            log.error(f"  ✗ 임베딩 실패: {map_id} [{lang}] - {e}")
        except asyncpg.PostgresError as e:
            log.error(f"  ✗ DB 저장 실패: {map_id} [{lang}] - {e}")


async def main():
    log.info("=== map_group_i18n + extraction_i18n 배치 임베딩 시작 ===")
    log.info(f"모델: {EMBED_MODEL} | 언어: {LANGS}")

    conn = await asyncpg.connect(DATABASE_URL)
    client = httpx.AsyncClient()

    try:
        # depth 1 최상위 지도만 조회
        map_rows = await conn.fetch("""
            SELECT id, name
            FROM map_group_i18n
            WHERE depth = 1
            ORDER BY id ASC
        """)

        total = len(map_rows)
        log.info(f"총 {total}개 지도 처리 예정 (언어 3개 → {total * 3}개 row 생성)")

        for map_row in map_rows:
            map_id = map_row["id"]
            map_dict = dict(map_row)

            # 하위 구역(depth 2) 조회
            sub_areas = await conn.fetch(
                """
                SELECT id, name
                FROM map_group_i18n
                WHERE parent_value = $1 AND depth = 2
                ORDER BY id ASC
            """,
                map_id,
            )

            # 해당 지도 탈출구 조회
            extractions = await conn.fetch(
                """
                SELECT id, name, faction, always_available, single_use,
                       requirements, tip
                FROM extraction_i18n
                WHERE map = $1
                ORDER BY faction ASC, id ASC
            """,
                map_id,
            )

            sub_areas_list = [dict(r) for r in sub_areas]
            extractions_list = [dict(r) for r in extractions]

            log.info(
                f"처리중: {map_id} | 구역 {len(sub_areas_list)}개 | 탈출구 {len(extractions_list)}개"
            )

            await process_map(conn, client, map_dict, sub_areas_list, extractions_list)

        log.info(f"=== 완료: {total}개 지도, {total * 3}개 row 생성/업데이트 ===")

    finally:
        await client.aclose()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
