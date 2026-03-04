"""
DYNAMIC_INFO_I18N + INFORMATION_I18N 배치 임베딩 스크립트
- DYNAMIC_INFO_I18N에서 event / patch 타입 추출
- link에서 id 파싱
  event: /event/detail/{id}
  patch: /patch-notes/detail/{id}
- INFORMATION_I18N에서 해당 id 조회 후 조인
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

DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")
LANGS = ["ko", "en", "ja"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# 유틸
def clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for img in soup.find_all("img"):
        img.decompose()

    # 인라인 태그 먼저 unwrap (텍스트 유지, 태그만 제거)
    for tag in soup.find_all(["a", "b", "strong", "em", "i", "span"]):
        tag.unwrap()

    # unwrap 후 다시 파싱 (변경사항 반영)
    soup = BeautifulSoup(str(soup), "html.parser")

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


# ID 추출
# { type_key: (link_prefix, db_type) }
TYPE_CONFIG = {
    "event": {
        "link_prefix": "/event/detail/",
        "db_type": "EVENT",
        "url_prefix": "https://eftlibrary.com/event/detail/",
    },
    "patch": {
        "link_prefix": "/patch-notes/detail/",
        "db_type": "PATCH-NOTES",
        "url_prefix": "https://eftlibrary.com/patch-notes/detail/",
    },
}


def extract_ids_by_type(json_value: dict) -> dict[str, set[str]]:
    """
    json_value에서 event/patch link 파싱
    반환: { "event": {"event24", ...}, "patch": {"patch22", ...} }
    """
    result = {t: set() for t in TYPE_CONFIG}
    for type_key, cfg in TYPE_CONFIG.items():
        items = json_value.get(type_key, [])
        for item in items:
            link = item.get("link", "")
            if cfg["link_prefix"] in link:
                item_id = link.split(cfg["link_prefix"])[-1].strip("/")
                if item_id:
                    result[type_key].add(item_id)
    return result


# content 빌더
LANG_LABELS = {
    "ko": {
        "event": "이벤트",
        "patch": "패치 노트",
        "updated": "업데이트",
        "content": "내용",
    },
    "en": {
        "event": "Event",
        "patch": "Patch Note",
        "updated": "Updated",
        "content": "Content",
    },
    "ja": {
        "event": "イベント",
        "patch": "パッチノート",
        "updated": "更新",
        "content": "内容",
    },
}

SEARCH_KEYWORDS = {
    "ko": {
        "event": "이벤트 진행중 현재 이벤트 최신 이벤트",
        "patch": "패치 노트 최신 패치 업데이트 변경사항",
    },
    "en": {
        "event": "event current active latest event",
        "patch": "patch note latest patch update changes",
    },
    "ja": {
        "event": "イベント 現在 最新イベント 開催中",
        "patch": "パッチノート 最新パッチ アップデート 変更点",
    },
}


def build_content(info_row: dict, type_key: str, lang: str) -> str:
    label = LANG_LABELS[lang]
    type_label = label[type_key]
    name = get_lang_value(info_row["name"], lang)
    description = clean_html(get_lang_value(info_row["description"], lang))
    update_time = info_row["update_time"]
    updated_str = update_time.strftime("%Y-%m-%d") if update_time else ""

    keywords = SEARCH_KEYWORDS[lang][type_key]

    parts = [
        f"{keywords} {name}",  # 검색 키워드
        f"{type_label}: {name}",
        f"{label['updated']}: {updated_str}",
    ]
    if description:
        parts.append(f"\n[{label['content']}]\n{description}")

    return "\n".join(parts).strip()


# 임베딩 + upsert
async def get_embedding(client: httpx.AsyncClient, text: str) -> list[float]:
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


async def upsert_rag_document(conn, source_id, lang, content, embedding, metadata):
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
        "information_i18n",
        source_id,
        lang,
        content,
        embedding_str,
        json.dumps(metadata, ensure_ascii=False),
    )


# 타입별 처리
async def process_type(
    conn: asyncpg.Connection,
    client: httpx.AsyncClient,
    type_key: str,
    ids: set[str],
):
    cfg = TYPE_CONFIG[type_key]
    if not ids:
        log.info(f"[{type_key}] id 없음, 스킵")
        return

    log.info(f"[{type_key}] id 목록: {sorted(ids)}")

    rows = await conn.fetch(
        """
        SELECT id, type, name, description, update_time
        FROM information_i18n
        WHERE id = ANY($1) AND type = $2
        ORDER BY update_time DESC
    """,
        list(ids),
        cfg["db_type"],
    )

    log.info(f"[{type_key}] 조회된 항목: {len(rows)}개")

    for row in rows:
        info = dict(row)
        item_id = info["id"]

        for lang in LANGS:
            content = build_content(info, type_key, lang)
            if not content.strip():
                log.warning(f"  ⚠ 빈 content 스킵: {item_id} [{lang}]")
                continue

            try:
                embedding = await get_embedding(client, content)
                metadata = {
                    "content_type": type_key,
                    "source_tables": ["dynamic_info_i18n", "information_i18n"],
                    "item_id": item_id,
                    "item_name": {
                        "ko": get_lang_value(info["name"], "ko"),
                        "en": get_lang_value(info["name"], "en"),
                        "ja": get_lang_value(info["name"], "ja"),
                    },
                    "url": f"{cfg['url_prefix']}{item_id}",
                }
                await upsert_rag_document(
                    conn, item_id, lang, content, embedding, metadata
                )
                log.info(f"  ✓ {item_id} [{lang}] 완료")

            except httpx.HTTPError as e:
                log.error(f"  ✗ 임베딩 실패: {item_id} [{lang}] - {e}")
            except asyncpg.PostgresError as e:
                log.error(f"  ✗ DB 저장 실패: {item_id} [{lang}] - {e}")


# 메인
async def main():
    log.info("=== dynamic_info event/patch + information 배치 임베딩 시작 ===")

    conn = await asyncpg.connect(DATABASE_URL)
    client = httpx.AsyncClient()

    try:
        # 1. DYNAMIC_INFO_I18N 전체 조회
        dynamic_rows = await conn.fetch("SELECT id, json_value FROM dynamic_info_i18n")

        # 2. event/patch id 수집
        all_ids: dict[str, set[str]] = {t: set() for t in TYPE_CONFIG}
        for row in dynamic_rows:
            json_value = parse_jsonb(row["json_value"]) or {}
            extracted = extract_ids_by_type(json_value)
            for type_key, ids in extracted.items():
                all_ids[type_key].update(ids)

        # 3. 타입별 처리
        for type_key, ids in all_ids.items():
            await process_type(conn, client, type_key, ids)

        log.info("=== 완료 ===")

    finally:
        await client.aclose()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
