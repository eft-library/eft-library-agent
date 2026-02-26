"""
story_i18n 배치 임베딩 스크립트
- story_i18n 테이블 전체 데이터를 읽어서
- HTML 파싱 후 텍스트 조합
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

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")
BATCH_SIZE = 10
LANGS = ["ko", "en", "ja"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────
# HTML 파싱
# ─────────────────────────────────────────
# 테이블에서 제거할 컬럼 헤더 (아이콘 등 이미지 컬럼)
SKIP_HEADERS = {"아이콘", "icon", "アイコン"}


def clean_html(html_text: str) -> str:
    """HTML 태그 제거, img 제거, table은 구조화된 텍스트로 변환, 아이콘 컬럼 제거"""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")

    # img 태그 제거
    for img in soup.find_all("img"):
        img.decompose()

    # table을 구조화된 텍스트로 변환
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        headers = []
        skip_indices = set()
        lines = []

        for i, row in enumerate(rows):
            cols = row.find_all(["th", "td"])
            values = [c.get_text(strip=True) for c in cols]

            if i == 0:
                # 아이콘 컬럼 인덱스 기록 후 헤더에서 제거
                skip_indices = {j for j, h in enumerate(values) if h in SKIP_HEADERS}
                headers = [h for j, h in enumerate(values) if j not in skip_indices]
            else:
                # 아이콘 컬럼 값 제거
                values = [v for j, v in enumerate(values) if j not in skip_indices]
                if headers:
                    line = " | ".join(f"{h}: {v}" for h, v in zip(headers, values))
                else:
                    line = " | ".join(values)
                lines.append(line)

        table.replace_with(soup.new_string("\n" + "\n".join(lines) + "\n"))

    return soup.get_text(separator="\n", strip=True)


def get_lang_value(jsonb_field: dict | str | None, lang: str) -> str:
    """JSONB {ko: '', en: '', ja: ''} 에서 언어별 값 추출 (str/dict 모두 처리)"""
    if not jsonb_field:
        return ""
    # asyncpg가 JSONB를 str로 줄 때 처리
    if isinstance(jsonb_field, str):
        try:
            jsonb_field = json.loads(jsonb_field)
        except json.JSONDecodeError:
            return ""
    return jsonb_field.get(lang, "") or ""


# ─────────────────────────────────────────
# content 조합
# ─────────────────────────────────────────
def build_content(row: dict, lang: str) -> str:
    """언어별 임베딩용 텍스트 조합"""
    name = get_lang_value(row["name"], lang)
    objectives = clean_html(get_lang_value(row["objectives"], lang))
    requirements = clean_html(get_lang_value(row["requirements"], lang))
    guide = clean_html(get_lang_value(row["guide"], lang))
    order = row["order"] or ""

    label = {
        "ko": {
            "story": "스토리",
            "order": "순서",
            "objectives": "목표",
            "requirements": "요구사항",
            "guide": "가이드",
        },
        "en": {
            "story": "Story",
            "order": "Order",
            "objectives": "Objectives",
            "requirements": "Requirements",
            "guide": "Guide",
        },
        "ja": {
            "story": "ストーリー",
            "order": "順序",
            "objectives": "目標",
            "requirements": "必要条件",
            "guide": "ガイド",
        },
    }[lang]

    parts = [f"{label['story']}: {name}"]
    if order:
        parts.append(f"{label['order']}: {order}")
    if objectives:
        parts.append(f"\n[{label['objectives']}]\n{objectives}")
    if requirements:
        parts.append(f"\n[{label['requirements']}]\n{requirements}")
    if guide:
        parts.append(f"\n[{label['guide']}]\n{guide}")

    return "\n".join(parts).strip()


# ─────────────────────────────────────────
# 임베딩 생성 (Ollama bge-m3)
# ─────────────────────────────────────────
async def get_embedding(client: httpx.AsyncClient, text: str) -> list[float]:
    """Ollama bge-m3로 임베딩 벡터 생성"""
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["embeddings"][0]


# ─────────────────────────────────────────
# DB upsert
# ─────────────────────────────────────────
async def upsert_rag_document(
    conn: asyncpg.Connection,
    source_id: str,
    lang: str,
    content: str,
    embedding: list[float],
    metadata: dict,
):
    """rag_documents upsert (중복이면 업데이트)"""
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
        "story_i18n",
        source_id,
        lang,
        content,
        embedding_str,
        json.dumps(metadata, ensure_ascii=False),
    )


# ─────────────────────────────────────────
# 메인 배치 처리
# ─────────────────────────────────────────
async def process_batch(
    conn: asyncpg.Connection, client: httpx.AsyncClient, rows: list[dict]
):
    """배치 단위로 임베딩 생성 및 upsert"""
    for row in rows:
        story_id = row["id"]
        story_name_ko = get_lang_value(row["name"], "ko") or story_id

        for lang in LANGS:
            content = build_content(row, lang)

            # 빈 content 스킵 (해당 언어 데이터 없는 경우)
            if not content.strip():
                log.warning(f"  ⚠ 빈 content 스킵: {story_id} [{lang}]")
                continue

            try:
                embedding = await get_embedding(client, content)

                metadata = {
                    "content_type": "single",
                    "source_tables": ["story_i18n"],
                    "story_id": story_id,
                    "story_name": story_name_ko,
                    "order": row["order"],
                    "url": f"https://eftlibrary.com/story/{story_id}",
                }

                # ── 확인용 출력 ──────────────────────────────────────
                # print(f"\n{'='*60}")
                # print(f"[{story_id}] [{lang}]")
                # print(f"{'─'*60}")
                # print(f"[content]\n{content}")
                # print(f"{'─'*60}")
                # print(f"[metadata]\n{json.dumps(metadata, ensure_ascii=False, indent=2)}")
                # print(f"{'─'*60}")
                # print(f"[embedding] 차원: {len(embedding)} | 앞 5개: {embedding[:5]}")
                # print(f"{'='*60}")
                # ────────────────────────────────────────────────────

                await upsert_rag_document(
                    conn, story_id, lang, content, embedding, metadata
                )
                log.info(f"  ✓ {story_id} [{lang}] 완료")

            except httpx.HTTPError as e:
                log.error(f"  ✗ 임베딩 실패: {story_id} [{lang}] - {e}")
            except asyncpg.PostgresError as e:
                log.error(f"  ✗ DB 저장 실패: {story_id} [{lang}] - {e}")


async def main():
    log.info("=== story_i18n 배치 임베딩 시작 ===")
    log.info(f"모델: {EMBED_MODEL} | 배치 크기: {BATCH_SIZE} | 언어: {LANGS}")

    conn = await asyncpg.connect(DATABASE_URL)
    client = httpx.AsyncClient()

    try:
        total = await conn.fetchval("SELECT COUNT(*) FROM story_i18n")
        log.info(f"총 {total}개 story 처리 예정 (언어 3개 → {total * 3}개 row 생성)")

        offset = 0
        processed = 0

        while offset < total:
            rows = await conn.fetch(
                """
                SELECT id, name, objectives, requirements, guide, "order"
                FROM story_i18n
                WHERE id != 'roadmap'
                ORDER BY "order" ASC NULLS LAST, id ASC
                LIMIT $1 OFFSET $2
            """,
                BATCH_SIZE,
                offset,
            )

            if not rows:
                break

            rows_dict = [dict(r) for r in rows]
            log.info(f"배치 처리중: {offset + 1} ~ {offset + len(rows_dict)} / {total}")

            await process_batch(conn, client, rows_dict)

            processed += len(rows_dict)
            offset += BATCH_SIZE

        log.info(
            f"=== 완료: {processed}개 story, {processed * 3}개 row 생성/업데이트 ==="
        )

    finally:
        await client.aclose()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
