"""
story_i18n 배치 임베딩 스크립트
- story_i18n 테이블 전체 데이터를 읽어서
- HTML 파싱 후 텍스트 조합
- bge-m3로 임베딩 생성
- rag_documents 테이블에 upsert

청크 분리:
  - story_id                : 스토리명만 (chunk_type: identifier) → RDB 조회용
  - story_id_objectives     : 목표 (chunk_type: content)
  - story_id_requirements   : 요구사항 (chunk_type: content)
  - story_id_guide          : 가이드 (chunk_type: content)
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
BATCH_SIZE = 10
LANGS = ["ko", "en", "ja"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# HTML 파싱
SKIP_HEADERS = {"아이콘", "icon", "アイコン"}


def clean_html(html_text: str) -> str:
    """HTML 태그 제거, img 제거, table은 구조화된 텍스트로 변환, 아이콘 컬럼 제거"""
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
    """JSONB {ko: '', en: '', ja: ''} 에서 언어별 값 추출 (str/dict 모두 처리)"""
    if not jsonb_field:
        return ""
    if isinstance(jsonb_field, str):
        try:
            jsonb_field = json.loads(jsonb_field)
        except json.JSONDecodeError:
            return ""
    return jsonb_field.get(lang, "") or ""


# 라벨
LABELS = {
    "ko": {
        "story": "스토리",
        "objectives": "목표",
        "requirements": "요구사항",
        "guide": "가이드",
    },
    "en": {
        "story": "Story",
        "objectives": "Objectives",
        "requirements": "Requirements",
        "guide": "Guide",
    },
    "ja": {
        "story": "ストーリー",
        "objectives": "目標",
        "requirements": "必要条件",
        "guide": "ガイド",
    },
}


# content 조합 - 식별용 (chunk_type: identifier)
def build_main_content(row: dict, lang: str) -> str:
    lb = LABELS[lang]
    name = get_lang_value(row["name"], lang)
    return f"{lb['story']}: {name}"


# content 조합 - 목표
def build_objectives_content(row: dict, lang: str) -> str:
    lb = LABELS[lang]
    name = get_lang_value(row["name"], lang)
    objectives = clean_html(get_lang_value(row.get("objectives"), lang))

    if not objectives:
        return ""

    parts = [
        f"{lb['story']}: {name}",
        f"\n[{lb['objectives']}]\n{objectives}",
    ]

    return "\n".join(parts).strip()


# content 조합 - 요구사항
def build_requirements_content(row: dict, lang: str) -> str:
    lb = LABELS[lang]
    name = get_lang_value(row["name"], lang)
    requirements = clean_html(get_lang_value(row.get("requirements"), lang))

    if not requirements:
        return ""

    parts = [
        f"{lb['story']}: {name}",
        f"\n[{lb['requirements']}]\n{requirements}",
    ]

    return "\n".join(parts).strip()


# content 조합 - 가이드
def build_guide_content(row: dict, lang: str) -> str:
    lb = LABELS[lang]
    name = get_lang_value(row["name"], lang)
    guide = clean_html(get_lang_value(row.get("guide"), lang))

    if not guide:
        return ""

    parts = [
        f"{lb['story']}: {name}",
        f"\n[{lb['guide']}]\n{guide}",
    ]

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
        "story_i18n",
        source_id,
        lang,
        content,
        embedding_str,
        chunk_type,
        ref_type,
        ref_id,
        json.dumps(metadata, ensure_ascii=False),
    )


# 메인 배치 처리
async def process_batch(
    conn: asyncpg.Connection, client: httpx.AsyncClient, rows: list[dict]
):
    for row in rows:
        story_id = row["id"]

        base_metadata = {
            "story_id": story_id,
            "story_name": {
                "ko": get_lang_value(row["name"], "ko"),
                "en": get_lang_value(row["name"], "en"),
                "ja": get_lang_value(row["name"], "ja"),
            },
            "url": f"https://eftlibrary.com/story/{story_id}",
        }

        docs = [
            {
                "source_id": story_id,
                "chunk_type": "identifier",  # 이름만 → RDB 조회용
                "build_fn": lambda lang, r=row: build_main_content(r, lang),
                "skip": False,
            },
            {
                "source_id": f"{story_id}_objectives",
                "chunk_type": "content",
                "build_fn": lambda lang, r=row: build_objectives_content(r, lang),
                "skip": not row.get("objectives"),
            },
            {
                "source_id": f"{story_id}_requirements",
                "chunk_type": "content",
                "build_fn": lambda lang, r=row: build_requirements_content(r, lang),
                "skip": not row.get("requirements"),
            },
            {
                "source_id": f"{story_id}_guide",
                "chunk_type": "content",
                "build_fn": lambda lang, r=row: build_guide_content(r, lang),
                "skip": not row.get("guide"),
            },
        ]

        for doc in docs:
            if doc["skip"]:
                continue

            for lang in LANGS:
                content = doc["build_fn"](lang)

                if not content.strip():
                    log.warning(f"  ⚠ 빈 content 스킵: {doc['source_id']} [{lang}]")
                    continue

                try:
                    embedding = await get_embedding(client, content)

                    await upsert_rag_document(
                        conn,
                        doc["source_id"],
                        lang,
                        content,
                        embedding,
                        doc["chunk_type"],  # identifier or content
                        "story",
                        story_id,  # ref_id는 항상 story_id로 통일
                        base_metadata,
                    )
                    log.info(f"  ✓ {doc['source_id']} [{lang}] 완료")

                except httpx.HTTPError as e:
                    log.error(f"  ✗ 임베딩 실패: {doc['source_id']} [{lang}] - {e}")
                except asyncpg.PostgresError as e:
                    log.error(f"  ✗ DB 저장 실패: {doc['source_id']} [{lang}] - {e}")


async def main():
    log.info("=== story_i18n 배치 임베딩 시작 ===")
    log.info(f"모델: {EMBED_MODEL} | 배치 크기: {BATCH_SIZE} | 언어: {LANGS}")

    conn = await asyncpg.connect(DATABASE_URL)
    client = httpx.AsyncClient()

    try:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM story_i18n WHERE id != 'roadmap'"
        )
        log.info(f"총 {total}개 story 처리 예정")

        offset = 0
        processed = 0

        while offset < total:
            rows = await conn.fetch(
                """
                SELECT id, name, objectives, requirements, guide
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

        log.info(f"=== 완료: {processed}개 story 처리 ===")

    finally:
        await client.aclose()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
