"""
quest_i18n + npc_i18n 배치 임베딩 스크립트
- quest.npc_id = npc.id JOIN (order IS NOT NULL인 npc만)
- objectives, finish_rewards, guide HTML 파싱
- bge-m3로 임베딩 생성
- rag_documents 테이블에 upsert

청크 분리:
  - quest_id       : 퀘스트명 + 상인명 + 목표 (chunk_type: identifier) → 맵/위치 검색용
  - quest_id_main  : 기본정보 + 목표 + 보상 (chunk_type: content)
  - quest_id_guide : 가이드 (chunk_type: content)
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
BATCH_SIZE = 50
LANGS = ["ko", "en", "ja"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ── 유틸 ───────


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


def clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for img in soup.find_all("img"):
        img.decompose()
    return soup.get_text(separator="\n", strip=True)


def yn(value: bool | None, lang: str) -> str:
    return "✅" if value else "❌"


# ── 라벨 ───────

LABELS = {
    "ko": {
        "quest": "퀘스트",
        "trader": "상인",
        "min_level": "최소 레벨",
        "kappa": "카파",
        "lightkeeper": "라이트키퍼",
        "prev_quest": "선행 퀘스트",
        "next_quest": "후행 퀘스트",
        "objectives": "목표",
        "rewards": "보상",
        "guide": "가이드",
        "standing": "평판",
    },
    "en": {
        "quest": "Quest",
        "trader": "Trader",
        "min_level": "Min Level",
        "kappa": "Kappa",
        "lightkeeper": "Lightkeeper",
        "prev_quest": "Required Quests",
        "next_quest": "Next Quests",
        "objectives": "Objectives",
        "rewards": "Rewards",
        "guide": "Guide",
        "standing": "Standing",
    },
    "ja": {
        "quest": "クエスト",
        "trader": "商人",
        "min_level": "最低レベル",
        "kappa": "カッパ",
        "lightkeeper": "ライトキーパー",
        "prev_quest": "前提クエスト",
        "next_quest": "次のクエスト",
        "objectives": "目標",
        "rewards": "報酬",
        "guide": "ガイド",
        "standing": "評判",
    },
}

NAME_KEY = {"ko": "name_ko", "en": "name_en", "ja": "name_ja"}


# ── content 빌더
def build_identifier_content(quest: dict, npc_name: str, lang: str) -> str:
    """
    퀘스트명 + 상인명 + 목표 → 맵/위치 키워드 포함으로 검색 정확도 향상
    레벨, 카파, 라이트키퍼 여부 추가
    """
    lb = LABELS[lang]
    nk = NAME_KEY[lang]

    quest_name = get_lang_value(quest["name"], lang)
    objectives = parse_jsonb(quest.get("objectives")) or []
    min_level = quest.get("min_player_level") or ""
    kappa = yn(quest.get("kappa_required"), lang)
    lightkeeper = yn(quest.get("lightkeeper_required"), lang)

    parts = [
        f"{lb['quest']}: {quest_name}",
        f"{lb['trader']}: {npc_name}",
    ]

    if min_level:
        parts.append(f"{lb['min_level']}: {min_level}")
    parts.append(f"{lb['kappa']}: {kappa}")
    parts.append(f"{lb['lightkeeper']}: {lightkeeper}")

    if objectives:
        lines = []
        for obj in objectives:
            desc_key = f"description_{lang}"
            desc = obj.get(desc_key, "")
            items = obj.get("items") or []
            line = f"- {desc}"
            if items:
                item_names = ", ".join(i.get(nk) or i.get("name_en", "") for i in items)
                line += f" [{item_names}]"
            lines.append(line)
        parts.append(f"\n[{lb['objectives']}]\n" + "\n".join(lines))

    return "\n".join(parts).strip()


# content 조합 - 퀘스트명 + 상인명 (identifier) -- 구버전 제거됨


def build_main_content(quest: dict, npc_name: str, lang: str) -> str:
    """기본정보 + 목표 + 보상"""
    lb = LABELS[lang]
    nk = NAME_KEY[lang]

    quest_name = get_lang_value(quest["name"], lang)
    task_reqs = parse_jsonb(quest.get("task_requirements")) or []
    task_next = parse_jsonb(quest.get("task_next")) or []
    objectives = parse_jsonb(quest.get("objectives")) or []
    rewards_raw = parse_jsonb(quest.get("finish_rewards")) or {}

    parts = [
        f"{lb['quest']}: {quest_name}",
        f"{lb['trader']}: {npc_name}",
    ]

    # 선행 퀘스트
    if task_reqs:
        lines = [
            f"- {t.get('task', {}).get(nk) or t.get('task', {}).get('name_en', '')}"
            for t in task_reqs
        ]
        parts.append(f"\n[{lb['prev_quest']}]\n" + "\n".join(lines))

    # 후행 퀘스트
    if task_next:
        lines = [
            f"- {t.get('task', {}).get(nk) or t.get('task', {}).get('name_en', '')}"
            for t in task_next
        ]
        parts.append(f"\n[{lb['next_quest']}]\n" + "\n".join(lines))

    # 목표
    if objectives:
        lines = []
        for obj in objectives:
            desc_key = f"description_{lang}"
            desc = obj.get(desc_key, "")
            count = obj.get("count")
            items = obj.get("items") or []
            line = f"- {desc}"
            if count and count > 1:
                line += f" ({count}개)" if lang == "ko" else f" (x{count})"
            if items:
                item_names = ", ".join(i.get(nk) or i.get("name_en", "") for i in items)
                line += f" [{item_names}]"
            lines.append(line)
        parts.append(f"\n[{lb['objectives']}]\n" + "\n".join(lines))

    # 보상
    reward_lines = []
    reward_items = rewards_raw.get("items") or []
    for r in reward_items:
        item = r.get("item") or {}
        item_name = item.get(nk) or item.get("name_en", "")
        quantity = r.get("quantity") or r.get("count", "")
        reward_lines.append(f"- {item_name.strip()} x{quantity}")

    trader_standings = rewards_raw.get("traderStanding") or []
    for ts in trader_standings:
        trader = ts.get("trader") or {}
        trader_name = get_lang_value(
            trader.get("name")
            or {
                "ko": trader.get("name_ko", ""),
                "en": trader.get("name_en", ""),
                "ja": trader.get("name_ja", ""),
            },
            lang,
        ) or trader.get("name_en", "")
        standing = ts.get("standing", "")
        reward_lines.append(f"- {trader_name} {lb['standing']} +{standing}")

    if reward_lines:
        parts.append(f"\n[{lb['rewards']}]\n" + "\n".join(reward_lines))

    return "\n".join(parts).strip()


def build_guide_content(quest: dict, npc_name: str, lang: str) -> str:
    """가이드"""
    lb = LABELS[lang]

    quest_name = get_lang_value(quest["name"], lang)
    guide = clean_html(get_lang_value(quest.get("guide"), lang))

    if not guide:
        return ""

    parts = [
        f"{lb['quest']}: {quest_name}",
        f"{lb['trader']}: {npc_name}",
        f"\n[{lb['guide']}]\n{guide}",
    ]

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
        "quest_i18n",
        source_id,
        lang,
        content,
        embedding_str,
        chunk_type,
        ref_type,
        ref_id,
        json.dumps(metadata, ensure_ascii=False),
    )


# ── 배치 처리 ───


async def process_batch(
    conn: asyncpg.Connection, client: httpx.AsyncClient, rows: list[dict], npc_map: dict
):
    for quest in rows:
        quest_id = quest["id"]
        npc_id = quest.get("npc_id") or ""
        npc_names = npc_map.get(npc_id, {"ko": "", "en": "", "ja": ""})

        base_metadata = {
            "quest_id": quest_id,
            "quest_name": {
                "ko": get_lang_value(quest["name"], "ko"),
                "en": get_lang_value(quest["name"], "en"),
                "ja": get_lang_value(quest["name"], "ja"),
            },
            "npc_id": npc_id,
            "npc_name": npc_names,
            "kappa_required": quest.get("kappa_required") or False,
            "lightkeeper_required": quest.get("lightkeeper_required") or False,
            "min_player_level": quest.get("min_player_level"),
            "url": f"https://eftlibrary.com/quest/detail/{quest.get('url_mapping') or quest_id}",
        }

        docs = [
            {
                "source_id": quest_id,
                "chunk_type": "identifier",
                "build_fn": lambda lang, q=quest, n=npc_names: build_identifier_content(
                    q, n.get(lang, ""), lang
                ),
                "skip": False,
            },
            {
                "source_id": f"{quest_id}_main",
                "chunk_type": "content",
                "build_fn": lambda lang, q=quest, n=npc_names: build_main_content(
                    q, n.get(lang, ""), lang
                ),
                "skip": False,
            },
            {
                "source_id": f"{quest_id}_guide",
                "chunk_type": "content",
                "build_fn": lambda lang, q=quest, n=npc_names: build_guide_content(
                    q, n.get(lang, ""), lang
                ),
                "skip": not quest.get("guide"),
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
                        doc["chunk_type"],
                        "quest",
                        quest_id,  # ref_id는 항상 quest_id로 통일
                        base_metadata,
                    )
                    log.info(f"  ✓ {doc['source_id']} [{lang}] 완료")

                except httpx.HTTPError as e:
                    log.error(f"  ✗ 임베딩 실패: {doc['source_id']} [{lang}] - {e}")
                except asyncpg.PostgresError as e:
                    log.error(f"  ✗ DB 저장 실패: {doc['source_id']} [{lang}] - {e}")


# ── 메인 ───────


async def main():
    log.info("=== quest_i18n 배치 임베딩 시작 ===")
    log.info(f"모델: {EMBED_MODEL} | 배치 크기: {BATCH_SIZE} | 언어: {LANGS}")

    conn = await asyncpg.connect(DATABASE_URL)
    client = httpx.AsyncClient()

    try:
        npc_rows = await conn.fetch("""
            SELECT id, name FROM npc_i18n
            WHERE "order" IS NOT NULL
        """)
        npc_map = {
            r["id"]: {
                "ko": get_lang_value(r["name"], "ko"),
                "en": get_lang_value(r["name"], "en"),
                "ja": get_lang_value(r["name"], "ja"),
            }
            for r in npc_rows
        }
        log.info(f"NPC 로드 완료: {len(npc_map)}개")

        total = await conn.fetchval("SELECT COUNT(*) FROM quest_i18n")
        log.info(f"총 {total}개 퀘스트 처리 예정")

        offset = 0
        processed = 0

        while offset < total:
            rows = await conn.fetch(
                """
                SELECT id, url_mapping, name, npc_id,
                       lightkeeper_required, kappa_required,
                       task_requirements, task_next,
                       objectives, finish_rewards,
                       min_player_level, guide
                FROM quest_i18n
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

            await process_batch(conn, client, rows_dict, npc_map)

            processed += len(rows_dict)
            offset += BATCH_SIZE

        log.info(f"=== 완료: {processed}개 퀘스트 처리 ===")

    finally:
        await client.aclose()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
