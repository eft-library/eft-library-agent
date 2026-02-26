"""
quest_i18n + npc_i18n 배치 임베딩 스크립트
- quest.npc_id = npc.id JOIN (order IS NOT NULL인 npc만)
- objectives, finish_rewards, guide HTML 파싱
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


def clean_html(html_text: str) -> str:
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    for img in soup.find_all("img"):
        img.decompose()
    return soup.get_text(separator="\n", strip=True)


def yn(value: bool | None, lang: str) -> str:
    yes = {"ko": "예", "en": "Yes", "ja": "はい"}
    no = {"ko": "아니오", "en": "No", "ja": "いいえ"}
    return yes[lang] if value else no[lang]


# 라벨
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


# content 조합
def build_content(quest: dict, npc_name: str, lang: str) -> str:
    lb = LABELS[lang]
    nk = NAME_KEY[lang]

    quest_name = get_lang_value(quest["name"], lang)
    min_level = quest.get("min_player_level") or ""
    kappa = yn(quest.get("kappa_required"), lang)
    lightkeeper = yn(quest.get("lightkeeper_required"), lang)

    task_reqs = parse_jsonb(quest.get("task_requirements")) or []
    task_next = parse_jsonb(quest.get("task_next")) or []
    objectives = parse_jsonb(quest.get("objectives")) or []
    rewards_raw = parse_jsonb(quest.get("finish_rewards")) or {}
    guide_html = get_lang_value(quest.get("guide"), lang)
    guide = clean_html(guide_html)

    parts = [
        f"{lb['quest']}: {quest_name}",
        f"{lb['trader']}: {npc_name}",
    ]
    if min_level:
        parts.append(f"{lb['min_level']}: {min_level}")
    parts.append(f"{lb['kappa']}: {kappa} | {lb['lightkeeper']}: {lightkeeper}")

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

    # 가이드
    if guide:
        parts.append(f"\n[{lb['guide']}]\n{guide}")

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
        "quest_i18n",
        source_id,
        lang,
        content,
        embedding_str,
        json.dumps(metadata, ensure_ascii=False),
    )


# 배치 처리
async def process_batch(
    conn: asyncpg.Connection, client: httpx.AsyncClient, rows: list[dict], npc_map: dict
):
    for quest in rows:
        quest_id = quest["id"]
        npc_id = quest.get("npc_id") or ""
        npc_names = npc_map.get(npc_id, {"ko": "", "en": "", "ja": ""})

        for lang in LANGS:
            npc_name = npc_names.get(lang, "")
            content = build_content(quest, npc_name, lang)

            if not content.strip():
                log.warning(f"  ⚠ 빈 content 스킵: {quest_id} [{lang}]")
                continue

            try:
                embedding = await get_embedding(client, content)

                metadata = {
                    "content_type": "joined",
                    "source_tables": ["quest_i18n", "npc_i18n"],
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

                # ── 확인용 출력 ──────────────────────────────────────
                # print(f"\n{'='*60}")
                # print(f"[{quest_id}] [{lang}]")
                # print(f"{'─'*60}")
                # print(f"[content]\n{content}")
                # print(f"{'─'*60}")
                # print(
                #     f"[metadata]\n{json.dumps(metadata, ensure_ascii=False, indent=2)}"
                # )
                # print(f"{'─'*60}")
                # print(f"[embedding] 차원: {len(embedding)} | 앞 5개: {embedding[:5]}")
                # print(f"{'='*60}")
                # ────────────────────────────────────────────────────

                await upsert_rag_document(
                    conn, quest_id, lang, content, embedding, metadata
                )
                log.info(f"  ✓ {quest_id} [{lang}] 완료")

            except httpx.HTTPError as e:
                log.error(f"  ✗ 임베딩 실패: {quest_id} [{lang}] - {e}")
            except asyncpg.PostgresError as e:
                log.error(f"  ✗ DB 저장 실패: {quest_id} [{lang}] - {e}")


# 메인
async def main():
    log.info("=== quest_i18n 배치 임베딩 시작 ===")
    log.info(f"모델: {EMBED_MODEL} | 배치 크기: {BATCH_SIZE} | 언어: {LANGS}")

    conn = await asyncpg.connect(DATABASE_URL)
    client = httpx.AsyncClient()

    try:
        # npc_map 미리 로드 (order IS NOT NULL인 것만)
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
        log.info(f"총 {total}개 퀘스트 처리 예정 (언어 3개 → {total * 3}개 row 생성)")

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

        log.info(
            f"=== 완료: {processed}개 퀘스트, {processed * 3}개 row 생성/업데이트 ==="
        )

    finally:
        await client.aclose()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
