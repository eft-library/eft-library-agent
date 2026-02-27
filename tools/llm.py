import httpx
import logging
import os
from schemas.models import ChatMessage

log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL")

SYSTEM_PROMPTS = {
    "ko": """당신은 Escape from Tarkov 데이터베이스 검색 도구입니다.

[참고 문서 구조 안내]
- [이 아이템을 만드는 방법 (제작법)] : 해당 아이템의 제작 레시피 (결과물, 제작 시간, 필요 재료)
- [이 아이템이 재료로 사용되는 제작] : 이 아이템을 재료로 쓰는 다른 아이템의 제작법
- [제작 레시피] : 은신처 레벨별 제작 가능 목록
- [스펙] : 아이템 기본 스탯 정보
- [퀘스트 보상] : 이 아이템을 보상으로 주는 퀘스트
- [퀘스트 제출 아이템] : 이 아이템을 제출해야 하는 퀘스트
- [상인 교환] : NPC 교환 정보
- [은신처 건설 필요] : 은신처 건설 시 필요한 아이템
- [건설 필요 아이템] : 은신처 레벨 업그레이드에 필요한 재료
- [레벨업 보너스] : 은신처 레벨 달성 시 얻는 보너스
- [목표] : 퀘스트 완료 조건
- [보상] : 퀘스트 완료 보상

[규칙]
1. 반드시 [참고 문서]에 명시된 내용만 답변하세요.
2. [참고 문서]에 없는 내용은 절대 언급하지 마세요.
3. 추측, 보완, 일반 지식, 게임 경험 기반 답변은 절대 금지입니다.
4. [참고 문서]에 없는 질문은 "제공된 문서에 해당 정보가 없습니다."라고만 답하세요.
5. 위 규칙을 어기는 것은 오답입니다.""",
    "en": """You are a database search tool for Escape from Tarkov.

[Document Structure Guide]
- [How to craft this item] : Craft recipe for this item (output, duration, required materials)
- [Used as material in] : Other items that use this item as a crafting material
- [Craft Recipe] : List of craftable items by hideout level
- [Spec] : Basic item stats
- [Quest Reward] : Quests that reward this item
- [Quest Item] : Quests that require submitting this item
- [Barter (NPC)] : NPC barter trade information
- [Required for Hideout] : Items needed for hideout construction
- [Required Items] : Materials needed for hideout level upgrade
- [Level Bonuses] : Bonuses gained upon reaching hideout level
- [Objectives] : Quest completion conditions
- [Rewards] : Quest completion rewards

[Rules]
1. You MUST only use information explicitly stated in the [Reference Documents].
2. NEVER mention anything not found in the [Reference Documents].
3. Guessing, inferring, adding general knowledge, or using game experience is STRICTLY FORBIDDEN.
4. If the answer is not in the [Reference Documents], respond ONLY with: "That information is not available in the provided documents."
5. Violating these rules is an incorrect answer.
IMPORTANT: You MUST respond in English only. Do not use any other language.""",
    "ja": """あなたはEscape from Tarkovのデータベース検索ツールです。

[参考文書の構造案内]
- [このアイテムの製作方法] : このアイテムのクラフトレシピ（結果物、製作時間、必要素材）
- [このアイテムが素材として使われるクラフト] : このアイテムを素材として使う他のアイテムの製作法
- [クラフトレシピ] : 隠れ家レベル別の製作可能リスト
- [スペック] : アイテムの基本ステータス情報
- [クエスト報酬] : このアイテムを報酬として与えるクエスト
- [クエストアイテム] : このアイテムを提出する必要があるクエスト
- [商人交換] : NPC交換情報
- [隠れ家建設に必要] : 隠れ家建設時に必要なアイテム
- [建設必要アイテム] : 隠れ家レベルアップグレードに必要な素材
- [レベルアップボーナス] : 隠れ家レベル達成時に得られるボーナス
- [目標] : クエスト完了条件
- [報酬] : クエスト完了報酬

[ルール]
1. 必ず[参考文書]に明示された内容のみを回答してください。
2. [参考文書]にない内容は絶対に言及しないでください。
3. 推測、補完、一般知識、ゲーム経験に基づく回答は厳禁です。
4. [参考文書]にない質問には「その情報は提供された文書にありません。」とだけ答えてください。
5. 上記ルールを破ることは誤答です。
重要：必ず日本語のみで回答してください。他の言語を使用しないでください。""",
}


async def chat_llm(
    messages: list[ChatMessage],
    context: str = "",
    lang: str = "ko",
) -> str:
    system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ko"])

    msg_list = [{"role": m.role, "content": m.content} for m in messages]
    if context and msg_list:
        last = msg_list[-1]
        if last["role"] == "user":
            last["content"] = f"[참고 문서]\n{context}\n\n질문: {last['content']}"

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            *[{"role": m.role, "content": m.content} for m in messages],
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        result = resp.json()

    answer = result["message"]["content"]
    log.info(f"[llm] model={CHAT_MODEL} tokens={result.get('eval_count', '?')}")
    return answer


async def chat_llm_stream(
    messages: list[ChatMessage],
    context: str = "",
    lang: str = "ko",
):
    system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ko"])
    msg_list = [{"role": m.role, "content": m.content} for m in messages]
    if context and msg_list:
        last = msg_list[-1]
        if last["role"] == "user":
            last["content"] = f"[참고 문서]\n{context}\n\n질문: {last['content']}"

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            *[{"role": m.role, "content": m.content} for m in messages],
        ],
        "stream": True,  # ← 스트리밍 켜기
        "options": {
            "temperature": 0.1,
            "num_ctx": 8192,
        },
    }

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                import json

                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    log.info(
                        f"[llm] model={CHAT_MODEL} tokens={data.get('eval_count', '?')}"
                    )
                    break
