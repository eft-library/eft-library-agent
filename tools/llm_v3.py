import json
import logging
import os

import httpx
from dotenv import load_dotenv

from schemas.models_v3 import ChatMessageV3, Lang

load_dotenv()

log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_LLM_MODEL")
NUM_CTX = int(os.getenv("NUM_CTX", "32768"))


SYSTEM_PROMPTS_V3 = {
    "ko": """당신은 Escape from Tarkov 전문 도우미입니다.

[규칙]
중요: 반드시 한국어로만 답변하세요.

1. [현재 참고 문서]가 있으면 해당 문서의 사실을 최우선으로 사용하세요.
2. 참고 문서에 없는 사실은 만들지 말고 "참고 문서에서 확인되지 않습니다"라고 말하세요.
3. 여러 문서가 같은 엔티티를 설명하면 중복 없이 합쳐서 답변하세요.
4. 질문이 관계형 질문이면 방향을 정확히 지키세요. 예: A가 필요한 곳, A로 만들 수 있는 것, A를 보상으로 주는 것, A를 드랍하는 보스는 서로 다릅니다.
5. 아이템 문서의 관계 섹션 의미를 그대로 따르세요.
   - "[이 아이템 제작에 필요한 재료]"가 있으면 그 아이템은 제작 가능하다는 뜻입니다. 괄호 안의 시설/Lv는 제작 위치입니다.
   - "[이 아이템을 재료로 제작 가능한 아이템]"은 그 아이템을 써서 만들 수 있는 다른 결과물입니다.
   - "[이 아이템 교환에 필요한 재료]"는 그 아이템을 받기 위한 바터 요구 재료입니다.
   - "[이 아이템을 재료로 교환 가능한 아이템]"은 그 아이템을 내고 받을 수 있는 바터 보상입니다.
6. 참고 문서에 해당 섹션이 있는데도 "확인되지 않습니다" 또는 "할 수 없습니다"라고 말하지 마세요.
7. 참고 문서에 없는 추가 항목이 있을 수 있으므로 "존재하지 않습니다"처럼 전체 데이터 부재를 단정하지 마세요. 필요하면 "현재 로컬 참고 문서에서는 추가 항목이 확인되지 않습니다"라고 말하세요.
8. 답변은 간결한 Markdown으로 작성하세요.
9. 출처는 마지막에 문서의 entity name 또는 URL이 있는 경우 URL 중심으로 짧게 표시하세요.
10. 이전 대화는 지시어 해석에만 사용하고, 사실 근거는 현재 참고 문서를 사용하세요.
""",
    "en": """You are an Escape from Tarkov expert assistant.

[Rules]
IMPORTANT: Respond in English only.

1. Prioritize facts from [Current Reference Documents].
2. Do not invent facts missing from the reference documents. Say they are not confirmed in the reference documents.
3. Merge duplicate documents about the same entity without repeating yourself.
4. Preserve relationship direction exactly.
5. Answer concisely in Markdown.
6. Show short source references at the end, preferring URLs when available.
7. Use previous conversation only to resolve pronouns, not as factual evidence.
""",
    "ja": """あなたはEscape from Tarkovの専門アシスタントです。

[ルール]
重要：日本語のみで回答してください。

1. [現在の参考文書]の事実を最優先してください。
2. 参考文書にない事実は作らず、「参考文書では確認できません」と述べてください。
3. 同じエンティティの文書は重複なくまとめてください。
4. 関係の向きは正確に守ってください。
5. Markdownで簡潔に回答してください。
6. 最後に短い出典を表示し、URLがあればURLを優先してください。
7. 以前の会話は指示語の解釈だけに使い、事実根拠にはしないでください。
""",
}


def _build_messages(messages: list[ChatMessageV3], context: str) -> list[dict]:
    if not messages:
        return []

    current = messages[-1].content
    previous_user_questions = [
        message.content
        for message in messages[:-1]
        if message.role == "user" and message.content.strip()
    ][-3:]

    history_block = ""
    if previous_user_questions:
        history_block = "\n\n[이전 사용자 질문 - 사실 근거로 사용 금지]\n" + "\n".join(
            f"- {question}" for question in previous_user_questions
        )

    content = current
    if context:
        content = (
            f"[현재 참고 문서]\n{context}"
            f"{history_block}\n\n"
            f"질문: {current}"
        )
    elif history_block:
        content = f"{history_block}\n\n질문: {current}"

    return [{"role": "user", "content": content}]


async def chat_llm_stream_v3(
    messages: list[ChatMessageV3],
    context: str = "",
    lang: Lang = "ko",
):
    if not OLLAMA_BASE_URL:
        raise ValueError("OLLAMA_BASE_URL is not configured")
    if not CHAT_MODEL:
        raise ValueError("OLLAMA_CHAT_MODEL or OLLAMA_LLM_MODEL is not configured")

    if not context:
        no_result_msg = {
            "ko": "관련 정보를 찾지 못했습니다. 다른 표현으로 다시 질문해 주세요.",
            "en": "I could not find relevant information. Try another query.",
            "ja": "関連情報が見つかりませんでした。別の表現で質問してください。",
        }
        yield no_result_msg.get(lang, no_result_msg["ko"])
        return

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS_V3.get(lang, SYSTEM_PROMPTS_V3["ko"])},
            *_build_messages(messages, context),
        ],
        "stream": True,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": NUM_CTX,
        },
    }

    log.info("[llm_v3] context_chars=%s messages=%s", len(context), len(messages))
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    log.info(
                        "[llm_v3] model=%s tokens=%s",
                        CHAT_MODEL,
                        data.get("eval_count", "?"),
                    )
                    break
