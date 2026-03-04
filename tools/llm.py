import httpx
import json
import logging
import os
from schemas.models import ChatMessage

log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL")

SYSTEM_PROMPTS = {
    "ko": """당신은 Escape from Tarkov 전문 도우미입니다.

[규칙]
중요: 반드시 한국어로만 답변하세요.
1. [참고 문서]가 제공된 경우, 반드시 해당 내용을 기반으로 답변하세요.
2. [참고 문서]의 내용을 최우선으로 사용하고, 문서에 있는 정보는 절대 "없다"고 하지 마세요.
3. [참고 문서]에 없는 내용은 추측하거나 보완하지 마세요.
4. [참고 문서]가 없거나 관련 정보가 전혀 없을 때만 "해당 정보가 없습니다."라고 답하세요.
5. 답변 마지막에 참고한 문서의 URL을 포함하세요. 형식: '참고: {URL}'
6. 마크다운 형식으로 작성하세요.
   - 항목명은 **굵게** 표시하세요.
   - 여러 값은 bullet 목록으로 정리하세요.
   - 불필요한 설명은 추가하지 마세요.
""",
    "en": """You are an Escape from Tarkov assistant.

[Rules]
IMPORTANT: Respond in English only.
1. When [Reference Documents] are provided, you MUST base your answer on them.
2. If information exists in [Reference Documents], NEVER say it's unavailable.
3. Do not guess or add information not found in [Reference Documents].
4. Only say "That information is not available." when no relevant documents are provided.
5. Always include the source URL at the end. Format: 'Reference: {URL}'
6. Use Markdown format.
   - Bold all field names.
   - Use bullet points for multiple values.
   - Keep responses concise.
""",
    "ja": """あなたはEscape from Tarkovの専門アシスタントです。

[ルール]
重要：日本語のみで回答してください。
1. [参考文書]が提供された場合、必ずその内容に基づいて回答してください。
2. [参考文書]に情報がある場合、「ない」と言わないでください。
3. [参考文書]にない内容は推測しないでください。
4. 関連文書が全くない場合のみ「その情報はありません。」と答えてください。
5. 回答の最後にURLを含めてください。形式：「参考：{URL}」
6. Markdown形式で記述してください。
   - 項目名は**太字**で表示してください。
   - 複数の値はbullet形式で整理してください。
""",
}


def _build_messages(messages: list[ChatMessage], context: str) -> list[dict]:
    """context를 마지막 user 메시지에 직접 주입"""
    msg_list = [{"role": m.role, "content": m.content} for m in messages]
    if context and msg_list and msg_list[-1]["role"] == "user":
        msg_list[-1][
            "content"
        ] = f"[참고 문서]\n{context}\n\n질문: {msg_list[-1]['content']}"
    return msg_list


async def chat_llm(
    messages: list[ChatMessage],
    context: str = "",
    lang: str = "ko",
) -> str:
    system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ko"])
    msg_list = _build_messages(messages, context)

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            *msg_list,
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
    msg_list = _build_messages(messages, context)

    log.info(f"[llm] context 길이: {len(context)}")
    log.info(f"[llm] context 내용:\n{context[:500]}")  # 앞 500자만
    log.info(f"[llm] 최종 메시지:\n{msg_list[-1]['content'][:500]}")

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            *msg_list,
        ],
        "stream": True,
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
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    log.info(
                        f"[llm_stream] model={CHAT_MODEL} tokens={data.get('eval_count', '?')}"
                    )
                    break
