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

1. [참고 문서]가 제공된 경우, 반드시 해당 내용을 최우선으로 사용하세요.
2. [참고 문서]에 있는 정보는 절대 "없다"고 하지 마세요.
3. [참고 문서]에 없는 내용은 Escape from Tarkov 전문 지식을 바탕으로 답변하세요.
4. 추천, 가이드, 팁 등 일반적인 질문은 전문 지식으로 자유롭게 답변하세요.
5. 답변에 관련 URL이 있으면 마지막에 포함하세요. 형식: '참고: {URL}'
6. 마크다운 형식으로 작성하세요.
   - 항목명은 **굵게** 표시하세요.
   - 여러 값은 bullet 목록으로 정리하세요.
   - 불필요한 설명은 추가하지 마세요.
""",
    "en": """You are an Escape from Tarkov expert assistant.

[Rules]
IMPORTANT: Respond in English only.

1. When [Reference Documents] are provided, prioritize their content above all else.
2. If information exists in [Reference Documents], NEVER say it's unavailable.
3. For content not in [Reference Documents], answer using your Escape from Tarkov expertise.
4. For recommendations, guides, and tips, answer freely using your expert knowledge.
5. Include relevant source URLs at the end if available. Format: 'Reference: {URL}'
6. Use Markdown format.
   - Bold all field names.
   - Use bullet points for multiple values.
   - Keep responses concise.
""",
    "ja": """あなたはEscape from Tarkovの専門アシスタントです。

[ルール]
重要：日本語のみで回答してください。

1. [参考文書]が提供された場合、その内容を最優先で使用してください。
2. [参考文書]に情報がある場合、「ない」と言わないでください。
3. [参考文書]にない内容はEscape from Tarkovの専門知識で回答してください。
4. おすすめ・ガイド・ヒントなどの一般的な質問は専門知識で自由に答えてください。
5. 関連URLがあれば最後に含めてください。形式：「参考：{URL}」
6. Markdown形式で記述してください。
   - 項目名は**太字**で表示してください。
   - 複数の値はbullet形式で整理してください。
   - 不要な説明は省いてください。
""",
}


def _build_messages(messages: list[ChatMessage], context: str) -> list[dict]:
    msg_list = [{"role": m.role, "content": m.content} for m in messages]
    if context and msg_list and msg_list[-1]["role"] == "user":
        msg_list[-1][
            "content"
        ] = f"[참고 문서]\n{context}\n\n질문: {msg_list[-1]['content']}"
    return msg_list


async def chat_llm_stream(
    messages: list[ChatMessage],
    context: str = "",
    lang: str = "ko",
):
    system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ko"])
    msg_list = _build_messages(messages, context)

    # 디버그 로그
    log.info(f"[llm_stream] context 길이: {len(context)}")
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            *msg_list,
        ],
        "stream": True,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": int(os.getenv("NUM_CTX")),
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
