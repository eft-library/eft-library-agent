import httpx
import logging
import os
from schemas.models import ChatMessage

log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL")

SYSTEM_PROMPTS = {
    "ko": """당신은 Escape from Tarkov 게임 전문 도우미입니다.
반드시 주어진 참고 문서에 있는 내용만 답변하세요.
참고 문서에 없는 내용은 절대 추측하거나 일반적인 정보를 제공하지 마세요.
문서에 없는 내용은 "해당 정보는 제공된 문서에 없습니다."라고만 답변하세요.""",
    "en": """You are an expert assistant for the game Escape from Tarkov.
You MUST only answer based on the provided reference documents.
NEVER guess, infer, or provide general knowledge not found in the documents.
If the information is not in the documents, only say "That information is not available in the provided documents."
IMPORTANT: You MUST respond in English only. Do not use any other language.""",
    "ja": """あなたはEscape from Tarkovゲームの専門アシスタントです。
必ず提供された参考文書に基づいてのみ回答してください。
文書にない内容は絶対に推測したり、一般的な情報を提供したりしないでください。
文書にない場合は「その情報は提供された文書にありません。」とだけ答えてください。
重要：必ず日本語のみで回答してください。他の言語を使用しないでください。""",
}


async def chat_llm(
    messages: list[ChatMessage],
    context: str = "",
    lang: str = "ko",
) -> str:
    system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ko"])
    if context:
        system += f"\n\n[참고 문서]\n{context}"

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            *[{"role": m.role, "content": m.content} for m in messages],
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
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
    if context:
        system += f"\n\n[참고 문서]\n{context}"

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            *[{"role": m.role, "content": m.content} for m in messages],
        ],
        "stream": True,  # ← 스트리밍 켜기
        "options": {
            "temperature": 0.3,
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
