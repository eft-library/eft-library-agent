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

1. [현재 참고 문서]가 제공된 경우, 반드시 해당 내용을 최우선으로 사용하세요.
2. [현재 참고 문서]에 있는 정보는 절대 "없다"고 하지 마세요.
3. [현재 참고 문서]에 없는 내용은 Escape from Tarkov 전문 지식을 바탕으로 답변하세요.
4. 추천, 가이드, 팁 등 일반적인 질문은 전문 지식으로 자유롭게 답변하세요.
5. [현재 참고 문서]에 명시된 URL만 마지막에 포함하세요. URL을 직접 만들거나 추측하지 마세요. 형식: '참고: {URL}'
6. 마크다운 형식으로 작성하세요.
   - 항목명은 **굵게** 표시하세요.
   - 여러 값은 bullet 목록으로 정리하세요.
   - 불필요한 설명은 추가하지 마세요.
7. [현재 참고 문서]는 현재 질문에만 해당하는 문서입니다.
   이전 대화의 내용을 사실 정보로 참조하지 마세요.
8. 이전 대화는 '그 맵', '그 보스' 같은 지시어 해석에만 활용하세요.
""",
    "en": """You are an Escape from Tarkov expert assistant.

[Rules]
IMPORTANT: Respond in English only.

1. When [Current Reference Documents] are provided, prioritize their content above all else.
2. If information exists in [Current Reference Documents], NEVER say it's unavailable.
3. For content not in [Current Reference Documents], answer using your Escape from Tarkov expertise.
4. For recommendations, guides, and tips, answer freely using your expert knowledge.
5. Only include URLs explicitly stated in [Current Reference Documents]. Never fabricate or guess URLs. Format: 'Reference: {URL}'
6. Use Markdown format.
   - Bold all field names.
   - Use bullet points for multiple values.
   - Keep responses concise.
7. [Current Reference Documents] are only relevant to the current question.
   Do not use previous conversation content as factual reference.
8. Use previous conversation only to interpret references like 'that map' or 'that boss'.
""",
    "ja": """あなたはEscape from Tarkovの専門アシスタントです。

[ルール]
重要：日本語のみで回答してください。

1. [現在の参考文書]が提供された場合、その内容を最優先で使用してください。
2. [現在の参考文書]に情報がある場合、「ない」と言わないでください。
3. [現在の参考文書]にない内容はEscape from Tarkovの専門知識で回答してください。
4. おすすめ・ガイド・ヒントなどの一般的な質問は専門知識で自由に答えてください。
5. [現在の参考文書]に明記されたURLのみ最後に含めてください。URLを推測・作成しないでください。形式：「参考：{URL}」
6. Markdown形式で記述してください。
   - 項目名は**太字**で表示してください。
   - 複数の値はbullet形式で整理してください。
   - 不要な説明は省いてください。
7. [現在の参考文書]は現在の質問にのみ該当する文書です。
   以前の会話内容を事実情報として参照しないでください。
8. 以前の会話は「そのマップ」「そのボス」などの指示語の解釈にのみ活用してください。
""",
}


# 이전 응답 잘못되었는데도 과하게 영향을 받아서 수정
def _build_messages(messages: list[ChatMessage], context: str) -> list[dict]:
    msg_list = []

    for i, m in enumerate(messages):
        is_last = i == len(messages) - 1

        if m.role == "user" and is_last:
            # 마지막 user 메시지에만 현재 RAG 주입
            content = m.content
            if context:
                content = f"[현재 참고 문서]\n{context}\n\n질문: {content}"
            msg_list.append({"role": "user", "content": content})

        elif m.role == "assistant":
            # 이전 assistant 답변에서 참고 문서 내용 제거
            # 문맥 파악용으로만 유지 (간략하게)
            msg_list.append({"role": "assistant", "content": m.content})

        else:
            # 이전 user 메시지는 참고 문서 없이 질문만
            msg_list.append({"role": "user", "content": m.content})

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

    # RAG 결과 없으면 바로 반환
    if not context:
        no_result_msg = {
            "ko": "관련 정보를 찾을 수 없습니다. 다른 검색어로 시도해보세요.",
            "en": "No relevant information found. Please try a different search term.",
            "ja": "関連情報が見つかりませんでした。別の検索ワードをお試しください。",
        }
        yield no_result_msg.get(lang, no_result_msg["ko"])
        return

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
