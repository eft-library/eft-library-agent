import json
import logging
import os
import re
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from schemas.models_v3 import Lang
from tools.llm_v3 import CHAT_MODEL, NUM_CTX, OLLAMA_BASE_URL

load_dotenv()

log = logging.getLogger(__name__)

ANSWERABILITY_TIMEOUT = float(os.getenv("RAG_V3_ANSWERABILITY_TIMEOUT", "30"))
ANSWERABILITY_MAX_CONTEXT_CHARS = int(
    os.getenv("RAG_V3_ANSWERABILITY_MAX_CONTEXT_CHARS", "6000")
)


@dataclass(frozen=True)
class AnswerabilityResultV3:
    answerable: bool
    reason: str
    confidence: float | None = None


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _lang_instruction(lang: Lang) -> str:
    if lang == "en":
        return "The user question is in English or should be answered in English."
    if lang == "ja":
        return "The user question is in Japanese or should be answered in Japanese."
    return "The user question is in Korean or should be answered in Korean."


async def assess_answerability_v3(
    query: str,
    context: str,
    lang: Lang = "ko",
) -> AnswerabilityResultV3:
    if not context.strip():
        return AnswerabilityResultV3(False, "empty_context", 1.0)
    if not OLLAMA_BASE_URL or not CHAT_MODEL:
        return AnswerabilityResultV3(True, "answerability_model_not_configured", None)

    trimmed_context = context[:ANSWERABILITY_MAX_CONTEXT_CHARS]
    prompt = f"""You are a retrieval quality judge for an Escape from Tarkov RAG system.

Decide whether the provided reference documents directly contain enough information to answer the user question.

Rules:
- Return JSON only.
- Do not answer the user question.
- Mark answerable=false if documents are merely related by a broad word but lack the specific requested fact.
- Mark answerable=false for real-money purchase, Steam, edition, discount, or external current price questions unless the documents explicitly contain that exact purchase/price information.
- Mark answerable=true if the documents contain the requested relationship, location, craft, reward, spawn, trader, quest, map, boss, or item fact.
- { _lang_instruction(lang) }

JSON schema:
{{"answerable": true|false, "confidence": 0.0-1.0, "reason": "short reason"}}

[Reference Documents]
{trimmed_context}

[User Question]
{query}
"""

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Return strict JSON only. No markdown. No extra text.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": min(NUM_CTX, 8192),
        },
    }

    try:
        async with httpx.AsyncClient(timeout=ANSWERABILITY_TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("[answerability_v3] check failed: %s", exc)
        return AnswerabilityResultV3(True, "answerability_check_failed", None)

    content = data.get("message", {}).get("content", "")
    parsed = _extract_json_object(content)
    if not parsed:
        log.warning("[answerability_v3] invalid response=%s", content[:300])
        return AnswerabilityResultV3(True, "answerability_parse_failed", None)

    answerable = bool(parsed.get("answerable"))
    confidence = parsed.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    reason = str(parsed.get("reason") or "no_reason")

    log.info(
        "[answerability_v3] answerable=%s confidence=%s reason=%s",
        answerable,
        confidence,
        reason[:120],
    )
    return AnswerabilityResultV3(answerable, reason, confidence)
