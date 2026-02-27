"""
MCP Server 진입점 - FastMCP 방식
"""

import os
from dotenv import load_dotenv

load_dotenv()

from fastmcp import FastMCP
from schemas.models import ChatMessage
from tools.retriever import search_rag as _search_rag
from tools.llm import chat_llm as _chat_llm
from tools.history import save_message as _save_message, get_history as _get_history
from db.connection import get_pool
from starlette.requests import Request
from starlette.responses import JSONResponse
from services.rag import run_rag_pipeline
from contextlib import asynccontextmanager
import logging.handlers

LOG_DIR = os.getenv("LOG_DIR")
os.makedirs(LOG_DIR, exist_ok=True)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 기존 핸들러 제거
root_logger.handlers.clear()

root_logger.addHandler(logging.StreamHandler())
root_logger.addHandler(
    logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(LOG_DIR, "mcp-server.log"),
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
)

for handler in root_logger.handlers:
    handler.setFormatter(log_formatter)

log = logging.getLogger(__name__)

MCP_HOST = os.getenv("MCP_HOST")
MCP_PORT = int(os.getenv("MCP_PORT"))


# Lifespan - DB 풀 초기화
@asynccontextmanager
async def lifespan(server):
    await get_pool()
    log.info("DB 풀 초기화 완료")
    yield
    from db.connection import close_pool

    await close_pool()
    log.info("DB 풀 종료 완료")


# FastMCP (MCP SSE용)
mcp = FastMCP("eft-library-rag", lifespan=lifespan)


# HTTP 엔드포인트 (FastAPI에서 호출용)
@mcp.custom_route("/api/rag/chat", methods=["POST"])
async def rag_chat(request: Request) -> JSONResponse:
    log.warning("=== rag_chat 진입 ===")
    try:
        body = await request.json()
        result = await run_rag_pipeline(
            session_id=body["session_id"],
            user_query=body["query"],
            lang=body.get("lang", "ko"),
            rag_limit=body.get("rag_limit", 5),
            history_limit=body.get("history_limit", 10),
            source_table=body.get("source_table"),
        )
        return JSONResponse(result)
    except KeyError as e:
        return JSONResponse({"detail": f"필수 파라미터 누락: {e}"}, status_code=422)
    except Exception as e:
        log.error(f"[rag_chat] error: {e}")
        return JSONResponse({"detail": str(e)}, status_code=500)


# Tool 1: search_rag
@mcp.tool()
async def search_rag(
    query: str,
    lang: str = "ko",
    limit: int = 5,
    source_table: str | None = None,
) -> list[dict]:
    """
    pgvector에서 질문과 유사한 문서를 검색합니다.

    Args:
        query:        검색할 질문 텍스트
        lang:         언어 (ko / en / ja)
        limit:        반환할 문서 수 (기본 5)
        source_table: 특정 테이블만 검색 (None이면 전체)
    """
    docs = await _search_rag(
        query=query, lang=lang, limit=limit, source_table=source_table
    )
    return [d.model_dump() for d in docs]


# Tool 2: chat_llm
@mcp.tool()
async def chat_llm(
    messages: list[dict],
    context: str = "",
) -> str:
    """
    Ollama qwen3:8b로 답변을 생성합니다.

    Args:
        messages: 대화 히스토리 [{"role": "user"/"assistant", "content": "..."}]
        context:  RAG에서 검색된 문서 내용 (system prompt에 주입)
    """
    chat_messages = [ChatMessage(**m) for m in messages]
    return await _chat_llm(messages=chat_messages, context=context)


# Tool 3: save_message
@mcp.tool()
async def save_message(
    session_id: str,
    role: str,
    content: str,
    lang: str = "ko",
    source_docs: list[dict] | None = None,
) -> dict:
    """
    채팅 메시지를 DB에 저장합니다.

    Args:
        session_id:  채팅 세션 UUID
        role:        'user' 또는 'assistant'
        content:     메시지 내용
        lang:        언어 (ko / en / ja)
        source_docs: 참조한 RAG 문서 목록
    """
    return await _save_message(
        session_id=session_id,
        role=role,
        content=content,
        lang=lang,
        source_docs=source_docs,
    )


# Tool 4: get_history
@mcp.tool()
async def get_history(
    session_id: str,
    limit: int = 10,
) -> list[dict]:
    """
    세션의 대화 히스토리를 조회합니다.
    Args:
        session_id: 채팅 세션 UUID
        limit:      가져올 최근 메시지 수 (기본 10)
    """
    messages = await _get_history(session_id=session_id, limit=limit)
    return [m.model_dump() for m in messages]


# 실행
if __name__ == "__main__":
    log.info(f"MCP Server 시작: {MCP_HOST}:{MCP_PORT}")
    mcp.run(transport="sse", host=MCP_HOST, port=MCP_PORT)
