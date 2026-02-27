from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.rag import run_rag_pipeline
import logging

log = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    query: str
    lang: str = "ko"
    rag_limit: int = 5
    history_limit: int = 10
    source_table: str | None = None


class ChatResponse(BaseModel):
    answer: str
    docs: list[dict]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await run_rag_pipeline(
            session_id=req.session_id,
            user_query=req.query,
            lang=req.lang,
            rag_limit=req.rag_limit,
            history_limit=req.history_limit,
            source_table=req.source_table,
        )
        return ChatResponse(answer=result["answer"], docs=result["docs"])
    except Exception as e:
        log.error(f"[chat] error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
