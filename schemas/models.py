from pydantic import BaseModel
from typing import Any


class SearchRagInput(BaseModel):
    query: str
    lang: str = "ko"
    limit: int = 5
    source_table: str | None = None


class RagDocument(BaseModel):
    source_table: str
    source_id: str
    lang: str
    content: str
    metadata: dict[str, Any]
    similarity: float


class ChatMessage(BaseModel):
    role: str  # 'user' | 'assistant'
    content: str


class ChatLlmInput(BaseModel):
    messages: list[ChatMessage]
    context: str = ""


class SaveMessageInput(BaseModel):
    session_id: str
    role: str
    content: str
    lang: str = "ko"
    source_docs: list[dict[str, Any]] | None = None


class GetHistoryInput(BaseModel):
    session_id: str
    limit: int = 10
