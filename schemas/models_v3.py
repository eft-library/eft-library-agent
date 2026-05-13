from typing import Any, Literal
from datetime import datetime

from pydantic import BaseModel, Field


Lang = Literal["ko", "en", "ja"]
ChunkType = Literal["identifier", "retrieval", "summary", "content", "relation", "guide"]
Domain = Literal["item", "quest", "map", "boss", "hideout", "trader", "information", "story"]


class RagChunkV3(BaseModel):
    domain: Domain
    entity_id: str
    chunk_id: str
    chunk_type: ChunkType
    lang: Lang = "ko"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    searchable: bool = True
    source_table: str | None = None
    source_updated_at: datetime | None = None


class RagDocumentV3(BaseModel):
    domain: str
    entity_id: str
    chunk_id: str
    chunk_type: str
    lang: str
    content: str
    metadata: dict[str, Any]
    similarity: float | None = None
    lexical_score: float | None = None
    trigram_score: float | None = None
    fused_score: float | None = None


class SearchRagInputV3(BaseModel):
    query: str
    lang: Lang = "ko"
    limit: int = 10
    domain: Domain | None = None


class ChatMessageV3(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class SaveMessageInputV3(BaseModel):
    session_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    lang: Lang = "ko"
    source_docs: list[dict[str, Any]] | None = None
