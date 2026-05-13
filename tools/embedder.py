import httpx
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")


async def get_embedding(text: str) -> list[float]:
    if not OLLAMA_BASE_URL:
        raise RuntimeError("OLLAMA_BASE_URL is not set")
    if not EMBED_MODEL:
        raise RuntimeError("OLLAMA_EMBED_MODEL is not set")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
