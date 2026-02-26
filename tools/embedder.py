import httpx
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")


async def get_embedding(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]
