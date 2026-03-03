import asyncio
import asyncpg
import httpx
import json
from dotenv import load_dotenv
import os

load_dotenv()

content_ko = """VPN 추천
타르코프 접속은 VPN 사용을 권장합니다.
타르코프 도서관에서 추천하는 기어업 VPN: https://...
"""


async def main():
    embedding_resp = await httpx.AsyncClient().post(
        f"{os.getenv('OLLAMA_BASE_URL')}/api/embed",
        json={"model": "bge-m3", "input": content_ko},
        timeout=30.0,
    )
    embedding = embedding_resp.json()["embeddings"][0]
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    await conn.execute(
        """
        INSERT INTO rag_documents (source_table, source_id, lang, content, embedding, metadata)
        VALUES ($1, $2, $3, $4, $5::vector, $6)
        ON CONFLICT (source_table, source_id, lang) DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata,
            updated_at = NOW()
    """,
        "custom",
        "vpn-guide",
        "ko",
        content_ko,
        embedding_str,
        json.dumps({"content_type": "custom", "url": "https://vpn.eftlibrary.com"}),
    )
    await conn.close()
    print("done")


asyncio.run(main())
