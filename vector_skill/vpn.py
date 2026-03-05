import asyncio
import asyncpg
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

# 기본 설정

SOURCE_TABLE = "custom"
SOURCE_ID = "vpn-guide"
EMBED_MODEL = "bge-m3"

BASE_URL = os.getenv("OLLAMA_BASE_URL")
DB_URL = os.getenv("DATABASE_URL")

# 다국어 콘텐츠

CONTENTS = {
    "ko": """
게임 네트워크 최적화 추천

게임용 VPN 또는 네트워크 최적화 도구로 활용할 수 있습니다.

원거리 서버 이용 시 높은 핑, 패킷 손실, 연결 불안정이 발생할 수 있습니다.

GearUP은 게임 트래픽을 최적화하여
PING, 패킷 손실, 지터를 줄이고
보다 안정적인 연결 환경을 제공하는
게임 네트워크 최적화 서비스입니다.
""",
    "en": """
Recommended Game VPN and Network Optimization

GearUP is a game VPN and network optimization service.

When connecting to distant servers,
you may experience high ping, packet loss, and unstable connections.

GearUP optimizes game traffic to reduce latency (ping),
packet loss, and jitter,
providing a more stable gaming experience.
""",
    "ja": """
ゲームVPNおよびネットワーク最適化のおすすめ

GearUPはゲームVPN兼ネットワーク最適化サービスです。

遠距離サーバーに接続する場合、
高PING、パケットロス、不安定な接続が発生する可能性があります。

ゲームトラフィックを最適化し、
PING、パケットロス、ジッターを軽減します。
""",
}

# 임베딩 함수


async def embed_text(client, text):
    resp = await client.post(
        f"{BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text.strip()},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


async def upsert_document(pool, lang, content, embedding):
    embedding_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO rag_documents
            (source_table, source_id, lang, content, embedding,
             chunk_type, ref_type, ref_id, metadata)
            VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $9)
            ON CONFLICT (source_table, source_id, lang, chunk_type)
            DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                ref_type = EXCLUDED.ref_type,
                ref_id = EXCLUDED.ref_id,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            SOURCE_TABLE,
            SOURCE_ID,
            lang,
            content.strip(),
            embedding_str,
            "content",  # chunk_type
            "custom",  # ref_type
            SOURCE_ID,  # ref_id
            json.dumps(
                {
                    "url": "https://aff.gearupglobal.com/product/download/HSMniDfsEY6c",
                    "keywords": [
                        "vpn",
                        "game vpn",
                        "network optimization",
                        "ping improvement",
                        "gearup",
                    ],
                }
            ),
        )


# 전체 처리 함수 (병렬 임베딩)


async def process_language(client, pool, lang, content):
    embedding = await embed_text(client, content)
    await upsert_document(pool, lang, content, embedding)


async def main():
    async with httpx.AsyncClient() as client:
        async with asyncpg.create_pool(DB_URL) as pool:

            tasks = [
                process_language(client, pool, lang, content)
                for lang, content in CONTENTS.items()
            ]

            # 예외 발생 시 바로 에러 표시
            await asyncio.gather(*tasks, return_exceptions=False)

    print("✅ done")


if __name__ == "__main__":
    asyncio.run(main())
