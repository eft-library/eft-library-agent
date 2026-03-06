import asyncio
import asyncpg
import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

SOURCE_TABLE = "custom"
EMBED_MODEL = "bge-m3"

BASE_URL = os.getenv("OLLAMA_BASE_URL")
DB_URL = os.getenv("DATABASE_URL")

METADATA = json.dumps(
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
)

# identifier: 검색용 키워드 중심
IDENTIFIERS = {
    "ko": "타르코프 EFT Escape from Tarkov 게임 VPN 네트워크 최적화 핑 개선 패킷 손실 GearUP 추천",
    "en": "Tarkov EFT Escape from Tarkov Game VPN network optimization ping improvement packet loss GearUP recommendation",
    "ja": "タルコフ EFT Escape from Tarkov ゲームVPN ネットワーク最適化 PING改善 パケットロス GearUP おすすめ",
}

# content: 실제 안내 본문
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

DOCS = [
    {
        "source_id": "vpn-guide",
        "chunk_type": "identifier",
        "ref_id": "vpn-guide",
        "contents": IDENTIFIERS,
    },
    {
        "source_id": "vpn-guide_content",
        "chunk_type": "content",
        "ref_id": "vpn-guide",
        "contents": CONTENTS,
    },
]


async def embed_text(client: httpx.AsyncClient, text: str) -> list[float]:
    resp = await client.post(
        f"{BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text.strip()},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


async def upsert_document(
    pool: asyncpg.Pool,
    source_id: str,
    lang: str,
    content: str,
    embedding: list[float],
    chunk_type: str,
    ref_id: str,
):
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
                content    = EXCLUDED.content,
                embedding  = EXCLUDED.embedding,
                ref_type   = EXCLUDED.ref_type,
                ref_id     = EXCLUDED.ref_id,
                metadata   = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            SOURCE_TABLE,
            source_id,
            lang,
            content.strip(),
            embedding_str,
            chunk_type,
            "custom",
            ref_id,
            METADATA,
        )


async def process(client: httpx.AsyncClient, pool: asyncpg.Pool, doc: dict, lang: str):
    content = doc["contents"][lang]
    embedding = await embed_text(client, content)
    await upsert_document(
        pool,
        doc["source_id"],
        lang,
        content,
        embedding,
        doc["chunk_type"],
        doc["ref_id"],
    )
    print(f"  ✓ {doc['source_id']} [{lang}]")


async def main():
    async with httpx.AsyncClient() as client:
        async with asyncpg.create_pool(DB_URL) as pool:
            tasks = [
                process(client, pool, doc, lang)
                for doc in DOCS
                for lang in IDENTIFIERS.keys()
            ]
            await asyncio.gather(*tasks, return_exceptions=False)

    print("✅ done")


if __name__ == "__main__":
    asyncio.run(main())
