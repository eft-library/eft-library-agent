# eft-library-agent

타르코프 도서관 Agent Chat / RAG 서버입니다.

현재는 normalized platform schema 기반의 V3 RAG를 별도 경로로 구축 중입니다.

## Setup

```bash
python -m venv venv
./venv/bin/pip install asyncpg httpx beautifulsoup4 python-dotenv fastmcp pydantic
```

필요한 주요 환경 변수는 `.env`에서 관리합니다.

- `DATABASE_URL`
- `OLLAMA_BASE_URL`
- `OLLAMA_CHAT_MODEL` 또는 `OLLAMA_LLM_MODEL`
- `OLLAMA_EMBED_MODEL`
- `RAG_LIMIT`
- `NUM_CTX`
- `RAG_TRGM_THRESHOLD`
- `RAG_RRF_K`
- `RAG_V3_MAX_CONTEXT_CHARS`

## V3 Storage

Agent-owned SQL은 `sql/` 아래에 둡니다.

- `sql/001_agent_core_v3.sql`
  - `rag_documents_v3`
  - `chat_messages_v3`
  - `vector`, `pg_trgm` 기반 인덱스
- `sql/002_rag_entity_aliases_v3.sql`
  - `rag_entity_aliases_v3`
  - LLM/manual/import/autocomplete alias 후보와 승인 상태 관리

SQL은 수동 적용 후 Python 빌더를 실행합니다.

## V3 Models

현재 사용하는 로컬 모델:

- Chat: `qwen3.5:4b`
- Embedding: `bge-m3:latest`

Qwen chat 호출은 `think: false`를 사용합니다.

## V3 Builders

도메인별 RAG 빌더:

- `rag_builders_v3/item.py`
- `rag_builders_v3/quest.py`
- `rag_builders_v3/map.py`
- `rag_builders_v3/boss.py`
- `rag_builders_v3/hideout.py`
- `rag_builders_v3/trader.py`
- `rag_builders_v3/information.py`
- `rag_builders_v3/story.py`

예시 실행:

```bash
./venv/bin/python -m rag_builders_v3.item --langs ko,en,ja
./venv/bin/python -m rag_builders_v3.quest --langs ko,en,ja
./venv/bin/python -m rag_builders_v3.map --langs ko,en,ja
./venv/bin/python -m rag_builders_v3.boss --langs ko,en,ja
./venv/bin/python -m rag_builders_v3.hideout --langs ko,en,ja
./venv/bin/python -m rag_builders_v3.trader --langs ko,en,ja
./venv/bin/python -m rag_builders_v3.information --langs ko,en,ja
./venv/bin/python -m rag_builders_v3.story --langs ko,en,ja
```

먼저 확인만 할 때:

```bash
./venv/bin/python -m rag_builders_v3.item --limit 1 --langs ko --dry-run
```

## Alias Workflow

아이템 alias는 데이터 기반으로 관리합니다.

1. LLM으로 후보를 생성해 `pending`으로 적재합니다.
2. 사람이 검수해 좋은 것만 `approved`로 변경합니다.
3. RAG 빌더는 `approved` alias만 chunk에 포함합니다.

후보 생성:

```bash
./venv/bin/python -m rag_alias_v3.item_alias_generator --lang ko --dry-run --limit 10
./venv/bin/python -m rag_alias_v3.item_alias_generator --lang ko
```

검수 후 아이템 빌더를 다시 실행해야 approved alias가 RAG chunk에 반영됩니다.

## V3 Retrieval

V3 검색은 `tools/retriever_v3.py`를 사용합니다.

검색 방식:

- vector search
- PostgreSQL `tsvector`
- PostgreSQL `pg_trgm`
- reciprocal rank fusion
- reranker 없음

예시:

```bash
./venv/bin/python -m tools.retriever_v3 "살레와 어디서 만들어" --domain item --limit 3
./venv/bin/python -m tools.retriever_v3 "프라퍼 2렙 바터" --domain trader --limit 3
./venv/bin/python -m tools.retriever_v3 "터미널 접근 스토리" --domain story --limit 3
./venv/bin/python -m tools.retriever_v3 "Smugglers 2026 어디 나와" --domain information --limit 3
```

## V3 Answer Pipeline

V3 답변 생성 경로:

- `tools/llm_v3.py`
- `services/rag_v3.py`
- API route: `POST /api/rag/v3/chat/stream`
- MCP tools:
  - `search_rag_v3`
  - `save_message_v3`
  - `get_history_v3`

스트리밍 API 요청 예시:

```bash
curl -N -X POST http://localhost:15000/api/rag/v3/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "00000000-0000-0000-0000-000000000001",
    "query": "살레와 어디서 만들어?",
    "lang": "ko",
    "domain": "item",
    "rag_limit": 3,
    "history_limit": 3
  }'
```

Backend FastAPI 연동:

- backend repo: `/Users/sun-yeob/Desktop/Work/my_git/eftLibrary/eft-library-back`
- backend route: `POST /api/chat/stream`
- backend service proxies to agent route: `POST /api/rag/v3/chat/stream`
- required backend env: `MCP_SERVER_URL=http://<agent-host>:15000`

Backend 요청 예시:

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "00000000-0000-0000-0000-000000000001",
    "query": "살레와 어디서 만들어?",
    "lang": "ko",
    "domain": "item",
    "rag_limit": 3,
    "history_limit": 3
  }'
```

주의:

- `domain`은 생략하거나 `item`, `quest`, `map`, `boss`, `hideout`, `trader`, `information`, `story` 중 하나만 보냅니다.
- Swagger 기본 예시값인 `"string"`을 그대로 보내면 안 됩니다.
- `rag_limit`, `history_limit`는 생략하거나 1 이상의 값을 보냅니다. `0`은 기본값 사용으로 처리합니다.

If backend logs show `incomplete chunked read`, the agent stream closed before sending a complete SSE response.
Both backend and agent stream layers should emit an SSE `error` event followed by `done` so FastAPI does not raise a traceback to the client.

## Run Server

```bash
./venv/bin/python main.py
```

서버는 `MCP_HOST`, `MCP_PORT` 값을 사용합니다.

운영 재시작:

```bash
./restart.sh restart
./restart.sh status
curl http://127.0.0.1:15000/api/rag/v3/health
```

`restart.sh`는 `main.py`를 재시작하며, V3 health route가 응답하는지 확인합니다.
V3 코드를 배포한 뒤에는 반드시 agent 서버를 재시작해야 backend `/api/chat/stream`이 새 `/api/rag/v3/chat/stream` 경로를 사용할 수 있습니다.
