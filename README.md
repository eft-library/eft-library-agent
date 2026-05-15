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
- `RAG_V3_ANSWERABILITY_CHECK`
- `RAG_V3_ANSWERABILITY_MAX_CONTEXT_CHARS`
- `RAG_V3_AUTO_DOMAIN_FILTER`
- `RAG_V3_DOMAIN_ROUTE_BOOST`
- `WEB_FALLBACK_ENABLED`
- `WEB_SEARCH_PROVIDER`
- `WEB_SEARCH_API_KEY`
- `WEB_FALLBACK_DOMAINS`
- `WEB_SEARCH_LIMIT`
- `WEB_FALLBACK_STEAM_APP_ID`
- `WEB_FALLBACK_STEAM_COUNTRY`
- `WEB_FALLBACK_STEAM_LANG`

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
- automatic high-confidence domain filtering
- soft domain route boosting
- reranker 없음

예시:

```bash
./venv/bin/python -m tools.retriever_v3 "살레와 어디서 만들어" --domain item --limit 3
./venv/bin/python -m tools.retriever_v3 "프라퍼 2렙 바터" --domain trader --limit 3
./venv/bin/python -m tools.retriever_v3 "터미널 접근 스토리" --domain story --limit 3
./venv/bin/python -m tools.retriever_v3 "Smugglers 2026 어디 나와" --domain information --limit 3
```

## V3 Evaluation

대표 질문 세트로 retrieval 품질을 빠르게 확인합니다.
현재 기본 세트는 엔티티 검색과 관계형 질문을 포함합니다.
실제 사용자 질문에서 뽑은 대표 케이스를 포함하되, 욕설/인사/사이트 불만/최신 외부 정보처럼 정답 기준이 다른 질문은 별도 개선 대상으로 다룹니다.

포함하는 관계형 질문 예:

- 선행/후행 퀘스트
- 필요 아이템과 보상
- 구매/제작 해금
- 아이템 사용처와 보상 퀘스트
- 보스 스폰/드랍
- 은신처 업그레이드 요구사항
- 스토리 로드맵 분기

```bash
./venv/bin/python -m tools.eval_v3
```

By default, eval searches without forcing a domain.
This is closer to real user traffic and can expose routing or cross-domain noise.
The retriever still may infer a domain internally from high-confidence user intent, such as quest dependencies, boss drops, map names, item usage, hideout requirements, or story roadmap questions.

To run the older scoped regression mode, where each case uses its expected domain hint:

```bash
./venv/bin/python -m tools.eval_v3 --use-case-domain
```

Eval logs progress for each case by default.
Use `--quiet` to suppress progress logs, or `--verbose` for more detailed logs.

실제 LLM 답변 생성까지 포함해서 확인할 때:

```bash
./venv/bin/python -m tools.eval_v3 --with-answer
```

특정 케이스만 실행:

```bash
./venv/bin/python -m tools.eval_v3 --case item_craft_salewa
```

Known follow-up candidates from historical user prompts:

- key location intent: `bloody 키 위치`, `블러드키 위치`, `세관 빨간 창고 여는 열쇠 뭐야?`
- unlock/quest dependency intent: `상인 예거는 어떤 퀘스트를 완료해야 해금이 돼?`, `카파퀘 꼭 퀘스트 해야 풀려`
- external/current facts: `지금 진행중인 이벤트 알려줘`, `현재 사용 가능한 코드`, `시세 알려줘`
- site/meta support: `광고 좀 지워줘`, `로그인이 안되`, `홈페이지가 화이트로 바뀌었어요`
- off-topic/safety: food recipes, insults, prompt-injection style requests

## V3 Answer Pipeline

V3 답변 생성 경로:

- `tools/llm_v3.py`
- `services/rag_v3.py`
- API route: `POST /api/rag/v3/chat/stream`
- MCP tools:
  - `search_rag_v3`
  - `save_message_v3`
  - `get_history_v3`

Before answer generation, `services/rag_v3.py` runs an answerability guard.
This guard checks whether retrieved local documents actually contain enough information to answer the user question.
If local results are broad keyword noise or lack the requested fact, the pipeline uses web fallback instead of answering from unrelated chunks.

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

## Web Fallback

Game-related questions should use local V3 RAG first.
If local RAG returns no documents or clearly lacks the requested fact, web fallback can be added as a second stage.
External purchase, edition, discount, Steam, or real-money price questions should prefer web fallback even if local RAG returns weak item matches.

Recommended approach:

- keep local RAG answer as the primary source
- call web search only on low-confidence or empty local results
- restrict web search to approved domains such as official Tarkov pages, Tarkov.dev, or the EFT Wiki
- return web-sourced answers with source URLs
- do not ingest web results into RAG tables unless explicitly requested

Default allowed domains:

- `store.steampowered.com/app/3932890`
- `gall.dcinside.com/mgallery/board`
- `tarkov.dev`
- `escapefromtarkov.fandom.com`

DCInside is a Korean community source and should be treated as lower-confidence reference.
Tarkov.dev and Fandom are useful for structured or cross-check information, but are mostly English.

Implementation options:

- a search API key/provider such as Tavily, Brave Search, SerpAPI, or Google Custom Search
- or a site-specific API/search endpoint if the target site provides one
- or `WEB_SEARCH_PROVIDER=public` for best-effort direct lookup against public community pages

Avoid brittle HTML scraping as the default production path.

Example `.env`:

```dotenv
WEB_FALLBACK_ENABLED=true
WEB_SEARCH_PROVIDER=public
WEB_SEARCH_LIMIT=5
RAG_V3_ANSWERABILITY_CHECK=true
RAG_V3_ANSWERABILITY_MAX_CONTEXT_CHARS=6000
WEB_FALLBACK_DOMAINS=store.steampowered.com/app/3932890,gall.dcinside.com/mgallery/board,tarkov.dev,escapefromtarkov.fandom.com
WEB_FALLBACK_STEAM_APP_ID=3932890
WEB_FALLBACK_STEAM_COUNTRY=KR
WEB_FALLBACK_STEAM_LANG=korean
```

`public` provider does not require an API key, but it is best-effort:

- Steam purchase/price questions use the public Steam Store appdetails API first.
- DCInside currently returns public HTML and can be parsed.
- ArcaLive is intentionally excluded because server-side requests are commonly blocked by Cloudflare challenge.
- Layout changes or anti-bot rules can break direct parsing.

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
