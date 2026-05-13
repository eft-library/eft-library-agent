
# 프로젝트 구조

mcp-server/
├── main.py              ← MCP Server 진입점 (SSE 방식으로 외부 오픈)
├── tools/
│   ├── embedder.py      ← bge-m3 임베딩 생성
│   ├── retriever.py     ← pgvector 유사 문서 검색
│   ├── llm.py           ← Ollama Qwen3 스트리밍 호출
│   └── history.py       ← 채팅 이력 저장/조회
├── services/
│   └── rag.py           ← RAG 파이프라인 조합
├── db/
│   └── connection.py    ← PostgreSQL 연결 (기존 DB 재사용)
└── schemas/
    └── models.py        ← Pydantic 모델

## 구축 순서

1. pgvector DDL 작업
   └── rag_documents 테이블, chat 테이블 생성

2. MCP Server 기본 세팅
   └── FastMCP SSE 방식으로 실행

3. 임베딩 Tool 구현
   └── bge-m3로 텍스트 → 벡터 변환

4. 기존 DB 데이터 배치 임베딩
   └── 기존 테이블 → rag_documents 적재 스크립트

5. Retriever Tool 구현
   └── pgvector cosine similarity 검색

6. LLM Tool 구현
   └── Ollama Qwen3 스트리밍 호출

7. RAG 파이프라인 조합
   └── 질문 → 임베딩 → 검색 → 프롬프트 → LLM

8. FastAPI chat.py 추가
   └── MCP Server 호출 → SSE로 Next.js에 전달

9. Next.js Chat UI 연결