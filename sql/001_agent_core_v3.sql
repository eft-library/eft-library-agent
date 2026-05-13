CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS rag_documents_v3 (
    id bigserial PRIMARY KEY,
    domain text NOT NULL,
    entity_id text NOT NULL,
    chunk_id text NOT NULL,
    chunk_type text NOT NULL,
    lang text NOT NULL,
    content text NOT NULL,
    embedding vector(1024) NOT NULL,
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('simple', coalesce(content, ''))
    ) STORED,
    searchable boolean NOT NULL DEFAULT true,
    source_table text,
    source_updated_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT rag_documents_v3_lang_check CHECK (lang IN ('ko', 'en', 'ja')),
    CONSTRAINT rag_documents_v3_chunk_type_check CHECK (
        chunk_type IN ('identifier', 'retrieval', 'summary', 'content', 'relation', 'guide')
    ),
    CONSTRAINT rag_documents_v3_unique_chunk UNIQUE (domain, entity_id, chunk_id, lang)
);

CREATE INDEX IF NOT EXISTS idx_rag_documents_v3_entity
    ON rag_documents_v3 (domain, entity_id)
    WHERE is_active;

CREATE INDEX IF NOT EXISTS idx_rag_documents_v3_searchable
    ON rag_documents_v3 (lang, domain, chunk_type)
    WHERE is_active AND searchable;

CREATE INDEX IF NOT EXISTS idx_rag_documents_v3_search_vector
    ON rag_documents_v3
    USING gin (search_vector)
    WHERE is_active AND searchable;

CREATE INDEX IF NOT EXISTS idx_rag_documents_v3_content_trgm
    ON rag_documents_v3
    USING gin (content gin_trgm_ops)
    WHERE is_active AND searchable;

CREATE INDEX IF NOT EXISTS idx_rag_documents_v3_embedding
    ON rag_documents_v3
    USING hnsw (embedding vector_cosine_ops)
    WHERE is_active AND searchable;

CREATE INDEX IF NOT EXISTS idx_rag_documents_v3_metadata
    ON rag_documents_v3
    USING gin (metadata);

CREATE TABLE IF NOT EXISTS chat_messages_v3 (
    id bigserial PRIMARY KEY,
    session_id uuid NOT NULL,
    role text NOT NULL,
    content text NOT NULL,
    lang text NOT NULL DEFAULT 'ko',
    source_docs jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT chat_messages_v3_role_check CHECK (role IN ('user', 'assistant', 'system')),
    CONSTRAINT chat_messages_v3_lang_check CHECK (lang IN ('ko', 'en', 'ja'))
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_v3_session_created
    ON chat_messages_v3 (session_id, created_at DESC);

COMMENT ON TABLE rag_documents_v3 IS 'Agent-owned V3 RAG chunks built from the normalized platform schema.';
COMMENT ON COLUMN rag_documents_v3.domain IS 'Stable RAG domain such as item, quest, map, boss, hideout, trader, information, or story.';
COMMENT ON COLUMN rag_documents_v3.entity_id IS 'Primary entity id from platform_db.sql.';
COMMENT ON COLUMN rag_documents_v3.chunk_id IS 'Stable chunk id within domain/entity/lang.';
COMMENT ON COLUMN rag_documents_v3.chunk_type IS 'Chunk purpose: identifier, retrieval, summary, content, relation, or guide.';
COMMENT ON COLUMN rag_documents_v3.searchable IS 'Whether the chunk participates in first-stage retrieval.';
COMMENT ON COLUMN rag_documents_v3.source_table IS 'Primary platform table used to build this chunk, when useful for debugging.';
COMMENT ON COLUMN rag_documents_v3.metadata IS 'Structured metadata used for source display, relation handling, and answer composition.';

COMMENT ON TABLE chat_messages_v3 IS 'Agent-owned V3 chat history table.';
