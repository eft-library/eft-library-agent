CREATE TABLE IF NOT EXISTS rag_entity_aliases_v3 (
    domain text NOT NULL,
    entity_id text NOT NULL,
    lang text NOT NULL,
    alias text NOT NULL,
    source text NOT NULL,
    status text NOT NULL DEFAULT 'pending',
    confidence numeric,
    note text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT rag_entity_aliases_v3_lang_check CHECK (lang IN ('ko', 'en', 'ja')),
    CONSTRAINT rag_entity_aliases_v3_status_check CHECK (
        status IN ('pending', 'approved', 'rejected')
    ),
    CONSTRAINT rag_entity_aliases_v3_source_check CHECK (
        source IN ('manual', 'llm', 'autocomplete', 'import')
    ),
    CONSTRAINT rag_entity_aliases_v3_non_empty_alias CHECK (length(trim(alias)) > 0),
    CONSTRAINT rag_entity_aliases_v3_unique_alias UNIQUE (domain, entity_id, lang, alias)
);

CREATE INDEX IF NOT EXISTS idx_rag_entity_aliases_v3_lookup
    ON rag_entity_aliases_v3 (domain, entity_id, lang, status);

CREATE INDEX IF NOT EXISTS idx_rag_entity_aliases_v3_alias_trgm
    ON rag_entity_aliases_v3
    USING gin (alias gin_trgm_ops);

COMMENT ON TABLE rag_entity_aliases_v3 IS 'Agent-owned entity aliases for retrieval. LLM-generated aliases should be stored as pending candidates and only approved aliases should be used by RAG builders.';
COMMENT ON COLUMN rag_entity_aliases_v3.domain IS 'Stable RAG domain such as item, quest, map, boss, hideout, trader, information, or story.';
COMMENT ON COLUMN rag_entity_aliases_v3.entity_id IS 'Primary entity id from platform_db.sql.';
COMMENT ON COLUMN rag_entity_aliases_v3.alias IS 'User-facing search alias, abbreviation, transliteration, or community term.';
COMMENT ON COLUMN rag_entity_aliases_v3.source IS 'Alias source: manual, llm, autocomplete, or import.';
COMMENT ON COLUMN rag_entity_aliases_v3.status IS 'Only approved aliases are used by RAG builders.';

-- Example:
-- INSERT INTO rag_entity_aliases_v3 (domain, entity_id, lang, alias, source, status, confidence, note)
-- VALUES ('item', '544fb45d4bdc2dee738b4568', 'ko', '살레와', 'manual', 'approved', 1.0, 'Common Korean EFT search term')
-- ON CONFLICT (domain, entity_id, lang, alias)
-- DO UPDATE SET
--     source = EXCLUDED.source,
--     status = EXCLUDED.status,
--     confidence = EXCLUDED.confidence,
--     note = EXCLUDED.note,
--     updated_at = now();
