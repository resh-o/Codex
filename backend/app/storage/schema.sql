-- Codex Stage 2 schema: pgvector-backed chunk store.
-- Apply against your Supabase/Postgres database, e.g.:
--   psql "$DATABASE_URL" -f app/storage/schema.sql
-- Safe to run repeatedly (idempotent).

-- pgvector provides the `vector` type and similarity operators.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Fields mirrored from the Stage-1 Chunk model.
    repo_url        text        NOT NULL,
    commit_sha      text        NOT NULL,
    file_path       text        NOT NULL,
    language        text        NOT NULL,
    chunk_type      text        NOT NULL,
    name            text        NOT NULL,
    qualified_name  text        NOT NULL,
    parent_name     text,
    start_line      integer     NOT NULL,
    end_line        integer     NOT NULL,
    content         text        NOT NULL,
    docstring       text,

    -- N MUST match Settings.embedding_dim / gemini-embedding-001 output dims.
    embedding       vector(768) NOT NULL,

    created_at      timestamptz NOT NULL DEFAULT now(),

    -- Re-ingesting the same commit must not duplicate chunks; this is the
    -- upsert conflict target used by repository.upsert_chunks().
    CONSTRAINT chunks_natural_key
        UNIQUE (repo_url, commit_sha, file_path, start_line, end_line)
);

-- Vector similarity index.
--
-- HNSW vs IVFFlat tradeoff:
--   * HNSW  -> better recall + query latency, works without a training step,
--              but slower to build and uses more memory.
--   * IVFFlat -> faster/cheaper to build, but needs data present before
--                building (to train the lists) and typically lower recall.
-- We pick HNSW for correctness-first quality; index parameter tuning
-- (m / ef_construction / ef_search, or IVFFlat lists) is deferred (Stage 5).
-- vector_cosine_ops matches the `<=>` cosine-distance operator used in queries.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Helps the optional per-repo filter in search_similar().
CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo_url, commit_sha);
