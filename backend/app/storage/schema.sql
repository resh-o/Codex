-- Codex schema: pgvector-backed chunk store (Stage 2) + full-text search
-- (Stage 3).  Apply against your Supabase/Postgres database, e.g.:
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


-- =========================================================================== --
-- Stage 3 migration: keyword / full-text search
--
-- Written as an additive ALTER rather than a rewrite of the CREATE TABLE above,
-- because Stage 2 databases already hold embedded chunks. Adding the column
-- rewrites the table once to backfill it -- but it is derived purely from
-- columns that are already there, so **no re-ingestion or re-embedding is
-- needed**. Existing `embedding` values are untouched.
-- =========================================================================== --

-- A *generated* (always-in-sync) tsvector, so keyword search can never drift
-- from `content` the way a trigger-maintained or app-maintained column can.
-- The expression must be IMMUTABLE, hence the explicit `'english'::regconfig`
-- (the one-argument to_tsvector() depends on default_text_search_config and is
-- only STABLE, so it is not allowed here).
--
-- Weighting, highest first, via setweight():
--   A  qualified_name -- an exact hit on `UserAuth.validate_token` is about as
--                       strong a relevance signal as this corpus offers.
--   B  name           -- the bare symbol, for queries that omit the class path.
--   C  docstring      -- prose written specifically to describe this chunk.
--   D  content        -- the body; the default (lowest) weight, since a token
--                       appearing somewhere in a 60-line function says little.
-- ts_rank_cd()'s default weight vector is {D,C,B,A} = {0.1, 0.2, 0.4, 1.0},
-- i.e. a qualified_name hit counts 10x a body hit. We keep those defaults --
-- tuning them belongs after Stage 5 produces eval numbers.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english'::regconfig, coalesce(qualified_name, '')), 'A') ||
        setweight(to_tsvector('english'::regconfig, coalesce(name, '')),           'B') ||
        setweight(to_tsvector('english'::regconfig, coalesce(docstring, '')),      'C') ||
        setweight(to_tsvector('english'::regconfig, coalesce(content, '')),        'D')
    ) STORED;

-- GIN is the right index for @@ lookups on a stored tsvector: slower to build
-- and update than GiST, but far faster and exact for search, and this column
-- only changes when a chunk is re-upserted.
CREATE INDEX IF NOT EXISTS chunks_search_vector_gin
    ON chunks USING gin (search_vector);
