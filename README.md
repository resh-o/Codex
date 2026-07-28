# Codex — Codebase Q&A / RAG

Codex is a codebase Q&A / RAG tool, built in stages and designed to be
rigorously engineered and formally evaluated. **Implemented so far:** Stage 1
(AST chunking) and Stage 2 (embeddings + flat vector search).

| Stage | What | Status |
|-------|------|--------|
| 1 | Ingestion + AST chunking (tree-sitter) | ✅ |
| 2 | Embeddings (Gemini) + flat vector search (pgvector) | ✅ |
| 3 | Hybrid retrieval (vector + keyword, RRF fusion) | ⬜ |
| 4 | Citation enforcement | ⬜ |
| 5 | Formal eval suite | ⬜ |

Jump to [Stage 2](#stage-2--embeddings--flat-vector-search).

---

# Stage 1: Repo Ingestion & AST Chunking

## What this stage does

Given a public git repo URL, the pipeline:

1. **Clones** it (or pulls if already cloned) with GitPython and records the
   commit SHA.
2. **Walks** the tree for source files (`.py`, `.ts`, `.tsx`, `.js`, `.jsx`),
   pruning vendored/build/VCS directories.
3. **Chunks** each file with [tree-sitter](https://tree-sitter.github.io/) at
   function / method / class boundaries — not naive line splitting — producing
   `Chunk` objects with precise line spans, qualified names, parent links, and
   extracted docstrings.

Chunking rules:

- Functions, methods, and classes each become a chunk. **Nested functions and
  methods are chunked independently**, with `parent_name` / `qualified_name`
  set from the enclosing definition chain (e.g. `UserAuth.validate_token`).
- Arrow functions / function expressions assigned to a `const`/`let`/`var`
  (`const multiply = (a, b) => …`) are chunked as functions too.
- Each file also yields one `module` chunk for top-level code that isn't inside
  any definition (imports, constants, `if __name__ == "__main__"` guards).
- **Robustness:** a file that fails to parse (syntax error) or whose language
  isn't supported degrades to a single whole-file `module` chunk and logs a
  warning — it never crashes the run. (Stage 5's eval can then score AST-chunked
  vs fallback-chunked files separately.)

### The `Chunk` model

`repo_url`, `commit_sha`, `file_path`, `language`, `chunk_type`
(`function|method|class|module`), `name`, `qualified_name`, `parent_name`,
`start_line`, `end_line` (1-indexed, inclusive), `content`, `docstring`.
See `backend/app/ingestion/models.py`.

## Project layout

```
backend/
  app/
    ingestion/
      cloner.py    # clone/pull via GitPython, clean exceptions
      walker.py    # file discovery + filtering (configurable excludes)
      chunker.py   # tree-sitter AST parsing -> Chunk objects
      models.py    # Chunk data model
      __init__.py  # ingest_repo() orchestration
    main.py        # FastAPI app: POST /ingest
  tests/           # pytest: test_walker.py, test_chunker.py, fixtures/
  requirements.txt
  requirements-dev.txt
  Dockerfile
docker-compose.yml
```

## Run it locally (without Docker)

Requires Python 3.11+ (developed on 3.13).

```bash
cd backend
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash);  use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then call the endpoint:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/pallets/click"}'
```

You'll get a JSON summary: files processed, chunks produced, breakdown by
`chunk_type` and `language`, and a small sample of chunks (not the full dump).
Interactive docs are at `http://localhost:8000/docs`.

## Run it with Docker

```bash
docker compose up --build
# POST to http://localhost:8000/ingest as above
```

The compose file runs only the backend for now; a Postgres + pgvector service is
stubbed as a commented placeholder for Stage 2.

## Run the tests

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

`test_walker.py` verifies directory/extension filtering; `test_chunker.py`
verifies chunk counts, `chunk_type` / `name` / `qualified_name` / `parent_name`,
that line spans slice back to the exact chunk content, docstring extraction, and
the syntax-error fallback.

---

# Stage 2: Embeddings & Flat Vector Search

Takes the chunks from Stage 1, embeds them with Gemini, stores them in Postgres
with pgvector, and exposes flat (vector-only) cosine similarity search.

## Embedding model

- **Model:** `gemini-embedding-001` — the current GA embedding model on the
  Gemini Developer API (verified against Google's docs, July 2026). Configurable
  via `GEMINI_EMBEDDING_MODEL`.
- **Dimensionality:** **768** (`EMBEDDING_DIM`). The model supports 128–3072 via
  Matryoshka Representation Learning; Google recommends 768 / 1536 / 3072. 768 is
  the sweet spot here — far cheaper to store and search than the 3072 default
  while keeping most of the retrieval quality. **This must match the `vector(N)`
  column in `app/storage/schema.sql`.**
- Embeddings are **L2-normalized** in the client (required for
  `gemini-embedding-001` at non-3072 dims). Documents use task type
  `RETRIEVAL_DOCUMENT`, queries use `RETRIEVAL_QUERY` (asymmetric retrieval).
- Requests are **batched** (default 100/req) with **exponential-backoff retries**
  on 429/5xx/network errors.

## New pieces

```
app/embeddings/  gemini_client.py (retrying API wrapper) · embedder.py (batching)
app/storage/     schema.sql (pgvector table + HNSW index) · db.py (pooled conn) · repository.py (upsert + <=> search)
app/search/      vector_search.py (embed query → cosine search → snippets)
```

Data model in Postgres: the `chunks` table mirrors the `Chunk` fields plus `id`
(uuid), `embedding vector(768)`, and `created_at`, with a unique natural key on
`(repo_url, commit_sha, file_path, start_line, end_line)` so re-ingesting a
commit upserts instead of duplicating. Similarity uses pgvector's `<=>` cosine
operator, indexed with **HNSW** (`vector_cosine_ops`).

## Setup

1. **Gemini API key** — create one at <https://aistudio.google.com/apikey>.
2. **Database** — either the bundled local pgvector (via docker-compose) or a
   hosted Supabase project. For Supabase, create the project and run the schema:
   ```bash
   psql "$DATABASE_URL" -f backend/app/storage/schema.sql
   ```
   (The backend also applies `schema.sql` automatically on first `/embed` or
   `/search`, so this step is optional if the DB user may create extensions.)
3. **Env** — copy the example and fill it in:
   ```bash
   cp backend/.env.example backend/.env   # then edit GEMINI_API_KEY, DATABASE_URL
   ```
   `.env` is gitignored.

## Run it (Docker — backend + local pgvector)

```bash
cp backend/.env.example backend/.env   # set GEMINI_API_KEY
docker compose up --build
```

## Run it (local, without Docker)

```bash
cd backend && source .venv/Scripts/activate   # (.venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
export GEMINI_API_KEY=...   DATABASE_URL=postgresql://codex:codex@localhost:5432/codex
uvicorn app.main:app --reload
```

Then embed a repo and search it:

```bash
# Ingest + embed + store every chunk (idempotent per commit):
curl -X POST localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/pallets/click"}'

# Flat vector search:
curl -X POST localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how are command options parsed?", "top_k": 5}'
```

`/embed` returns a summary (chunks embedded, chunks skipped as already-stored,
failures, rows inserted/updated). `/search` returns ranked chunks with
similarity scores, `file_path`, line span, and a content snippet. Missing/invalid
Gemini or DB credentials return a clean `503`, not a stack trace.

## Run the Stage 2 tests

```bash
cd backend && pytest
```

- `test_embedder.py` — batching correctness and retry/backoff behavior, with the
  Gemini client fully mocked (no network).
- `test_vector_search.py` — search-service ordering contract against a fake
  repository. A real pgvector round-trip (upsert + `<=>` ordering + idempotent
  re-upsert) also runs **if** `TEST_DATABASE_URL` is set:
  ```bash
  TEST_DATABASE_URL=postgresql://codex:codex@localhost:5432/codex pytest tests/test_vector_search.py
  ```

## Deliberately *not* included yet

- **Stage 3** — hybrid retrieval (vector + keyword/BM25, RRF fusion).
- **Stage 4** — citation enforcement (structured, validated file/line citations).
- **Stage 5** — the formal eval suite (retrieval + citation metrics).
- Frontend, and any answering/generation logic.
- Vector-index tuning (HNSW `m`/`ef`, or IVFFlat lists) — correctness first.
