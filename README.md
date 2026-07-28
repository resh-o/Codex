# Codex — Stage 1: Repo Ingestion & AST Chunking

Codex is a codebase Q&A / RAG tool, built in stages and designed to be
rigorously engineered and formally evaluated. **This repo currently implements
Stage 1 only:** cloning a repository and splitting its source into structured,
AST-aware chunks that later stages will embed, index, and retrieve.

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

## Deliberately *not* included yet

These belong to later stages and are intentionally absent:

- **Stage 2** — embeddings (Gemini) and vector search (Supabase pgvector), plus
  any persistence. Stage 1 holds nothing in a database.
- **Stage 3** — hybrid retrieval (vector + keyword, RRF fusion).
- **Stage 4** — citation enforcement (structured, validated file/line citations).
- **Stage 5** — the formal eval suite (retrieval + citation metrics).
- Frontend, and any answering/retrieval logic.

The `Chunk` model and `ingest_repo()` function are the stable seam those stages
will plug into.
