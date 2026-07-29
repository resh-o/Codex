"""FastAPI app: Stage 1 ingestion, Stage 2 embeddings, Stage 3 hybrid search.

Endpoints
---------
* ``POST /ingest`` — Stage 1: clone + chunk, return a summary (no persistence).
* ``POST /embed``  — Stage 2: ingest + embed + upsert into pgvector.
* ``POST /search`` — Stage 3: hybrid (vector + keyword, RRF-fused) search, with
  ``mode`` to fall back to either retriever on its own.

Credential/connection problems (missing Gemini key, unreachable DB) surface as
clean HTTP errors rather than leaking raw tracebacks to the client.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import ConfigError, get_settings
from .embeddings import Embedder, EmbeddingError, GeminiClient
from .ingestion import Chunk, CloneError, ingest_repo
from .search import (
    HybridSearchService,
    KeywordSearchService,
    VectorSearchService,
)
from .storage import (
    StorageError,
    apply_schema,
    get_existing_keys,
    get_pool,
    upsert_chunks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Codex API",
    version="0.3.0",
    description=(
        "Stage 1 (AST chunking) + Stage 2 (embeddings) + Stage 3 (hybrid "
        "vector/keyword retrieval with RRF fusion)."
    ),
)

SAMPLE_SIZE = 5


# --------------------------------------------------------------------------- #
# Lazily-built, process-wide singletons for the embedding + search stack.
# --------------------------------------------------------------------------- #

# Reentrant: _get_search_service() builds the embedder via _get_embedder()
# while holding the lock, so the same thread must be able to re-acquire it.
_lock = threading.RLock()
_embedder: Embedder | None = None
_search_service: VectorSearchService | None = None
_schema_ready = False


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                settings = get_settings()
                client = GeminiClient(settings)
                _embedder = Embedder(client, batch_size=settings.embedding_batch_size)
    return _embedder


def _get_search_service() -> VectorSearchService:
    global _search_service
    if _search_service is None:
        with _lock:
            if _search_service is None:
                _search_service = VectorSearchService(_get_embedder())
    return _search_service


def _ensure_storage_ready() -> None:
    """Open the pool and apply the schema once (idempotent)."""
    global _schema_ready
    get_pool()  # raises ConfigError/StorageError with a clean message
    if not _schema_ready:
        with _lock:
            if not _schema_ready:
                apply_schema()
                _schema_ready = True


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class IngestRequest(BaseModel):
    repo_url: str = Field(..., description="A cloneable git URL (https or ssh).")


class ChunkSample(BaseModel):
    file_path: str
    language: str
    chunk_type: str
    name: str
    qualified_name: str
    parent_name: str | None
    start_line: int
    end_line: int
    docstring: str | None


class IngestResponse(BaseModel):
    repo_url: str
    commit_sha: str
    files_processed: int
    chunks_produced: int
    chunks_by_type: dict[str, int]
    chunks_by_language: dict[str, int]
    sample_chunks: list[ChunkSample]


class EmbedRequest(BaseModel):
    repo_url: str = Field(..., description="A cloneable git URL (https or ssh).")


class EmbedResponse(BaseModel):
    repo_url: str
    commit_sha: str
    files_processed: int
    chunks_total: int
    chunks_embedded: int
    chunks_skipped: int
    embedding_failures: int
    rows_inserted: int
    rows_updated: int
    embedding_model: str
    embedding_dim: int


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural-language or code query.")
    repo_url: str | None = Field(None, description="Optional: restrict to one repo.")
    top_k: int = Field(10, ge=1, le=100)


class SearchResultModel(BaseModel):
    repo_url: str
    file_path: str
    language: str
    chunk_type: str
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    similarity: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    repo_url: str | None
    count: int
    results: list[SearchResultModel]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _sample(chunk: Chunk) -> ChunkSample:
    return ChunkSample(
        file_path=chunk.file_path,
        language=chunk.language,
        chunk_type=chunk.chunk_type,
        name=chunk.name,
        qualified_name=chunk.qualified_name,
        parent_name=chunk.parent_name,
        start_line=chunk.start_line,
        end_line=chunk.end_line,
        docstring=chunk.docstring,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Stage 1: clone a repo, chunk it, return a summary (no persistence)."""
    try:
        result = ingest_repo(request.repo_url)
    except CloneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Ingestion failed for %s", request.repo_url)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    chunks = result.chunks
    by_type = Counter(c.chunk_type for c in chunks)
    by_language = Counter(c.language for c in chunks)
    sample = _pick_sample(chunks)
    return IngestResponse(
        repo_url=result.repo_url,
        commit_sha=result.commit_sha,
        files_processed=len(result.files),
        chunks_produced=len(chunks),
        chunks_by_type=dict(sorted(by_type.items())),
        chunks_by_language=dict(sorted(by_language.items())),
        sample_chunks=[_sample(c) for c in sample],
    )


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    """Stage 2: ingest, embed new chunks, and upsert them into pgvector."""
    settings = get_settings()
    try:
        embedder = _get_embedder()
        _ensure_storage_ready()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = ingest_repo(request.repo_url)
    except CloneError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks = result.chunks
    # Skip chunks already stored for this exact commit (idempotent re-runs).
    try:
        existing = get_existing_keys(result.repo_url, result.commit_sha)
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    to_embed = [
        c
        for c in chunks
        if (c.file_path, c.start_line, c.end_line) not in existing
    ]

    try:
        embedded, stats = embedder.embed_chunks(to_embed)
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc

    # If there was work to do and every batch failed, that's an error condition
    # (e.g. an invalid key or the API being down), not a 200 with a sad summary.
    if to_embed and stats.chunks_embedded == 0 and stats.failures > 0:
        detail = stats.errors[0] if stats.errors else "all embedding batches failed"
        raise HTTPException(status_code=502, detail=f"Embedding failed: {detail}")

    try:
        upserted = upsert_chunks(embedded)
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return EmbedResponse(
        repo_url=result.repo_url,
        commit_sha=result.commit_sha,
        files_processed=len(result.files),
        chunks_total=len(chunks),
        chunks_embedded=stats.chunks_embedded,
        chunks_skipped=len(chunks) - len(to_embed),
        embedding_failures=stats.failures,
        rows_inserted=upserted.inserted,
        rows_updated=upserted.updated,
        embedding_model=settings.embedding_model,
        embedding_dim=settings.embedding_dim,
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Stage 2: flat cosine similarity search over stored chunks."""
    try:
        service = _get_search_service()
        _ensure_storage_ready()
    except ConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        results = service.search(
            request.query, top_k=request.top_k, repo_url=request.repo_url
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (ConfigError, EmbeddingError) as exc:
        raise HTTPException(status_code=502, detail=f"Query embedding failed: {exc}") from exc
    except StorageError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SearchResponse(
        query=request.query,
        repo_url=request.repo_url,
        count=len(results),
        results=[SearchResultModel(**vars(r)) for r in results],
    )


def _pick_sample(chunks: list[Chunk]) -> list[Chunk]:
    """Pick up to SAMPLE_SIZE chunks, favouring variety across chunk types."""
    if len(chunks) <= SAMPLE_SIZE:
        return chunks
    seen_types: set[str] = set()
    diverse: list[Chunk] = []
    for chunk in chunks:
        if chunk.chunk_type not in seen_types:
            seen_types.add(chunk.chunk_type)
            diverse.append(chunk)
        if len(diverse) >= SAMPLE_SIZE:
            break
    for chunk in chunks:
        if len(diverse) >= SAMPLE_SIZE:
            break
        if chunk not in diverse:
            diverse.append(chunk)
    return diverse[:SAMPLE_SIZE]
